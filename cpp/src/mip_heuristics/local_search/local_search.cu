/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "lagrangian.cuh"
#include "local_search.cuh"

#include <cuopt/error.hpp>

#include <branch_and_bound/branch_and_bound.hpp>
#include <mip_heuristics/diversity/diversity_manager.cuh>
#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/relaxed_lp/relaxed_lp.cuh>
#include <mip_heuristics/utils.cuh>
#include <utilities/seed_generator.cuh>
#include <utilities/timer.hpp>

#include <mip_heuristics/feasibility_jump/fj_cpu.cuh>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
local_search_t<i_t, f_t>::local_search_t(mip_solver_context_t<i_t, f_t>& context_,
                                         rmm::device_uvector<f_t>& lp_optimal_solution_)
  : context(context_),
    lp_optimal_solution(lp_optimal_solution_),
    fj_sol_on_lp_opt(context.problem_ptr->n_variables,
                     context.problem_ptr->handle_ptr->get_stream()),
    fj(context),
    // fj_tree(fj),
    constraint_prop(context),
    line_segment_search(fj, constraint_prop),
    fp(context,
       fj,
       // fj_tree,
       constraint_prop,
       line_segment_search,
       lp_optimal_solution_),
    rng(cuopt::seed_generator::get_seed()),
    problem_with_objective_cut(*context.problem_ptr, context.problem_ptr->handle_ptr)
{
  const int n_cpufj = context.settings.heuristic_params.num_cpufj_threads;
  ls_cpu_fj.resize(n_cpufj);
  // Run a few scratch climbers instead of a single one: they are randomized, so more of them means
  // more independent chances of hitting a first feasible solution. Kept small (and bounded by the
  // configured thread count) so they do not starve the rest of the heuristics.
  scratch_cpu_fj.resize(std::clamp(n_cpufj, 2, 4));
  fj.settings.n_of_minimums_for_exit = context.settings.heuristic_params.n_of_minimums_for_exit;
}

template <typename i_t, typename f_t>
void local_search_t<i_t, f_t>::start_cpufj_scratch_threads(population_t<i_t, f_t>& population)
{
  // TODO: Find a way to enable this in low core count scenarios
  if (omp_get_num_threads() < CUOPT_MIP_FJ_REQUIRED_THREAD_COUNT) return;

  pop_ptr = &population;
  std::vector<f_t> default_weights(context.problem_ptr->n_constraints, 1.);

  solution_t<i_t, f_t> solution(*context.problem_ptr);
  thrust::fill(solution.handle_ptr->get_thrust_policy(),
               solution.assignment.begin(),
               solution.assignment.end(),
               0.0);
  solution.clamp_within_bounds();
  // All climbers start from the same clamped all-zero point and get their diversity from the
  // randomize flag, rather than from randomized starting assignments: starting from zero is a much
  // better base for the FJ weights, and the randomized climbers diverge quickly anyway.
  for (size_t i = 0; i < scratch_cpu_fj.size(); ++i) {
    auto& climber = scratch_cpu_fj[i];
    climber       = fj.create_cpu_climber(solution,
                                    default_weights,
                                    default_weights,
                                    0.,
                                    context.preempt_heuristic_solver_,
                                    fj_settings_t{},
                                    true);
    climber->log_prefix = "******* scratch " + std::to_string(i) + ": ";
    climber->improvement_callback =
      [this, &population, problem_ptr = context.problem_ptr](
        f_t obj, const std::vector<f_t>& assignment, double /*work_units*/) {
        population.add_external_solution(assignment, obj, solution_origin_t::CPUFJ);
        if (obj < local_search_best_obj) {
          CUOPT_LOG_TRACE("******* New local search best obj %g, best overall %g",
                          problem_ptr->get_user_obj_from_solver_obj(obj),
                          problem_ptr->get_user_obj_from_solver_obj(
                            population.is_feasible() ? population.best_feasible().get_objective()
                                                     : std::numeric_limits<f_t>::max()));
          local_search_best_obj = obj;
        }
      };
  }

  CUOPT_LOG_DEBUG("Launching %d scratch CPUFJ tasks", scratch_cpu_fj.size());

  for (size_t i = 0; i < scratch_cpu_fj.size(); ++i) {
    auto ptr = scratch_cpu_fj[i].get();
#pragma omp task firstprivate(ptr) priority(CUOPT_DEFAULT_TASK_PRIORITY) \
  depend(out : *ptr) default(none)
    cpufj_solve(ptr);
  }
}

template <typename i_t, typename f_t>
void local_search_t<i_t, f_t>::start_cpufj_lptopt_scratch_threads(
  population_t<i_t, f_t>& population)
{
  // TODO: Find a way to enable this in low core count scenarios
  if (omp_get_num_threads() < CUOPT_MIP_FJ_REQUIRED_THREAD_COUNT) return;

  pop_ptr = &population;

  std::vector<f_t> default_weights(context.problem_ptr->n_constraints, 1.);

  solution_t<i_t, f_t> solution_lp(*context.problem_ptr);
  solution_lp.copy_new_assignment(
    host_copy(lp_optimal_solution, context.problem_ptr->handle_ptr->get_stream()));
  solution_lp.round_random_nearest(500);
  scratch_cpu_fj_on_lp_opt = fj.create_cpu_climber(
    solution_lp, default_weights, default_weights, 0., context.preempt_heuristic_solver_);
  scratch_cpu_fj_on_lp_opt->log_prefix = "******* scratch on LP optimal: ";
  scratch_cpu_fj_on_lp_opt->improvement_callback =
    [this, &population](f_t obj, const std::vector<f_t>& h_vec, double /*work_units*/) {
      population.add_external_solution(h_vec, obj, solution_origin_t::CPUFJ);
      if (obj < this->local_search_best_obj) {
        CUOPT_LOG_DEBUG("******* New local search best obj %g, best overall %g",
                        context.problem_ptr->get_user_obj_from_solver_obj(obj),
                        context.problem_ptr->get_user_obj_from_solver_obj(
                          population.is_feasible() ? population.best_feasible().get_objective()
                                                   : std::numeric_limits<f_t>::max()));
        this->local_search_best_obj = obj;
      }
    };

  CUOPT_LOG_DEBUG("Launching scratch CPUFJ (on LP optimal) task");

#pragma omp task shared(scratch_cpu_fj_on_lp_opt) \
  priority(CUOPT_DEFAULT_TASK_PRIORITY) default(none) depend(out : *scratch_cpu_fj_on_lp_opt)
  cpufj_solve(scratch_cpu_fj_on_lp_opt.get());
}

template <typename i_t, typename f_t>
void local_search_t<i_t, f_t>::stop_cpufj_scratch_threads()
{
  if (omp_get_num_threads() < CUOPT_MIP_FJ_REQUIRED_THREAD_COUNT) return;

  // Signal every climber first, then wait: waiting on one at a time would let the others keep
  // running for the duration of each wait. The null check covers the case where the scratch threads
  // were never started (or were started before the climbers were created).
  for (auto& climber : scratch_cpu_fj)
    if (climber) climber->halted = true;
  for (auto& climber : scratch_cpu_fj) {
    if (!climber) continue;
#pragma omp taskwait depend(in : *climber)
  }

  if (scratch_cpu_fj_on_lp_opt) {
    scratch_cpu_fj_on_lp_opt->halted = true;
#pragma omp taskwait depend( \
    in : *scratch_cpu_fj_on_lp_opt)  // Wait for the scratch CPU FJ (LP optimal) task to finish

    CUOPT_LOG_DEBUG("All scratch CPUFJ tasks were stopped");
  }
}

template <typename i_t, typename f_t>
void local_search_t<i_t, f_t>::start_cpufj_deterministic(mip::branch_and_bound_t<i_t, f_t>& bb)
{
  producer_sync_t& producer_sync = bb.get_producer_sync();

  if (omp_get_num_threads() < CUOPT_MIP_FJ_REQUIRED_THREAD_COUNT) {
    producer_sync.registration_complete();
    return;
  }

  std::vector<f_t> default_weights(context.problem_ptr->n_constraints, 1.);

  solution_t<i_t, f_t> solution(*context.problem_ptr);
  thrust::fill(solution.handle_ptr->get_thrust_policy(),
               solution.assignment.begin(),
               solution.assignment.end(),
               0.0);
  solution.clamp_within_bounds();

  deterministic_cpu_fj = fj.create_cpu_climber(solution,
                                               default_weights,
                                               default_weights,
                                               0.,
                                               context.preempt_heuristic_solver_,
                                               fj_settings_t{},
                                               /*randomize=*/true);

  deterministic_cpu_fj->log_prefix = "******* deterministic CPUFJ: ";

  // Register with producer_sync for B&B synchronization
  deterministic_cpu_fj->producer_sync = &producer_sync;
  producer_sync.register_producer(&deterministic_cpu_fj->work_units_elapsed);

  // Set up callback to send solutions to B&B with work unit timestamps
  deterministic_cpu_fj->improvement_callback =
    [&bb](f_t obj, const std::vector<f_t>& h_vec, double work_units) {
      bb.queue_external_solution_deterministic(h_vec, work_units);
    };

  CUOPT_LOG_DEBUG("Launching deterministic CPUFJ task");
#pragma omp task shared(deterministic_cpu_fj) priority(CUOPT_DEFAULT_TASK_PRIORITY) default(none) \
  depend(inout : *deterministic_cpu_fj)
  cpufj_solve(deterministic_cpu_fj.get());

  // Signal that registration is complete - B&B can now wait on producers
  producer_sync.registration_complete();
}

template <typename i_t, typename f_t>
void local_search_t<i_t, f_t>::stop_cpufj_deterministic()
{
  if (deterministic_cpu_fj) {
    if (deterministic_cpu_fj->producer_sync) {
      deterministic_cpu_fj->producer_sync->deregister_producer(
        &deterministic_cpu_fj->work_units_elapsed);
    }

    deterministic_cpu_fj->halted = true;
#pragma omp taskwait depend( \
    in : *deterministic_cpu_fj)  // Wait for deterministic CPU FJ task to finish
    CUOPT_LOG_DEBUG("Deterministic CPUFJ task was stopped");
  }
}

template <typename i_t, typename f_t>
bool local_search_t<i_t, f_t>::do_fj_solve(solution_t<i_t, f_t>& solution,
                                           fj_t<i_t, f_t>& in_fj,
                                           f_t time_limit,
                                           const std::string& source)
{
  if (time_limit == 0.) return solution.get_feasible();

  timer_t timer(time_limit);
  const auto old_n_cstr_weights      = in_fj.cstr_weights.size();
  const auto expected_n_cstr_weights = static_cast<size_t>(solution.problem_ptr->n_constraints);
  // in case this is the first time run, resize
  if (old_n_cstr_weights != expected_n_cstr_weights) {
    in_fj.cstr_weights.resize(solution.problem_ptr->n_constraints,
                              solution.handle_ptr->get_stream());
    cuopt_assert(in_fj.cstr_weights.size() == expected_n_cstr_weights,
                 "Constraint weights must match constraint count after resize");
    // Initialize only newly grown entries; shrinking does not need initialization.
    if (old_n_cstr_weights < expected_n_cstr_weights) {
      cuopt_assert(old_n_cstr_weights <= in_fj.cstr_weights.size(),
                   "Constraint weight fill start must be within range");
      thrust::uninitialized_fill(solution.handle_ptr->get_thrust_policy(),
                                 in_fj.cstr_weights.begin() + old_n_cstr_weights,
                                 in_fj.cstr_weights.end(),
                                 1.);
    }
  }
  auto h_weights          = cuopt::host_copy(in_fj.cstr_weights, solution.handle_ptr->get_stream());
  auto h_objective_weight = in_fj.objective_weight.value(solution.handle_ptr->get_stream());
  for (auto& cpu_fj : ls_cpu_fj) {
    cpu_fj = fj.create_cpu_climber(solution,
                                   h_weights,
                                   h_weights,
                                   h_objective_weight,
                                   context.preempt_heuristic_solver_,
                                   fj_settings_t{},
                                   true);
  }

  auto solution_copy = solution;

  // Start CPU solver in background thread
#pragma omp taskgroup
  {
    if (ls_cpu_fj.size() > 0 && omp_get_num_threads() > CUOPT_MIP_FJ_REQUIRED_THREAD_COUNT) {
      size_t n = std::min<size_t>(omp_get_num_threads() - 1, ls_cpu_fj.size());
      CUOPT_LOG_DEBUG("Launching %d CPUFJ tasks", n);

#pragma omp taskloop shared(ls_cpu_fj) default(none) num_tasks(n) \
  nogroup priority(CUOPT_DEFAULT_TASK_PRIORITY)
      for (size_t i = 0; i < n; ++i) {
        cpufj_solve(ls_cpu_fj[i].get());
      }
    }

    // Run GPU solver
    in_fj.settings.time_limit = timer.remaining_time();
    in_fj.solve(solution);

    for (size_t i = 0; i < ls_cpu_fj.size(); ++i) {
      ls_cpu_fj[i]->halted = true;
    }
  }  // implicit barrier that waits all CPU FJ tasks to finish

  CUOPT_LOG_DEBUG("All CPUFJ tasks were stopped");

  solution_t<i_t, f_t> solution_cpu(*solution.problem_ptr);
  f_t best_cpu_obj = std::numeric_limits<f_t>::max();

  for (size_t i = 0; i < ls_cpu_fj.size(); ++i) {
    if (ls_cpu_fj[i]->feasible_found) {
      f_t cpu_obj = ls_cpu_fj[i]->h_best_objective;
      if (cpu_obj < best_cpu_obj) {
        best_cpu_obj = cpu_obj;
        solution_cpu.copy_new_assignment(ls_cpu_fj[i]->h_best_assignment);
        solution_cpu.compute_feasibility();
      }
    }
  }
  bool cpu_sol_found = best_cpu_obj < std::numeric_limits<f_t>::max();

  bool gpu_feasible = solution.get_feasible();
  bool cpu_feasible = cpu_sol_found && solution_cpu.get_feasible();

  static std::unordered_map<std::string, int> total_calls;
  static std::unordered_map<std::string, int> cpu_better;

  CUOPT_LOG_DEBUG("GPU FJ returns feas %d, obj %g", gpu_feasible, solution.get_user_objective());
  CUOPT_LOG_DEBUG("CPU FJ returns feas %d, obj %g, stats %d/%d",
                  cpu_feasible,
                  solution_cpu.get_user_objective(),
                  total_calls[source],
                  cpu_better[source]);

  total_calls[source]++;
  if (cpu_feasible && !gpu_feasible ||
      (cpu_feasible && solution_cpu.get_objective() < solution.get_objective())) {
    CUOPT_LOG_DEBUG(
      "CPU FJ returns better solution! cpu_obj %g, gpu_obj %g, stats %d/%d, source %s",
      solution_cpu.get_user_objective(),
      solution.get_user_objective(),
      total_calls[source],
      cpu_better[source],
      source.c_str());
    solution.copy_from(solution_cpu);
    cpu_better[source]++;
  }
  solution.compute_feasibility();

  return cpu_feasible;
}

template <typename i_t, typename f_t>
void local_search_t<i_t, f_t>::generate_fast_solution(solution_t<i_t, f_t>& solution, timer_t timer)
{
  thrust::fill(solution.handle_ptr->get_thrust_policy(),
               solution.assignment.begin(),
               solution.assignment.end(),
               0.0);
  solution.clamp_within_bounds();
  fj.settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
  fj.settings.n_of_minimums_for_exit = 500;
  fj.settings.update_weights         = true;
  fj.settings.feasibility_run        = true;
  fj.settings.time_limit             = std::min(30., timer.remaining_time());
  while (!context.diversity_manager_ptr->check_b_b_preemption() && !timer.check_time_limit()) {
    timer_t constr_prop_timer = timer_t(std::min(timer.remaining_time(), 2.));
    // do constraint prop on lp optimal solution
    constraint_prop.apply_round(solution, 1., constr_prop_timer);
    if (solution.compute_feasibility()) { return; }
    if (timer.check_time_limit()) { return; };
    f_t time_limit = std::min(3., timer.remaining_time());
    // run fj on the solution
    do_fj_solve(solution, fj, time_limit, "fast");
    // TODO check if FJ returns the same solution
    // check if the solution is feasible
    if (solution.compute_feasibility()) { return; }
  }
}

template <typename i_t, typename f_t>
bool local_search_t<i_t, f_t>::run_local_search(solution_t<i_t, f_t>& solution,
                                                const weight_t<i_t, f_t>& weights,
                                                timer_t timer,
                                                const ls_config_t<i_t, f_t>& ls_config)
{
  raft::common::nvtx::range fun_scope("local search");
  fj_settings_t fj_settings;
  if (timer.check_time_limit()) return false;
  // adjust these time limits
  if (!solution.get_feasible()) {
    if (ls_config.at_least_one_parent_feasible) {
      fj_settings.time_limit = 0.5;
      timer                  = timer_t(fj_settings.time_limit);
    } else {
      fj_settings.time_limit = 0.25;
      timer                  = timer_t(fj_settings.time_limit);
    }
  } else {
    fj_settings.time_limit = std::min(1., timer.remaining_time());
  }
  fj_settings.update_weights  = false;
  fj_settings.feasibility_run = false;
  fj.set_fj_settings(fj_settings);
  bool is_feas   = false;
  ls_method_t rd = static_cast<ls_method_t>(
    std::uniform_int_distribution(static_cast<int>(ls_method_t::FJ_ANNEALING),
                                  static_cast<int>(ls_method_t::FJ_LINE_SEGMENT))(rng));
  if (ls_config.ls_method == ls_method_t::FJ_LINE_SEGMENT) {
    rd = ls_method_t::FJ_LINE_SEGMENT;
  } else if (ls_config.ls_method == ls_method_t::FJ_ANNEALING) {
    rd = ls_method_t::FJ_ANNEALING;
  }
  if (rd == ls_method_t::FJ_LINE_SEGMENT && lp_optimal_exists) {
    fj.copy_weights(weights, solution.handle_ptr);
    is_feas = run_fj_line_segment(solution, timer, ls_config);
  } else {
    fj.copy_weights(weights, solution.handle_ptr);
    is_feas = run_fj_annealing(solution, timer, ls_config);
    if (lp_optimal_exists) { is_feas = run_fj_line_segment(solution, timer, ls_config); }
  }
  return is_feas;
}

template <typename i_t, typename f_t>
bool local_search_t<i_t, f_t>::run_fj_until_timer(solution_t<i_t, f_t>& solution,
                                                  const weight_t<i_t, f_t>& weights,
                                                  timer_t timer)
{
  bool is_feasible;
  fj.settings.n_of_minimums_for_exit = 1e6;
  fj.settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
  fj.settings.update_weights         = false;
  fj.settings.feasibility_run        = false;
  fj.copy_weights(weights, solution.handle_ptr);
  f_t time_limit = timer.remaining_time() * 0.95;
  do_fj_solve(solution, fj, time_limit, "until_timer");
  CUOPT_LOG_DEBUG("Initial FJ feasibility done");
  is_feasible = solution.compute_feasibility();
  if (fj.settings.feasibility_run || timer.check_time_limit()) { return is_feasible; }
  return is_feasible;
}

template <typename i_t, typename f_t>
bool local_search_t<i_t, f_t>::run_fj_annealing(solution_t<i_t, f_t>& solution,
                                                timer_t timer,
                                                const ls_config_t<i_t, f_t>& ls_config)
{
  raft::common::nvtx::range fun_scope("run_fj_annealing");
  auto prev_settings = fj.settings;

  solution.compute_feasibility();
  CUOPT_LOG_DEBUG("Running FJ Annealing on solution with obj %g/%g, feas? %d",
                  solution.get_user_objective(),
                  solution.get_objective(),
                  solution.get_feasible());

  // run in FEASIBLE_FIRST to priorize feasibility-improving moves
  fj.settings.n_of_minimums_for_exit                    = ls_config.n_local_mins;
  fj.settings.mode                                      = fj_mode_t::EXIT_NON_IMPROVING;
  fj.settings.candidate_selection                       = fj_candidate_selection_t::FEASIBLE_FIRST;
  fj.settings.iteration_limit                           = ls_config.iteration_limit;
  fj.settings.parameters.allow_infeasibility_iterations = 100;
  fj.settings.update_weights                            = 1;
  fj.settings.baseline_objective_for_longer_run         = ls_config.best_objective_of_parents;
  f_t time_limit                                        = std::min(10., timer.remaining_time());
  do_fj_solve(solution, fj, time_limit, "annealing");
  bool is_feasible = solution.compute_feasibility();

  fj.settings = prev_settings;
  return is_feasible;
}

template <typename i_t, typename f_t>
bool local_search_t<i_t, f_t>::run_fj_line_segment(solution_t<i_t, f_t>& solution,
                                                   timer_t timer,
                                                   const ls_config_t<i_t, f_t>& ls_config)
{
  raft::common::nvtx::range fun_scope("run_fj_line_segment");
  rmm::device_uvector<f_t> starting_point(solution.assignment, solution.handle_ptr->get_stream());
  line_segment_search.settings.best_of_parents_cost = ls_config.best_objective_of_parents;
  line_segment_search.settings.parents_infeasible   = !ls_config.at_least_one_parent_feasible;
  line_segment_search.settings.recombiner_mode      = false;
  line_segment_search.settings.n_local_min          = ls_config.n_local_mins_for_line_segment;
  line_segment_search.settings.n_points_to_search   = ls_config.n_points_to_search_for_line_segment;
  line_segment_search.settings.iteration_limit      = ls_config.iteration_limit_for_line_segment;

  bool feas = line_segment_search.search_line_segment(solution,
                                                      starting_point,
                                                      lp_optimal_solution,
                                                      /*feasibility_run=*/false,
                                                      timer);
  return feas;
}

template <typename i_t, typename f_t>
bool local_search_t<i_t, f_t>::check_fj_on_lp_optimal(solution_t<i_t, f_t>& solution,
                                                      bool perturb,
                                                      timer_t timer)
{
  raft::common::nvtx::range fun_scope("check_fj_on_lp_optimal");
  if (lp_optimal_exists) {
    raft::copy(solution.assignment.data(),
               lp_optimal_solution.data(),
               solution.assignment.size(),
               solution.handle_ptr->get_stream());
    cuopt_func_call(solution.test_variable_bounds(false));
  }
  if (perturb) {
    CUOPT_LOG_DEBUG("Perturbating solution on initial fj on optimal run!");
    f_t perturbation_ratio = 0.2;
    solution.assign_random_within_bounds(perturbation_ratio);
  }
  cuopt_func_call(solution.test_variable_bounds(false));
  f_t lp_run_time_after_feasible = std::min(1., timer.remaining_time());
  timer_t bounds_prop_timer      = timer_t(std::min(timer.remaining_time(), 10.));
  bool is_feasible =
    constraint_prop.apply_round(solution, lp_run_time_after_feasible, bounds_prop_timer);
  if (!is_feasible) {
    const f_t lp_run_time = 2.;
    relaxed_lp_settings_t lp_settings;
    lp_settings.time_limit = std::min(lp_run_time, timer.remaining_time());
    lp_settings.tolerance  = solution.problem_ptr->tolerances.absolute_tolerance;
    run_lp_with_vars_fixed(
      *solution.problem_ptr, solution, solution.problem_ptr->integer_indices, lp_settings);
  } else {
    return is_feasible;
  }
  cuopt_func_call(solution.test_variable_bounds());
  fj.settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
  fj.settings.n_of_minimums_for_exit = 20000;
  fj.settings.update_weights         = true;
  fj.settings.feasibility_run        = false;
  f_t time_limit                     = std::min(30., timer.remaining_time());
  do_fj_solve(solution, fj, time_limit, "on_lp_optimal");
  return solution.get_feasible();
}

/**
 * @brief Shifting rounding heuristic: round the integers of `solution` one at a time and repair the
 *        constraints that break along the way.
 *
 * Unlike plain rounding, a violated row is not left violated: the loop alternates between fixing the
 * next fractional integer and shifting some variable to repair a violated row. Runs entirely on the
 * host on a copy of the assignment (the working set is small and the access pattern is irregular),
 * writes the result back and reports whether it ended up feasible. On failure `solution` still holds
 * the partially shifted point, so callers must check the return value before keeping it.
 */
template <typename i_t, typename f_t>
bool local_search_t<i_t, f_t>::run_shifting(solution_t<i_t, f_t>& solution, timer_t timer)
{
  namespace simplex = cuopt::mathematical_optimization::simplex;
  auto* problem      = solution.problem_ptr;
  if (problem->n_integer_vars == 0 || timer.check_time_limit()) return false;

  simplex::user_problem_t<i_t, f_t> host(problem->handle_ptr);
  problem->get_host_user_problem(host);
  auto x = host_copy(solution.assignment, problem->handle_ptr->get_stream());
  // Guard against a problem/solution size mismatch (e.g. presolve resized in between)
  if (x.size() != static_cast<size_t>(host.num_cols)) return false;

  // The host problem stores rows as sense + rhs; convert to explicit [lb, ub] ranges
  const f_t inf = std::numeric_limits<f_t>::infinity();
  const f_t tol = std::max<f_t>(problem->tolerances.absolute_tolerance, 1e-7);
  std::vector<f_t> row_lb(host.num_rows, -inf), row_ub(host.num_rows, inf);
  for (i_t row = 0; row < host.num_rows; ++row) {
    const char sense = host.row_sense.empty() ? 'E' : host.row_sense[row];
    if (sense == 'E') row_lb[row] = row_ub[row] = host.rhs[row];
    else if (sense == 'L') row_ub[row] = host.rhs[row];
    else row_lb[row] = host.rhs[row];
  }

  // Single pass over the column-major matrix builds everything the loop below needs: a row-major view
  // (to pick repair candidates), the row activities (kept incrementally up to date afterwards), and
  // the variable locks. up_locks[col] counts the rows that increasing col could push out of its
  // range, down_locks[col] the same for decreasing it; a variable with few locks in one direction is
  // a cheap one to move that way.
  std::vector<std::vector<std::pair<i_t, f_t>>> rows(host.num_rows);
  std::vector<f_t> activity(host.num_rows, 0);
  std::vector<i_t> up_locks(host.num_cols, 0), down_locks(host.num_cols, 0);
  for (i_t col = 0; col < host.num_cols; ++col) {
    x[col] = std::clamp(x[col], host.lower[col], host.upper[col]);
    for (i_t p = host.A.col_start[col]; p < host.A.col_start[col + 1]; ++p) {
      const i_t row = host.A.i[p];
      const f_t a   = host.A.x[p];
      rows[row].emplace_back(col, a);
      activity[row] += a * x[col];
      up_locks[col] += (a > 0 && std::isfinite(row_ub[row])) ||
                       (a < 0 && std::isfinite(row_lb[row]));
      down_locks[col] += (a > 0 && std::isfinite(row_lb[row])) ||
                         (a < 0 && std::isfinite(row_ub[row]));
    }
  }

  std::vector<unsigned char> fractional(host.num_cols, 0), queued(host.num_rows, 0);
  std::vector<i_t> order;
  for (i_t col = 0; col < host.num_cols; ++col) {
    if (host.var_types[col] != simplex::variable_type_t::CONTINUOUS &&
        std::abs(x[col] - std::round(x[col])) > problem->tolerances.integrality_tolerance) {
      fractional[col] = 1;
      order.push_back(col);
    }
  }
  // Round the most constrained variables first: they have the fewest safe values left, so deciding
  // them while the rest of the assignment is still flexible gives the repair step more room.
  std::stable_sort(order.begin(), order.end(), [&](i_t a, i_t b) {
    return std::max(up_locks[a], down_locks[a]) > std::max(up_locks[b], down_locks[b]);
  });

  auto violation = [&](i_t row) {
    return std::max<f_t>({0, row_lb[row] - activity[row], activity[row] - row_ub[row]});
  };
  // Worklist of violated rows. `queued` keeps it duplicate-free; entries may still go stale (a later
  // move can repair a queued row), so consumers re-check the violation before acting on one.
  std::vector<i_t> bad_rows;
  auto enqueue = [&](i_t row) {
    if (!queued[row] && violation(row) > tol) {
      queued[row] = 1;
      bad_rows.push_back(row);
    }
  };
  for (i_t row = 0; row < host.num_rows; ++row) enqueue(row);

  // Set col to value (clamped to its bounds), updating the activities of the rows it appears in.
  // Returns false when the move was a no-op, which the caller counts as a stall.
  auto move = [&](i_t col, f_t value) {
    value           = std::clamp(value, host.lower[col], host.upper[col]);
    const f_t delta = value - x[col];
    if (std::abs(delta) <= tol) return false;
    x[col] = value;
    for (i_t p = host.A.col_start[col]; p < host.A.col_start[col + 1]; ++p) {
      const i_t row = host.A.i[p];
      activity[row] += host.A.x[p] * delta;
      enqueue(row);
    }
    return true;
  };

  // Repairing violations takes priority over rounding, so the step count has to allow for more than
  // one step per integer; bail out early if we make no progress for a few consecutive steps.
  size_t order_pos    = 0;
  i_t stalled         = 0;
  const i_t max_steps = std::max<i_t>(problem->n_integer_vars + 6, 32);
  for (i_t step = 0; step < max_steps && stalled <= 5 && !timer.check_time_limit(); ++step) {
    // Pop the next still-violated row, if any
    i_t row = -1;
    while (!bad_rows.empty()) {
      const i_t candidate = bad_rows.back();
      bad_rows.pop_back();
      queued[candidate] = 0;
      if (violation(candidate) > tol) { row = candidate; break; }
    }
    // Nothing violated: round the next fractional integer (skipping the ones a repair step already
    // rounded). When there are none left the assignment is integral and feasible, so stop.
    if (row < 0) {
      while (order_pos < order.size() && !fractional[order[order_pos]]) ++order_pos;
      if (order_pos == order.size()) break;
      const i_t col   = order[order_pos++];
      fractional[col] = 0;
      // Round away from the more locked direction, i.e. towards the side that breaks fewer rows
      const f_t value = up_locks[col] >= down_locks[col] ? std::floor(x[col] + tol)
                                                        : std::ceil(x[col] - tol);
      stalled         = move(col, value) ? 0 : stalled + 1;
      continue;
    }

    // Repair `row`: pick the variable whose shift in the required direction hurts least
    const bool need_up = activity[row] < row_lb[row];
    f_t best_score = inf, best_coef = 0;
    i_t best_col = -1;
    int best_dir = 0;
    for (const auto [col, a] : rows[row]) {
      const int dir = ((need_up && a > 0) || (!need_up && a < 0)) ? 1 : -1;
      // Already at the bound we would need to move past
      if ((dir > 0 && x[col] >= host.upper[col] - tol) ||
          (dir < 0 && x[col] <= host.lower[col] + tol)) continue;
      // Lower is better. Still-fractional integers score below zero, so repairing a row by rounding
      // one of them (which we have to do anyway) always beats disturbing a settled variable; among
      // those, prefer the one with fewer locks in the direction we are moving. Everything else is
      // scored by its objective deterioration, with a penalty that favours continuous variables.
      f_t score = fractional[col]
                    ? -1 + f_t{1} / (f_t{1} + (dir > 0 ? down_locks[col] : up_locks[col]))
                    : dir * host.objective[col] +
                        (host.var_types[col] != simplex::variable_type_t::CONTINUOUS);
      if (score < best_score) {
        best_score = score; best_col = col; best_dir = dir; best_coef = a;
      }
    }
    if (best_col < 0) { ++stalled; continue; }

    // How far to shift: rounding to the next integer in the repair direction, a unit step for an
    // already integral integer, or exactly enough to close the violation for a continuous variable
    f_t value;
    if (fractional[best_col]) {
      value                = best_dir > 0 ? std::ceil(x[best_col] - tol) : std::floor(x[best_col] + tol);
      fractional[best_col] = 0;
    } else if (host.var_types[best_col] != simplex::variable_type_t::CONTINUOUS) {
      value = x[best_col] + best_dir;
    } else {
      value = x[best_col] + best_dir * violation(row) / std::abs(best_coef);
    }
    stalled = move(best_col, value) ? 0 : stalled + 1;
    // A single step may not be enough (bounds clamping, integer steps), so requeue if still violated
    enqueue(row);
  }

  // Push the shifted point back to the device and let the solution decide on feasibility: the loop
  // above may have run out of steps or time with violations left.
  raft::copy(solution.assignment.data(), x.data(), x.size(), problem->handle_ptr->get_stream());
  solution.clamp_within_bounds();
  return solution.compute_feasibility();
}

template <typename i_t, typename f_t>
bool local_search_t<i_t, f_t>::run_fj_on_zero(solution_t<i_t, f_t>& solution, timer_t timer)
{
  raft::common::nvtx::range fun_scope("run_fj_on_zero");
  thrust::fill(solution.handle_ptr->get_thrust_policy(),
               solution.assignment.begin(),
               solution.assignment.end(),
               0.0);
  solution.clamp_within_bounds();
  fj.settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
  fj.settings.n_of_minimums_for_exit = 20000;
  fj.settings.update_weights         = true;
  fj.settings.feasibility_run        = false;
  f_t time_limit                     = std::min(30., timer.remaining_time());
  bool is_feasible                   = do_fj_solve(solution, fj, time_limit, "on_zero");
  return is_feasible;
}

template <typename i_t, typename f_t>
bool local_search_t<i_t, f_t>::run_staged_fp(solution_t<i_t, f_t>& solution,
                                             timer_t timer,
                                             population_t<i_t, f_t>* population_ptr)
{
  raft::common::nvtx::range fun_scope("run_staged_fp");
  cuopt_assert(population_ptr != nullptr, "Population pointer must not be null");
  auto n_vars         = solution.problem_ptr->n_variables;
  auto n_binary_vars  = solution.problem_ptr->get_n_binary_variables();
  auto n_integer_vars = solution.problem_ptr->n_integer_vars;

  auto binary_only  = (n_binary_vars == n_integer_vars);
  auto integer_only = (n_binary_vars == 0);
  bool is_feasible  = false;

  if (binary_only || integer_only) {
    return run_fp(solution, timer, population_ptr);
  } else {
    const i_t n_fp_iterations = 1000000;
    fp.cycle_queue.reset(solution);
    fp.reset();
    fp.resize_vectors(*solution.problem_ptr, solution.handle_ptr);
    for (i_t i = 0; i < n_fp_iterations && !timer.check_time_limit(); ++i) {
      population_ptr->add_external_solutions_to_population();
      if (context.preempt_heuristic_solver_.load()) {
        CUOPT_LOG_DEBUG("Preempting heuristic solver!");
        return false;
      }
      CUOPT_LOG_DEBUG("Running staged FP from beginning it %d", i);
      fp.relax_general_integers(solution);
      timer_t binary_timer(timer.remaining_time() / 3);
      i_t binary_it_counter = 0;
      for (; binary_it_counter < 100; ++binary_it_counter) {
        population_ptr->add_external_solutions_to_population();
        if (context.preempt_heuristic_solver_.load()) {
          CUOPT_LOG_DEBUG("Preempting heuristic solver!");
          return false;
        }
        CUOPT_LOG_DEBUG(
          "Running binary problem from it %d large_restart_it %d", binary_it_counter, i);
        is_feasible = fp.run_single_fp_descent(solution);
        if (is_feasible) { break; }
        if (timer.check_time_limit()) {
          fp.revert_relaxation(solution);
          solution.round_nearest();
          CUOPT_LOG_DEBUG("Time limit reached during binary stage!");
          return false;
        }
        is_feasible = fp.restart_fp(solution);
        if (is_feasible) { break; }
        // give the integer FP some chance
        if (binary_timer.check_time_limit()) {
          CUOPT_LOG_DEBUG("Binary FP time limit reached during binary stage!");
          break;
        }
      }
      CUOPT_LOG_DEBUG("Exited binary problem at it %d large_restart_it %d feas %d",
                      binary_it_counter,
                      i,
                      is_feasible);
      // TODO try resetting and not resetting the alpha
      fp.revert_relaxation(solution);
      fp.last_distances.resize(0);

      for (i_t integer_it_counter = 0; integer_it_counter < 500; ++integer_it_counter) {
        CUOPT_LOG_DEBUG(
          "Running integer problem from it %d large_restart_it %d", integer_it_counter, i);
        is_feasible = fp.run_single_fp_descent(solution);
        if (is_feasible) { return true; }
        if (timer.check_time_limit()) {
          CUOPT_LOG_DEBUG("FP time limit reached during integer stage!");
          solution.round_nearest();
          return false;
        }
        is_feasible = fp.restart_fp(solution);
        if (is_feasible) { return true; }
      }
    }
  }
  return false;
}

template <typename i_t, typename f_t>
void local_search_t<i_t, f_t>::resize_vectors(problem_t<i_t, f_t>& problem,
                                              const raft::handle_t* handle_ptr)
{
  fj_sol_on_lp_opt.resize(problem.n_variables, handle_ptr->get_stream());
  fp.resize_vectors(problem, handle_ptr);
}

template <typename i_t, typename f_t>
void local_search_t<i_t, f_t>::save_solution_and_add_cutting_plane(
  solution_t<i_t, f_t>& solution, rmm::device_uvector<f_t>& best_solution, f_t& best_objective)
{
  raft::common::nvtx::range fun_scope("save_solution_and_add_cutting_plane");
  if (solution.get_objective() < best_objective) {
    raft::copy(best_solution.data(),
               solution.assignment.data(),
               solution.assignment.size(),
               solution.handle_ptr->get_stream());
    best_objective = solution.get_objective();
    f_t objective_cut =
      best_objective - std::max(std::abs(0.001 * best_objective), OBJECTIVE_EPSILON);
    problem_with_objective_cut.add_cutting_plane_at_objective(objective_cut);
  }
}

template <typename i_t, typename f_t>
void local_search_t<i_t, f_t>::resize_to_new_problem()
{
  resize_vectors(problem_with_objective_cut, problem_with_objective_cut.handle_ptr);
  // hint for next PR in case load balanced is reintroduced
  // lb_constraint_prop.temp_problem.setup(problem_with_objective_cut);
  // lb_constraint_prop.bounds_update.setup(lb_constraint_prop.temp_problem);
  constraint_prop.bounds_update.resize(problem_with_objective_cut);
}

template <typename i_t, typename f_t>
void local_search_t<i_t, f_t>::resize_to_old_problem(problem_t<i_t, f_t>* old_problem_ptr)
{
  resize_vectors(*old_problem_ptr, old_problem_ptr->handle_ptr);
  // hint for next PR in case load balanced is reintroduced
  // lb_constraint_prop.temp_problem.setup(*old_problem_ptr);
  // lb_constraint_prop.bounds_update.setup(lb_constraint_prop.temp_problem);
  constraint_prop.bounds_update.resize(*old_problem_ptr);
}

template <typename i_t, typename f_t>
void local_search_t<i_t, f_t>::reset_alpha_and_save_solution(
  solution_t<i_t, f_t>& solution,
  problem_t<i_t, f_t>* old_problem_ptr,
  population_t<i_t, f_t>* population_ptr,
  i_t i,
  i_t last_improved_iteration,
  rmm::device_uvector<f_t>& best_solution,
  f_t& best_objective)
{
  raft::common::nvtx::range fun_scope("reset_alpha_and_save_solution");
  fp.config.alpha = default_alpha;
  solution_t<i_t, f_t> solution_copy(solution);
  solution_copy.problem_ptr = old_problem_ptr;
  solution_copy.resize_to_problem();
  population_ptr->add_solution(std::move(solution_copy));
  population_ptr->add_external_solutions_to_population();
  if (!cutting_plane_added_for_active_run) {
    solution.problem_ptr = &problem_with_objective_cut;
    solution.resize_to_problem();
    resize_to_new_problem();
    cutting_plane_added_for_active_run = true;
    raft::copy(population_ptr->weights.cstr_weights.data(),
               fj.cstr_weights.data(),
               population_ptr->weights.cstr_weights.size(),
               solution.handle_ptr->get_stream());
  }
  population_ptr->update_weights();
  save_solution_and_add_cutting_plane(
    population_ptr->best_feasible(), best_solution, best_objective);
  raft::copy(solution.assignment.data(),
             best_solution.data(),
             solution.assignment.size(),
             solution.handle_ptr->get_stream());
  population_ptr->print();
}

template <typename i_t, typename f_t>
void local_search_t<i_t, f_t>::reset_alpha_and_run_recombiners(
  solution_t<i_t, f_t>& solution,
  problem_t<i_t, f_t>* old_problem_ptr,
  population_t<i_t, f_t>* population_ptr,
  i_t i,
  i_t last_improved_iteration,
  rmm::device_uvector<f_t>& best_solution,
  f_t& best_objective)
{
  raft::common::nvtx::range fun_scope("reset_alpha_and_run_recombiners");
  const auto& hp                               = context.settings.heuristic_params;
  const i_t iterations_for_stagnation          = hp.stagnation_trigger;
  const i_t max_iterations_without_improvement = hp.max_iterations_without_improvement;
  population_ptr->add_external_solutions_to_population();
  if (population_ptr->current_size() > 1 &&
      i - last_improved_iteration > iterations_for_stagnation) {
    fp.config.alpha = default_alpha;
    population_ptr->diversity_step(max_iterations_without_improvement);
    population_ptr->print();
    population_ptr->update_weights();
    save_solution_and_add_cutting_plane(
      population_ptr->best_feasible(), best_solution, best_objective);
    raft::copy(solution.assignment.data(),
               best_solution.data(),
               solution.assignment.size(),
               solution.handle_ptr->get_stream());
  }
}

template <typename i_t, typename f_t>
bool local_search_t<i_t, f_t>::run_fp(solution_t<i_t, f_t>& solution,
                                      timer_t timer,
                                      population_t<i_t, f_t>* population_ptr)
{
  raft::common::nvtx::range fun_scope("run_fp");
  cuopt_assert(population_ptr != nullptr, "Population pointer must not be null");
  const i_t n_fp_iterations          = 1000000;
  bool is_feasible                   = solution.compute_feasibility();
  cutting_plane_added_for_active_run = is_feasible;
  double best_objective =
    is_feasible ? solution.get_objective() : std::numeric_limits<double>::max();
  rmm::device_uvector<f_t> best_solution(solution.assignment, solution.handle_ptr->get_stream());
  problem_t<i_t, f_t>* old_problem_ptr = solution.problem_ptr;
  fp.timer                             = timer_t(timer.remaining_time());
  // if it has not been initialized yet, create a new problem and move it to the cut problem
  if (!problem_with_objective_cut.cutting_plane_added) {
    problem_with_objective_cut = std::move(problem_t<i_t, f_t>(*old_problem_ptr));
  }
  if (is_feasible) {
    CUOPT_LOG_DEBUG("FP initial solution is feasible, adding cutting plane at obj");
    f_t objective_cut =
      best_objective - std::max(std::abs(0.001 * best_objective), OBJECTIVE_EPSILON);
    problem_with_objective_cut.add_cutting_plane_at_objective(objective_cut);
    // Do the copy here for proper handling of the added constraints weight
    fj.copy_weights(
      population_ptr->weights, solution.handle_ptr, problem_with_objective_cut.n_constraints);
    solution.problem_ptr = &problem_with_objective_cut;
    solution.resize_to_problem();
    resize_to_new_problem();
  }
  i_t last_improved_iteration = 0;
  for (i_t i = 0; i < n_fp_iterations && !timer.check_time_limit(); ++i) {
    if (timer.check_time_limit()) {
      is_feasible = false;
      break;
    }
    CUOPT_LOG_DEBUG("fp_loop it %d last_improved_iteration %d", i, last_improved_iteration);
    population_ptr->add_external_solutions_to_population();
    if (context.preempt_heuristic_solver_.load()) {
      CUOPT_LOG_DEBUG("Preempting heuristic solver!");
      break;
    }
    is_feasible = fp.run_single_fp_descent(solution);
    population_ptr->add_external_solutions_to_population();
    CUOPT_LOG_DEBUG("Population size at iteration %d: %d", i, population_ptr->current_size());
    if (context.preempt_heuristic_solver_.load()) {
      CUOPT_LOG_DEBUG("Preempting heuristic solver!");
      break;
    }
    if (is_feasible) {
      CUOPT_LOG_DEBUG("Found feasible in FP with obj %f. Continue with FJ!",
                      solution.get_objective());
      reset_alpha_and_save_solution(solution,
                                    old_problem_ptr,
                                    population_ptr,
                                    i,
                                    last_improved_iteration,
                                    best_solution,
                                    best_objective);
      last_improved_iteration = i;
    }
    // if not feasible, it means it is a cycle
    else {
      if (timer.check_time_limit()) {
        is_feasible = false;
        break;
      }
      is_feasible = fp.restart_fp(solution);
      population_ptr->add_external_solutions_to_population();
      if (context.preempt_heuristic_solver_.load()) {
        CUOPT_LOG_DEBUG("Preempting heuristic solver!");
        break;
      }
      if (is_feasible) {
        CUOPT_LOG_DEBUG("Found feasible during restart with obj %f. Continue with FJ!",
                        solution.get_objective());
        reset_alpha_and_save_solution(solution,
                                      old_problem_ptr,
                                      population_ptr,
                                      i,
                                      last_improved_iteration,
                                      best_solution,
                                      best_objective);
        last_improved_iteration = i;
      } else {
        reset_alpha_and_run_recombiners(solution,
                                        old_problem_ptr,
                                        population_ptr,
                                        i,
                                        last_improved_iteration,
                                        best_solution,
                                        best_objective);
      }
    }
  }
  raft::copy(solution.assignment.data(),
             best_solution.data(),
             solution.assignment.size(),
             solution.handle_ptr->get_stream());
  solution.problem_ptr = old_problem_ptr;
  solution.resize_to_problem();
  resize_to_old_problem(old_problem_ptr);
  solution.handle_ptr->sync_stream();
  return is_feasible;
}

template <typename i_t, typename f_t>
bool local_search_t<i_t, f_t>::generate_solution(solution_t<i_t, f_t>& solution,
                                                 bool perturb,
                                                 population_t<i_t, f_t>* population_ptr,
                                                 f_t time_limit)
{
  raft::common::nvtx::range fun_scope("generate_solution");
  cuopt_assert(population_ptr != nullptr, "Population pointer must not be null");
  timer_t timer(time_limit);
  auto n_vars         = solution.problem_ptr->n_variables;
  auto n_binary_vars  = solution.problem_ptr->get_n_binary_variables();
  auto n_integer_vars = solution.problem_ptr->n_integer_vars;
  bool is_feasible    = check_fj_on_lp_optimal(solution, perturb, timer);
  if (is_feasible) {
    CUOPT_LOG_DEBUG("Solution generated with FJ on LP optimal: is_feasible %d", is_feasible);
    return true;
  }
  population_ptr->add_external_solutions_to_population();
  if (context.preempt_heuristic_solver_.load()) {
    CUOPT_LOG_DEBUG("Preempting heuristic solver!");
    return is_feasible;
  }
  if (!perturb) {
    raft::copy(fj_sol_on_lp_opt.data(),
               solution.assignment.data(),
               solution.assignment.size(),
               solution.handle_ptr->get_stream());
    fj.reset_weights(solution.handle_ptr->get_stream());
    is_feasible = run_fj_on_zero(solution, timer);
    if (is_feasible) {
      CUOPT_LOG_DEBUG("Solution generated with FJ on zero solution: is_feasible %d", is_feasible);
      return true;
    }
    raft::copy(solution.assignment.data(),
               fj_sol_on_lp_opt.data(),
               solution.assignment.size(),
               solution.handle_ptr->get_stream());
  }
  population_ptr->add_external_solutions_to_population();
  if (context.preempt_heuristic_solver_.load()) {
    CUOPT_LOG_DEBUG("Preempting heuristic solver!");
    return is_feasible;
  }
  fp.timer = timer;
  // continue with the solution with fj on lp optimal
  fp.cycle_queue.reset(solution);
  fp.reset();
  fp.resize_vectors(*solution.problem_ptr, solution.handle_ptr);
  is_feasible = run_staged_fp(solution, timer, population_ptr);
  // is_feasible = run_fp(solution, timer);
  CUOPT_LOG_DEBUG("Solution generated with FP: is_feasible %d", is_feasible);
  return is_feasible;
}

#if MIP_INSTANTIATE_FLOAT
template class local_search_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class local_search_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::mip
