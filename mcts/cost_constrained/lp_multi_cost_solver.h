// Copyright (c) 2020 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef SOLVE_LP_MULTIPLE_COST_H_
#define SOLVE_LP_MULTIPLE_COST_H_

#include "mcts/statistics/uct_statistic.h"
#include "ortools/linear_solver/linear_solver.h"
#include <vector>
#include <unordered_map>

namespace mcts{ 

PolicySampled lp_multiple_cost_solver(const std::vector<ActionIdx>& feasible_actions,
           const std::vector<UctStatistic>& cost_statistics, const std::vector<Cost> cost_constraints,
            std::vector<double> lambdas,
           std::mt19937& random_generator) {

    std::vector<std::vector<Cost>> action_values;
    for (std::size_t cost_idx = 0; cost_idx < cost_statistics.size(); ++cost_idx) {
        std::vector<Cost> action_values_lambda;
        const auto& cost_stats = cost_statistics.at(cost_idx).get_ucb_statistics();
        for (const auto feasible_action : feasible_actions) {
            action_values_lambda.push_back(cost_statistics.at(cost_idx).get_normalized_ucb_value(feasible_action));
        }
        action_values.push_back(action_values_lambda);
    }
    const auto num_actions = action_values[0].size();

    // Create the linear solver with the GLOP backend.
    operations_research::MPSolver solver("multiple_cost", operations_research::MPSolver::GLOP_LINEAR_PROGRAMMING);

    // Create the variables x and y.
    std::vector<operations_research::MPVariable* > action_weights;
    std::vector<operations_research::MPVariable* > pos_errors;
    std::vector<operations_research::MPVariable* > neg_errors;

    for (std::size_t action_idx = 0; action_idx < num_actions; ++action_idx) {
        action_weights.push_back(solver.MakeNumVar(0.0, 1.0, std::string("w_") + std::to_string(action_idx)));
    }

    for (std::size_t lambda_idx = 0; lambda_idx <  lambdas.size(); ++lambda_idx) {
        pos_errors.push_back(solver.MakeNumVar(0.0, 1.0, std::string("pe_") + std::to_string(lambda_idx)));
        neg_errors.push_back(solver.MakeNumVar(0.0, 1.0, std::string("ne_") + std::to_string(lambda_idx)));
    }

    // constraints for weights times costs and errors
    for (std::size_t lambda_idx = 0; lambda_idx <  lambdas.size(); ++lambda_idx) {
        operations_research::MPConstraint* const ct = solver.MakeRowConstraint(cost_constraints[lambda_idx], cost_constraints[lambda_idx],
                                                            std::string("c") + std::to_string(lambda_idx));
        for (std::size_t action_idx = 0; action_idx <  num_actions; ++action_idx) {
            ct->SetCoefficient(action_weights[action_idx], action_values[lambda_idx][action_idx]);
        }
        for (std::size_t error_idx = 0; error_idx <  lambdas.size(); ++error_idx) {
            if (error_idx == lambda_idx) {
                ct->SetCoefficient(pos_errors[error_idx], 1.0);
                ct->SetCoefficient(neg_errors[error_idx], -1.0);
            } else {
                ct->SetCoefficient(pos_errors[error_idx], 0.0);
                ct->SetCoefficient(neg_errors[error_idx], 0.0);
            }
        }
    }

    // constraints for weight sum
    operations_research::MPConstraint* const ct_sum = solver.MakeRowConstraint(1.0, 1.0, "weight_sum");
    for (std::size_t action_idx = 0; action_idx <  num_actions; ++action_idx) {
            ct_sum->SetCoefficient(action_weights[action_idx], 1.0);
    }
    for (std::size_t error_idx = 0; error_idx <  lambdas.size(); ++error_idx) {
        ct_sum->SetCoefficient(pos_errors[error_idx], 0.0);
        ct_sum->SetCoefficient(neg_errors[error_idx], 0.0);
    }

    // Create the objective function, 3 * x + y.
    operations_research::MPObjective* const objective = solver.MutableObjective();
    for (std::size_t action_idx = 0; action_idx <  num_actions; ++action_idx) {
            objective->SetCoefficient(action_weights[action_idx], 0.0);
    }
    for (std::size_t error_idx = 0; error_idx <  lambdas.size(); ++error_idx) {
        objective->SetCoefficient(pos_errors[error_idx], lambdas[error_idx]);
        objective->SetCoefficient(neg_errors[error_idx], lambdas[error_idx]);
    }
    objective->SetMinimization();
    
    solver.Solve();

    Policy policy;
    const auto& cost_stats = cost_statistics.at(0).get_ucb_statistics();
    for ( const auto action : cost_stats) {
        policy[action.first] = 0.0f;
    }
    std::vector<int> discrete_probability_weights;
    for (std::size_t action_idx = 0; action_idx < action_weights.size(); ++action_idx) {
        policy[feasible_actions.at(action_idx)] = action_weights[action_idx]->solution_value();
        discrete_probability_weights.push_back(std::nearbyint(
                    action_weights[action_idx]->solution_value()*1000.0));
    }

    std::discrete_distribution<> action_dist(discrete_probability_weights.begin(),
                                        discrete_probability_weights.end());
    const auto sampled_action = feasible_actions.at(action_dist(random_generator));
    return std::make_pair(sampled_action, policy);
}


}

#endif