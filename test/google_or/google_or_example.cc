#include "ortools/linear_solver/linear_solver.h"
#include <vector>
#include <unordered_map>

namespace operations_research {

void SimpleLpProgram() {
    std::vector<std::vector<double>> action_values{{0.6, 0.1, 0.2, 0.3, 0.1, 0.8, 0.3}, {0.1, 0.7, 0.01, 0.3, 0.2, 0.1, 0.1}};
    std::vector<double> lambdas{0.2, 0.3};
    std::vector<double> constraints{0.3, 0.0};


    auto start = std::chrono::high_resolution_clock::now();
    // Create the linear solver with the GLOP backend.
    MPSolver solver("simple_lp_program", MPSolver::GLOP_LINEAR_PROGRAMMING);

    // Create the variables x and y.
    std::vector<MPVariable* > action_weights;
    std::vector<MPVariable* > pos_errors;
    std::vector<MPVariable* > neg_errors;

    for (std::size_t action_idx = 0; action_idx <  action_values[0].size(); ++action_idx) {
        action_weights.push_back(solver.MakeNumVar(0.0, 1.0, std::string("w_") + std::to_string(action_idx)));
    }

    for (std::size_t lambda_idx = 0; lambda_idx <  lambdas.size(); ++lambda_idx) {
        pos_errors.push_back(solver.MakeNumVar(0.0, 1.0, std::string("pe_") + std::to_string(lambda_idx)));
        neg_errors.push_back(solver.MakeNumVar(0.0, 1.0, std::string("ne_") + std::to_string(lambda_idx)));
    }

    LOG(INFO) << "Number of variables = " << solver.NumVariables();

    // constraints for weights times costs and errors
    for (std::size_t lambda_idx = 0; lambda_idx <  lambdas.size(); ++lambda_idx) {
        MPConstraint* const ct = solver.MakeRowConstraint(constraints[lambda_idx], constraints[lambda_idx],
                                                            std::string("c") + std::to_string(lambda_idx));
        for (std::size_t action_idx = 0; action_idx <  action_values[lambda_idx].size(); ++action_idx) {
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
    MPConstraint* const ct_sum = solver.MakeRowConstraint(1.0, 1.0, "weight_sum");
    for (std::size_t action_idx = 0; action_idx <  action_values[0].size(); ++action_idx) {
            ct_sum->SetCoefficient(action_weights[action_idx], 1.0);
    }
    for (std::size_t error_idx = 0; error_idx <  lambdas.size(); ++error_idx) {
        ct_sum->SetCoefficient(pos_errors[error_idx], 0.0);
        ct_sum->SetCoefficient(neg_errors[error_idx], 0.0);
    }

    LOG(INFO) << "Number of constraints = " << solver.NumConstraints();

    // Create the objective function, 3 * x + y.
    MPObjective* const objective = solver.MutableObjective();
    for (std::size_t action_idx = 0; action_idx <  action_values[0].size(); ++action_idx) {
            objective->SetCoefficient(action_weights[action_idx], 0.0);
    }
    for (std::size_t error_idx = 0; error_idx <  lambdas.size(); ++error_idx) {
        objective->SetCoefficient(pos_errors[error_idx], lambdas[error_idx]);
        objective->SetCoefficient(neg_errors[error_idx], lambdas[error_idx]);
    }
    objective->SetMinimization();

    
    solver.Solve();
    unsigned int duration =
      std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::high_resolution_clock::now() - start ).count();

    LOG(INFO) << "Solution:" << std::endl;
    LOG(INFO) << "Objective value = " << objective->Value();
    LOG(INFO) << "Runtime: " << duration; 
    for (std::size_t action_idx = 0; action_idx <  action_values[0].size(); ++action_idx) {
         LOG(INFO) << "Solution a" << action_idx << " = "  << action_weights[action_idx]->solution_value(); 
    }
    for (std::size_t error_idx = 0; error_idx <  lambdas.size(); ++error_idx) {
        LOG(INFO) << "Solution ne" << error_idx << " = "  << neg_errors[error_idx]->solution_value(); 
        LOG(INFO) << "Solution pe" << error_idx << " = "  << pos_errors[error_idx]->solution_value(); 
    }
    }
    }  // namespace operations_research

int main() {
  operations_research::SimpleLpProgram();
  return EXIT_SUCCESS;
}
