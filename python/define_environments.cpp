// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#include "python/define_environments.hpp"
#include "mcts/mcts.h"
#include "mcts/random_generator.h"
#include "environments/crossing_state.h"
#include "mcts/statistics/uct_statistic.h"
#include "mcts/hypothesis/hypothesis_statistic.h"
#include "mcts/heuristics/random_heuristic.h"

namespace py = pybind11;
using namespace mcts;

void define_environments(py::module m)
{
    py::class_<AgentPolicyCrossingState,
             std::shared_ptr<AgentPolicyCrossingState>>(m, "AgentPolicyCrossingState")
      .def(py::init<const std::pair<int, int>&>())
      .def("__repr__", [](const AgentPolicyCrossingState &m) {
        return "mamcts.AgentPolicyCrossingState";
      });

    py::class_<CrossingState,
             std::shared_ptr<CrossingState>>(m, "CrossingState")
      .def(py::init<const std::unordered_map<AgentIdx, HypothesisId>&>())
      .def("__repr__", [](const CrossingState &m) {
        return "mamcts.CrossingState";
      })
      .def("add_hypothesis", &CrossingState::add_hypothesis);
}