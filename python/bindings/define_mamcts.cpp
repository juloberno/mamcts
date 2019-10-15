// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#include "mcts/mcts.h"
#include "mcts/random_generator.h"
#include "python/bindings/define_mamcts.hpp"
#include "environments/crossing_state.h"
#include "mcts/statistics/uct_statistic.h"
#include "mcts/hypothesis/hypothesis_statistic.h"
#include "mcts/heuristics/random_heuristic.h"

namespace py = pybind11;
using namespace mcts;

std::mt19937 mcts::RandomGenerator::random_generator_;


void define_mamcts(py::module m)
{
    using mcts1 = Mcts<CrossingState, UctStatistic, HypothesisStatistic, RandomHeuristic>;
    py::class_<mcts1,
             std::shared_ptr<mcts1>>(m, "MctsCrossingStateUctUct")
      .def(py::init<>())
      .def("__repr__", [](const mcts1 &m) {
        return "mamcts.MctsCrossingStateUctUct";
      });
}