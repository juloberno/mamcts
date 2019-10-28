// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#include "mcts/mcts.h"
#include "mcts/random_generator.h"
#include "python/bindings/define_mamcts.hpp"
#include "environments/crossing_state.h"
#include "mcts/hypothesis/hypothesis_belief_tracker.h"
#include "mcts/statistics/uct_statistic.h"
#include "mcts/hypothesis/hypothesis_statistic.h"
#include "mcts/heuristics/random_heuristic.h"

namespace py = pybind11;
using namespace mcts;

std::mt19937 mcts::RandomGenerator::random_generator_;


void define_mamcts(py::module m)
{   
    py::class_<MctsParameters>(m, "MctsParameters")
      .def(py::init<>())
      .def("__repr__", [](const MctsParameters &m) {
        return "mamcts.MctsParameters";
      })
      .def_readwrite("DISCOUNT_FACTOR", &MctsParameters::DISCOUNT_FACTOR)
      .def_readwrite("hypothesis_statistic", &MctsParameters::hypothesis_statistic)
      .def_readwrite("uct_statistic", &MctsParameters::uct_statistic)
      .def_readwrite("random_heuristic", &MctsParameters::random_heuristic);

    py::class_<MctsParameters::RandomHeuristic>(m, "MctsParametersRandomHeuristic")
      .def(py::init<>())
      .def("__repr__", [](const MctsParameters::RandomHeuristic &m) {
        return "mamcts.MctsParametersRandomHeuristic";
      })
      .def_readwrite("MAX_SEARCH_TIME", &MctsParameters::RandomHeuristic::MAX_SEARCH_TIME)
      .def_readwrite("MAX_NUMBER_OF_ITERATIONS",
               &MctsParameters::RandomHeuristic::MAX_NUMBER_OF_ITERATIONS);


    py::class_<MctsParameters::UctStatistic>(m ,"MctsParametersUctStatistic")
      .def(py::init<>())
      .def("__repr__", [](const MctsParameters::UctStatistic &m) {
        return "mamcts.MctsParametersUctStatistic";
      })
      .def_readwrite("LOWER_BOUND", &MctsParameters::UctStatistic::LOWER_BOUND)
      .def_readwrite("UPPER_BOUND", &MctsParameters::UctStatistic::UPPER_BOUND)
      .def_readwrite("EXPLORATION_CONSTANT", &MctsParameters::UctStatistic::EXPLORATION_CONSTANT);

    py::class_<MctsParameters::HypothesisStatistic>(m, "MctsParametersHypothesisStatistic")
      .def(py::init<>())
      .def("__repr__", [](const MctsParameters::HypothesisStatistic &m) {
        return "mamcts.MctsParametersHypothesisStatistic";
      })
      .def_readwrite("COST_BASED_ACTION_SELECTION",
                 &MctsParameters::HypothesisStatistic::COST_BASED_ACTION_SELECTION)
      .def_readwrite("UPPER_COST_BOUND",
                 &MctsParameters::HypothesisStatistic::UPPER_COST_BOUND)
      .def_readwrite("LOWER_COST_BOUND",
                &MctsParameters::HypothesisStatistic::LOWER_COST_BOUND)
      .def_readwrite("PROGRESSIVE_WIDENING_K",
                &MctsParameters::HypothesisStatistic::PROGRESSIVE_WIDENING_K)
      .def_readwrite("PROGRESSIVE_WIDENING_ALPHA",
                &MctsParameters::HypothesisStatistic::PROGRESSIVE_WIDENING_ALPHA)
      .def_readwrite("EXPLORATION_CONSTANT",
                &MctsParameters::HypothesisStatistic::EXPLORATION_CONSTANT);

    using mcts1 = Mcts<CrossingState<int>, UctStatistic, HypothesisStatistic, RandomHeuristic>;
    py::class_<mcts1,
             std::shared_ptr<mcts1>>(m, "MctsCrossingStateIntUctUct")
      .def(py::init<const MctsParameters&>())
      .def("__repr__", [](const mcts1 &m) {
        return "mamcts.MctsCrossingStateIntUctUct";
      });

    py::class_<HypothesisBeliefTracker> belief_tracker(m, "HypothesisBeliefTracker");

    py::enum_<HypothesisBeliefTracker::PosteriorType>(belief_tracker , "PosteriorType")
      .value("PRODUCT", HypothesisBeliefTracker::PosteriorType::PRODUCT)
      .value("SUM", HypothesisBeliefTracker::PosteriorType::SUM)
      .export_values();
}