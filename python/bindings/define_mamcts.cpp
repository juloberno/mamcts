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

void define_mamcts(py::module m)
{   
    py::class_<MctsParameters>(m, "MctsParameters")
      .def(py::init<>())
      .def("__repr__", [](const MctsParameters &m) {
        return "mamcts.MctsParameters";
      })
      .def_readwrite("RANDOM_SEED", &MctsParameters::RANDOM_SEED)
      .def_readwrite("DISCOUNT_FACTOR", &MctsParameters::DISCOUNT_FACTOR)
      .def_readwrite("hypothesis_statistic", &MctsParameters::hypothesis_statistic)
      .def_readwrite("uct_statistic", &MctsParameters::uct_statistic)
      .def_readwrite("random_heuristic", &MctsParameters::random_heuristic)
      .def_readwrite("hypothesis_belief_tracker", &MctsParameters::hypothesis_belief_tracker);

    py::class_<MctsParameters::RandomHeuristicParameters>(m, "MctsParametersRandomHeuristicParameters")
      .def(py::init<>())
      .def("__repr__", [](const MctsParameters::RandomHeuristicParameters &m) {
        return "mamcts.MctsParametersRandomHeuristicParameters";
      })
      .def_readwrite("MAX_SEARCH_TIME", &MctsParameters::RandomHeuristicParameters::MAX_SEARCH_TIME)
      .def_readwrite("MAX_NUMBER_OF_ITERATIONS",
               &MctsParameters::RandomHeuristicParameters::MAX_NUMBER_OF_ITERATIONS);


    py::class_<MctsParameters::UctStatisticParameters>(m ,"MctsParametersUctStatisticParametersParameters")
      .def(py::init<>())
      .def("__repr__", [](const MctsParameters::UctStatisticParameters &m) {
        return "mamcts.MctsParametersUctStatisticParameters";
      })
      .def_readwrite("LOWER_BOUND", &MctsParameters::UctStatisticParameters::LOWER_BOUND)
      .def_readwrite("UPPER_BOUND", &MctsParameters::UctStatisticParameters::UPPER_BOUND)
      .def_readwrite("EXPLORATION_CONSTANT", &MctsParameters::UctStatisticParameters::EXPLORATION_CONSTANT);

    py::class_<MctsParameters::HypothesisStatisticParameters>(m, "MctsParametersHypothesisStatisticParametersParameters")
      .def(py::init<>())
      .def("__repr__", [](const MctsParameters::HypothesisStatisticParameters &m) {
        return "mamcts.MctsParametersHypothesisStatisticParameters";
      })
      .def_readwrite("COST_BASED_ACTION_SELECTION",
                 &MctsParameters::HypothesisStatisticParameters::COST_BASED_ACTION_SELECTION)
      .def_readwrite("UPPER_COST_BOUND",
                 &MctsParameters::HypothesisStatisticParameters::UPPER_COST_BOUND)
      .def_readwrite("LOWER_COST_BOUND",
                &MctsParameters::HypothesisStatisticParameters::LOWER_COST_BOUND)
      .def_readwrite("PROGRESSIVE_WIDENING_K",
                &MctsParameters::HypothesisStatisticParameters::PROGRESSIVE_WIDENING_K)
      .def_readwrite("PROGRESSIVE_WIDENING_ALPHA",
                &MctsParameters::HypothesisStatisticParameters::PROGRESSIVE_WIDENING_ALPHA)
      .def_readwrite("EXPLORATION_CONSTANT",
                &MctsParameters::HypothesisStatisticParameters::EXPLORATION_CONSTANT);

    py::class_<MctsParameters::HypothesisBeliefTrackerParameters>(m ,"MctsParametersHypothesisBeliefTrackerParameters")
      .def(py::init<>())
      .def("__repr__", [](const MctsParameters::HypothesisBeliefTrackerParameters &m) {
        return "mamcts.MctsParametersHypothesisBeliefTrackerParameters";
      })
      .def_readwrite("RANDOM_SEED_HYPOTHESIS_SAMPLING", &MctsParameters::HypothesisBeliefTrackerParameters::RANDOM_SEED_HYPOTHESIS_SAMPLING)
      .def_readwrite("HISTORY_LENGTH", &MctsParameters::HypothesisBeliefTrackerParameters::HISTORY_LENGTH)
      .def_readwrite("PROBABILITY_DISCOUNT", &MctsParameters::HypothesisBeliefTrackerParameters::PROBABILITY_DISCOUNT)
      .def_readwrite("POSTERIOR_TYPE", &MctsParameters::HypothesisBeliefTrackerParameters::POSTERIOR_TYPE);

    using mcts1 = Mcts<CrossingState<int>, UctStatisticParameters, HypothesisStatisticParameters, RandomHeuristicParameters>;
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