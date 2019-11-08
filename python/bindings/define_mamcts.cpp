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
      .def_readwrite("hypothesis_belief_tracker", &MctsParameters::hypothesis_belief_tracker)
      .def(py::pickle(
        [](const MctsParameters &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            py::dict d;
            d["RANDOM_SEED"] = p.RANDOM_SEED;
            d["DISCOUNT_FACTOR"] = p.DISCOUNT_FACTOR;
            d["hypothesis_statistic"] = p.hypothesis_statistic;
            d["uct_statistic"] = p.uct_statistic;
            d["random_heuristic"] = p.random_heuristic;
            d["hypothesis_belief_tracker"] = p.hypothesis_belief_tracker;
            return d;
        },
        [](py::dict d) { // __setstate__
            if (d.size() != 6)
                throw std::runtime_error("Invalid MctsParameters state!");

            /* Create a new C++ instance */
            MctsParameters p;
            p.RANDOM_SEED = d["RANDOM_SEED"].cast<unsigned int>();
            p.DISCOUNT_FACTOR = d["DISCOUNT_FACTOR"].cast<double>();
            p.hypothesis_statistic = d["hypothesis_statistic"].cast<MctsParameters::HypothesisStatisticParameters>();
            p.uct_statistic = d["uct_statistic"].cast<MctsParameters::UctStatisticParameters>();
            p.random_heuristic = d["random_heuristic"].cast<MctsParameters::RandomHeuristicParameters>();
            p.hypothesis_belief_tracker = d["hypothesis_belief_tracker"].cast<MctsParameters::HypothesisBeliefTrackerParameters>();
            return p;
        }
    ));

    py::class_<MctsParameters::RandomHeuristicParameters>(m, "MctsParametersRandomHeuristicParameters")
      .def(py::init<>())
      .def("__repr__", [](const MctsParameters::RandomHeuristicParameters &m) {
        return "mamcts.MctsParametersRandomHeuristicParameters";
      })
      .def_readwrite("MAX_SEARCH_TIME", &MctsParameters::RandomHeuristicParameters::MAX_SEARCH_TIME)
      .def_readwrite("MAX_NUMBER_OF_ITERATIONS",
               &MctsParameters::RandomHeuristicParameters::MAX_NUMBER_OF_ITERATIONS)
      .def(py::pickle(
        [](const MctsParameters::RandomHeuristicParameters &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            py::dict d;
            d["MAX_SEARCH_TIME"] = p.MAX_SEARCH_TIME;
            d["MAX_NUMBER_OF_ITERATIONS"] = p.MAX_NUMBER_OF_ITERATIONS;
            return d;
        },
        [](py::dict d) { // __setstate__
            if (d.size() != 2)
                throw std::runtime_error("Invalid RandomHeuristicParameters state!");

            /* Create a new C++ instance */
            MctsParameters::RandomHeuristicParameters p;
            p.MAX_SEARCH_TIME = d["MAX_SEARCH_TIME"].cast<double>();
            p.MAX_NUMBER_OF_ITERATIONS = d["MAX_NUMBER_OF_ITERATIONS"].cast<unsigned int>();
            return p;
        }
    ));


    py::class_<MctsParameters::UctStatisticParameters>(m ,"MctsParametersUctStatisticParametersParameters")
      .def(py::init<>())
      .def("__repr__", [](const MctsParameters::UctStatisticParameters &m) {
        return "mamcts.MctsParametersUctStatisticParameters";
      })
      .def_readwrite("LOWER_BOUND", &MctsParameters::UctStatisticParameters::LOWER_BOUND)
      .def_readwrite("UPPER_BOUND", &MctsParameters::UctStatisticParameters::UPPER_BOUND)
      .def_readwrite("EXPLORATION_CONSTANT", &MctsParameters::UctStatisticParameters::EXPLORATION_CONSTANT)
      .def(py::pickle(
        [](const MctsParameters::UctStatisticParameters &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            py::dict d;
            d["LOWER_BOUND"] = p.LOWER_BOUND;
            d["UPPER_BOUND"] = p.UPPER_BOUND;
            d["EXPLORATION_CONSTANT"] = p.EXPLORATION_CONSTANT;
            return d;
        },
        [](py::dict d) { // __setstate__
            if (d.size() != 3)
                throw std::runtime_error("Invalid UctStatisticParameters state!");

            /* Create a new C++ instance */
            MctsParameters::UctStatisticParameters p;
            p.LOWER_BOUND = d["LOWER_BOUND"].cast<double>();
            p.UPPER_BOUND = d["UPPER_BOUND"].cast<double>();
            p.EXPLORATION_CONSTANT = d["EXPLORATION_CONSTANT"].cast<double>();
            return p;
        }
    ));

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
                &MctsParameters::HypothesisStatisticParameters::EXPLORATION_CONSTANT)
      .def(py::pickle(
        [](const MctsParameters::HypothesisStatisticParameters &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            py::dict d;
            d["COST_BASED_ACTION_SELECTION"] = p.COST_BASED_ACTION_SELECTION;
            d["UPPER_COST_BOUND"] = p.UPPER_COST_BOUND;
            d["LOWER_COST_BOUND"] = p.LOWER_COST_BOUND;
            d["PROGRESSIVE_WIDENING_K"] = p.PROGRESSIVE_WIDENING_K;
            d["PROGRESSIVE_WIDENING_ALPHA"] = p.PROGRESSIVE_WIDENING_ALPHA;
            d["EXPLORATION_CONSTANT"] = p.EXPLORATION_CONSTANT;
            return d;
        },
        [](py::dict d) { // __setstate__
            if (d.size() != 6)
                throw std::runtime_error("Invalid HypothesisStatisticParameters state!");

            /* Create a new C++ instance */
            MctsParameters::HypothesisStatisticParameters p;
            p.COST_BASED_ACTION_SELECTION = d["COST_BASED_ACTION_SELECTION"].cast<bool>();
            p.UPPER_COST_BOUND = d["UPPER_COST_BOUND"].cast<double>();
            p.LOWER_COST_BOUND = d["LOWER_COST_BOUND"].cast<double>();
            p.PROGRESSIVE_WIDENING_K = d["PROGRESSIVE_WIDENING_K"].cast<double>();
            p.PROGRESSIVE_WIDENING_ALPHA = d["PROGRESSIVE_WIDENING_ALPHA"].cast<double>();
            p.EXPLORATION_CONSTANT = d["EXPLORATION_CONSTANT"].cast<double>();
            return p;
        }
    ));

    py::class_<MctsParameters::HypothesisBeliefTrackerParameters>(m ,"MctsParametersHypothesisBeliefTrackerParameters")
      .def(py::init<>())
      .def("__repr__", [](const MctsParameters::HypothesisBeliefTrackerParameters &m) {
        return "mamcts.MctsParametersHypothesisBeliefTrackerParameters";
      })
      .def_readwrite("RANDOM_SEED_HYPOTHESIS_SAMPLING", &MctsParameters::HypothesisBeliefTrackerParameters::RANDOM_SEED_HYPOTHESIS_SAMPLING)
      .def_readwrite("HISTORY_LENGTH", &MctsParameters::HypothesisBeliefTrackerParameters::HISTORY_LENGTH)
      .def_readwrite("PROBABILITY_DISCOUNT", &MctsParameters::HypothesisBeliefTrackerParameters::PROBABILITY_DISCOUNT)
      .def_readwrite("POSTERIOR_TYPE", &MctsParameters::HypothesisBeliefTrackerParameters::POSTERIOR_TYPE)
      .def_readwrite("FIXED_HYPOTHESIS_SET", &MctsParameters::HypothesisBeliefTrackerParameters::FIXED_HYPOTHESIS_SET)
      .def(py::pickle(
        [](const MctsParameters::HypothesisBeliefTrackerParameters &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            py::dict d;
            d["RANDOM_SEED_HYPOTHESIS_SAMPLING"] = p.RANDOM_SEED_HYPOTHESIS_SAMPLING;
            d["HISTORY_LENGTH"] = p.HISTORY_LENGTH;
            d["PROBABILITY_DISCOUNT"] = p.PROBABILITY_DISCOUNT;
            d["POSTERIOR_TYPE"] = p.POSTERIOR_TYPE;
            d["FIXED_HYPOTHESIS_SET"] = p.FIXED_HYPOTHESIS_SET;
            return d;
        },
        [](py::dict d) { // __setstate__
            if (d.size() != 5)
                throw std::runtime_error("Invalid HypothesisBeliefTrackerParameters state!");

            /* Create a new C++ instance */
            MctsParameters::HypothesisBeliefTrackerParameters p;
            p.RANDOM_SEED_HYPOTHESIS_SAMPLING = d["RANDOM_SEED_HYPOTHESIS_SAMPLING"].cast<unsigned int>();
            p.HISTORY_LENGTH = d["HISTORY_LENGTH"].cast<unsigned int>();
            p.PROBABILITY_DISCOUNT = d["PROBABILITY_DISCOUNT"].cast<float>();
            p.POSTERIOR_TYPE = d["POSTERIOR_TYPE"].cast<int>();
            p.FIXED_HYPOTHESIS_SET = d["FIXED_HYPOTHESIS_SET"].cast<
                        std::unordered_map<unsigned char, unsigned int>>();
            return p;
        }
    ));

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