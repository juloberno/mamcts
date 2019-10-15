// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#include "python/bindings/define_environments.hpp"
#include "mcts/mcts.h"
#include "environments/crossing_state.h"

namespace py = pybind11;
using namespace mcts;

void define_environments(py::module m)
{
    py::class_<Viewer,
             PyViewer,
             std::shared_ptr<Viewer>>(m, "Viewer")
      .def("drawPoint", &Viewer::drawPoint)
      .def("drawLine", &Viewer::drawLine);

    py::class_<CrossingStateParameters,
             std::shared_ptr<CrossingStateParameters>>(m, "CrossingStateParameters")
      .def_readonly_static("MAX_VELOCITY_EGO", &CrossingStateParameters::MAX_VELOCITY_EGO)
      .def_readonly_static("MIN_VELOCITY_EGO",&CrossingStateParameters::MIN_VELOCITY_EGO)
      .def_readonly_static("MIN_VELOCITY_OTHER",&CrossingStateParameters::MIN_VELOCITY_OTHER)
      .def_readonly_static("MIN_VELOCITY_OTHER",&CrossingStateParameters::MIN_VELOCITY_OTHER)
      .def_readonly_static("EGO_GOAL_POS",&CrossingStateParameters::EGO_GOAL_POS)
      .def_readonly_static("CHAIN_LENGTH",&CrossingStateParameters::CHAIN_LENGTH)
      .def_property_readonly("CROSSING_POINT",&CrossingStateParameters::CROSSING_POINT)
      .def_property_readonly("NUM_EGO_ACTIONS",&CrossingStateParameters::NUM_EGO_ACTIONS)
      .def_property_readonly("NUM_OTHER_ACTIONS",&CrossingStateParameters::NUM_OTHER_ACTIONS)
      .def("__repr__", [](const CrossingStateParameters &m) {
        return "mamcts.CrossingStateParameters";
      });

    py::class_<AgentState,
             std::shared_ptr<AgentState>>(m, "AgentCrossingState")
      .def("__repr__", [](const AgentState &m) {
        return "mamcts.AgentCrossingState";
      })
      .def_readonly("position", &AgentState::x_pos)
      .def_readonly("last_action", &AgentState::last_action);

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
      .def_property_readonly("other_agents_states", &CrossingState::get_agent_states)
      .def_property_readonly("ego_agent_state", &CrossingState::get_ego_state)
      .def("add_hypothesis", &CrossingState::add_hypothesis);
}