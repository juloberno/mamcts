// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef PYTHON_DEFINE_CROSSING_STATE_HPP_
#define PYTHON_DEFINE_CROSSING_STATE_HPP_

#include "python/bindings/common.hpp"
#include "environments/crossing_state.h"
#include "environments/crossing_state_episode_runner.h"

namespace py = pybind11;
using namespace mcts;

template <typename Domain>
void define_crossing_state(py::module m, std::string suffix) {

    std::string name1 = "CrossingStateParameters" + suffix;
    py::class_<CrossingStateParameters<Domain>,
             std::shared_ptr<CrossingStateParameters<Domain>>>(m, name1.c_str())
      .def_readwrite("NUM_OTHER_AGENTS", &CrossingStateParameters<Domain>::NUM_OTHER_AGENTS)
      .def_readwrite("OTHER_AGENTS_POLICY_RANDOM_SEED", &CrossingStateParameters<Domain>::OTHER_AGENTS_POLICY_RANDOM_SEED)
      .def_readwrite("MAX_VELOCITY_EGO", &CrossingStateParameters<Domain>::MAX_VELOCITY_EGO)
      .def_readwrite("MIN_VELOCITY_EGO",&CrossingStateParameters<Domain>::MIN_VELOCITY_EGO)
      .def_readwrite("MIN_VELOCITY_OTHER",&CrossingStateParameters<Domain>::MIN_VELOCITY_OTHER)
      .def_readwrite("MIN_VELOCITY_OTHER",&CrossingStateParameters<Domain>::MIN_VELOCITY_OTHER)
      .def_readwrite("EGO_GOAL_POS",&CrossingStateParameters<Domain>::EGO_GOAL_POS)
      .def_readwrite("CHAIN_LENGTH",&CrossingStateParameters<Domain>::CHAIN_LENGTH)
      .def_property_readonly("CROSSING_POINT",&CrossingStateParameters<Domain>::CROSSING_POINT)
      .def_property_readonly("NUM_EGO_ACTIONS",&CrossingStateParameters<Domain>::NUM_EGO_ACTIONS)
      .def_property_readonly("NUM_OTHER_ACTIONS",&CrossingStateParameters<Domain>::NUM_OTHER_ACTIONS)
      .def("__repr__", [&](const CrossingStateParameters<Domain> &m) {
        return "";//add_suffix("mamcts.CrossingStateParameters");
      });

    std::string name2 = "AgentCrossingState" + suffix;
    py::class_<AgentState<Domain>,
             std::shared_ptr<AgentState<Domain>>>(m, name2.c_str())
      .def("__repr__", [&](const AgentState<Domain> &m) {
        return "";// add_suffix("mamcts.AgentCrossingState");
      })
      .def_readonly("position", &AgentState<Domain>::x_pos)
      .def_readonly("last_action", &AgentState<Domain>::last_action);

    std::string name3 =  "AgentPolicyCrossingState" + suffix;
    py::class_<AgentPolicyCrossingState<Domain>,
             std::shared_ptr<AgentPolicyCrossingState<Domain>>>(m, name3.c_str())
      .def(py::init<const std::pair<int, int>&, const CrossingStateParameters<Domain>&>())
      .def("info", &AgentPolicyCrossingState<Domain>::info)
      .def("__repr__", [&](const AgentPolicyCrossingState<Domain> &m) {
        return "";// add_suffix("mamcts.AgentPolicyCrossingState");
      });

    std::string name4 = "CrossingState" + suffix;
    py::class_<CrossingState<Domain>,
             std::shared_ptr<CrossingState<Domain>>>(m, name4.c_str())
      .def(py::init<const std::unordered_map<AgentIdx, HypothesisId>&, const CrossingStateParameters<Domain>&>())
      .def("__repr__", [&](const CrossingState<Domain> &m) {
        return "";// add_suffix("mamcts.CrossingState");
      })
      .def("draw", &CrossingState<Domain>::draw)
      .def_property_readonly("other_agents_states", &CrossingState<Domain>::get_agent_states)
      .def_property_readonly("ego_agent_state", &CrossingState<Domain>::get_ego_state)
      .def("add_hypothesis", &CrossingState<Domain>::add_hypothesis);

    std::string name5 = "CrossingStateEpisodeRunner" + suffix;
    py::class_<CrossingStateEpisodeRunner<Domain>,
             std::shared_ptr<CrossingStateEpisodeRunner<Domain>>>(m, name5.c_str())
      .def(py::init<const std::unordered_map<AgentIdx, AgentPolicyCrossingState<Domain>>&,
                            const std::vector<AgentPolicyCrossingState<Domain>>&,
                            const mcts::MctsParameters&,
                            const CrossingStateParameters<Domain>&,
                            const unsigned int&,
                            const unsigned int&,
                            const unsigned int&,
                            mcts::Viewer*>())
      .def("__repr__", [&](const CrossingStateEpisodeRunner<Domain> &m) {
        return "";// add_suffix("mamcts.CrossingStateEpisodeRunner");
      })
      .def("step", &CrossingStateEpisodeRunner<Domain>::step)
      .def("run", &CrossingStateEpisodeRunner<Domain>::run);

    std::string name6 = "CrossingStateDefaultParameters" + suffix;
    m.def(name6.c_str(), &default_crossing_state_parameters<Domain>);
}

#endif // PYTHON_DEFINE_CROSSING_STATE_HPP_