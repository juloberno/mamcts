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
      .def(py::init<>())
      .def_readwrite("NUM_OTHER_AGENTS", &CrossingStateParameters<Domain>::NUM_OTHER_AGENTS)
      .def_readwrite("OTHER_AGENTS_POLICY_RANDOM_SEED", &CrossingStateParameters<Domain>::OTHER_AGENTS_POLICY_RANDOM_SEED)
      .def_readwrite("COST_ONLY_COLLISION", &CrossingStateParameters<Domain>::COST_ONLY_COLLISION)
      .def_readwrite("MAX_VELOCITY_EGO", &CrossingStateParameters<Domain>::MAX_VELOCITY_EGO)
      .def_readwrite("MIN_VELOCITY_EGO",&CrossingStateParameters<Domain>::MIN_VELOCITY_EGO)
      .def_readwrite("MIN_VELOCITY_OTHER",&CrossingStateParameters<Domain>::MIN_VELOCITY_OTHER)
      .def_readwrite("MAX_VELOCITY_OTHER",&CrossingStateParameters<Domain>::MAX_VELOCITY_OTHER)
      .def_readwrite("EGO_GOAL_POS",&CrossingStateParameters<Domain>::EGO_GOAL_POS)
      .def_readwrite("CHAIN_LENGTH",&CrossingStateParameters<Domain>::CHAIN_LENGTH)
      .def_property_readonly("CROSSING_POINT",&CrossingStateParameters<Domain>::CROSSING_POINT)
      .def_property_readonly("NUM_EGO_ACTIONS",&CrossingStateParameters<Domain>::NUM_EGO_ACTIONS)
      .def_readwrite("NUM_OTHER_ACTIONS",&CrossingStateParameters<Domain>::NUM_OTHER_ACTIONS)
      .def_readwrite("REWARD_COLLISION",&CrossingStateParameters<Domain>::REWARD_COLLISION)
      .def_readwrite("REWARD_GOAL_REACHED",&CrossingStateParameters<Domain>::REWARD_GOAL_REACHED)
      .def_readwrite("REWARD_STEP",&CrossingStateParameters<Domain>::REWARD_STEP)
      .def("__repr__", [](const CrossingStateParameters<Domain> &m) {
        return typeid(m).name();
      })
      .def(py::pickle(
        [](const CrossingStateParameters<Domain> &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            py::dict d;
            d["NUM_OTHER_AGENTS"] = p.NUM_OTHER_AGENTS;
            d["OTHER_AGENTS_POLICY_RANDOM_SEED"] = p.OTHER_AGENTS_POLICY_RANDOM_SEED;
            d["COST_ONLY_COLLISION"] = p.COST_ONLY_COLLISION;
            d["MAX_VELOCITY_EGO"] = p.MAX_VELOCITY_EGO;
            d["MIN_VELOCITY_EGO"] = p.MIN_VELOCITY_EGO;
            d["MIN_VELOCITY_OTHER"] = p.MIN_VELOCITY_OTHER;
            d["MAX_VELOCITY_OTHER"] = p.MAX_VELOCITY_OTHER;
            d["EGO_GOAL_POS"] = p.EGO_GOAL_POS;
            d["CHAIN_LENGTH"] = p.CHAIN_LENGTH;
            d["NUM_OTHER_ACTIONS"] = p.NUM_OTHER_ACTIONS;
            d["REWARD_COLLISION"] = p.REWARD_COLLISION;
            d["REWARD_GOAL_REACHED"] = p.REWARD_GOAL_REACHED;
            d["REWARD_STEP"] = p.REWARD_STEP;
            return d;
        },
        [](py::dict d) { // __setstate__
            if (d.size() != 13)
                throw std::runtime_error("Invalid CrossingStateParameters state!");

            /* Create a new C++ instance */
            CrossingStateParameters<Domain> p;
            p.NUM_OTHER_AGENTS = d["NUM_OTHER_AGENTS"].cast<unsigned int>();
            p.OTHER_AGENTS_POLICY_RANDOM_SEED = d["OTHER_AGENTS_POLICY_RANDOM_SEED"].cast<unsigned int>();
            p.COST_ONLY_COLLISION = d["COST_ONLY_COLLISION"].cast<bool>();
            p.MAX_VELOCITY_EGO = d["MAX_VELOCITY_EGO"].cast<Domain>();
            p.MIN_VELOCITY_EGO = d["MIN_VELOCITY_EGO"].cast<Domain>();
            p.MIN_VELOCITY_OTHER = d["MIN_VELOCITY_OTHER"].cast<Domain>();
            p.MAX_VELOCITY_OTHER = d["MAX_VELOCITY_OTHER"].cast<Domain>();
            p.EGO_GOAL_POS = d["EGO_GOAL_POS"].cast<Domain>();
            p.CHAIN_LENGTH = d["CHAIN_LENGTH"].cast<Domain>();
            p.NUM_OTHER_ACTIONS = d["NUM_OTHER_ACTIONS"].cast<unsigned int>();
            p.REWARD_COLLISION = d["REWARD_COLLISION"].cast<Reward>();
            p.REWARD_GOAL_REACHED = d["REWARD_GOAL_REACHED"].cast<Reward>();
            p.REWARD_STEP = d["REWARD_STEP"].cast<Reward>();

            return p;
        }
    ));

    std::string name2 = "AgentCrossingState" + suffix;
    py::class_<AgentState<Domain>,
             std::shared_ptr<AgentState<Domain>>>(m, name2.c_str())
      .def("__repr__", [](const AgentState<Domain> &m) {
        return typeid(m).name();
      })
      .def_readonly("position", &AgentState<Domain>::x_pos)
      .def_readonly("last_action", &AgentState<Domain>::last_action);

    std::string name3 =  "AgentPolicyCrossingState" + suffix;
    py::class_<AgentPolicyCrossingState<Domain>,
             std::shared_ptr<AgentPolicyCrossingState<Domain>>>(m, name3.c_str())
      .def(py::init<const std::pair<Domain, Domain>&, const CrossingStateParameters<Domain>&>())
      .def("info", &AgentPolicyCrossingState<Domain>::info)
      .def("__repr__", [](const AgentPolicyCrossingState<Domain> &m) {
        return typeid(m).name();
      });

    std::string name4 = "CrossingState" + suffix;
    py::class_<CrossingState<Domain>,
             std::shared_ptr<CrossingState<Domain>>>(m, name4.c_str())
      .def(py::init<const std::unordered_map<AgentIdx, HypothesisId>&, const CrossingStateParameters<Domain>&>())
      .def("__repr__", [](const CrossingState<Domain> &m) {
        return typeid(m).name();
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
      .def("__repr__", [](const CrossingStateEpisodeRunner<Domain> &m) {
        return typeid(m).name();
      })
      .def("step", &CrossingStateEpisodeRunner<Domain>::step)
      .def("run", &CrossingStateEpisodeRunner<Domain>::run);

    std::string name6 = "CrossingStateDefaultParameters" + suffix;
    m.def(name6.c_str(), &default_crossing_state_parameters<Domain>);
}

#endif // PYTHON_DEFINE_CROSSING_STATE_HPP_