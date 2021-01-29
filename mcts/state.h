// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef MCTS_STATE_H
#define MCTS_STATE_H

#include <memory>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <functional>
#include <iostream>
#include <valarray>
#include "common.h"


namespace mcts {


typedef std::size_t ActionIdx;
typedef unsigned int AgentIdx;
typedef std::vector<ActionIdx> JointAction;

typedef double Reward;
typedef double Cost;
typedef std::vector<Cost> EgoCosts;

template <class T>
std::unordered_map<ActionIdx, T> ActionMapping(ActionIdx num_actions, const T& init) {
    std::unordered_map<ActionIdx, T> ego_action_map;
    for(ActionIdx ai = 0; ai < num_actions; ++ai) {
        ego_action_map[ai] = init;
    }
    return ego_action_map;
}

template <class T>
std::unordered_map<AgentIdx, T> AgentMapping(std::vector<AgentIdx> agent_idxs, const T& init) {
    std::unordered_map<AgentIdx, T> agent_map;
    for(const AgentIdx& ai : agent_idxs) {
        agent_map[ai] = init;
    }
    return agent_map;
}

    template <typename T>
inline std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b)
    {
        MCTS_EXPECT_TRUE(a.size() == b.size());
        std::vector<T> result;
        result.reserve(a.size());

        std::transform(a.begin(), a.end(), b.begin(),
                       std::back_inserter(result), std::plus<T>());
        return result;
    }

    template <typename T>
inline std::vector<T>& operator+=(std::vector<T>& a, const std::vector<T>& b)
    {
        MCTS_EXPECT_TRUE(a.size() == b.size());
        for(uint i=0; i<a.size(); i++){
            a[i] = a[i] + b[i];
        }
        return a;
    }

inline std::ostream& operator<<(std::ostream& os, const JointAction& a)
    {
        os << "[";
        for (auto it = a.begin(); it != a.end(); ++it)
            os << (int)(*it) << " ";
        os << "]";
        return os;
    }

inline std::ostream& operator<<(std::ostream& os, const std::vector<Reward>& r)
    {
        os << "[";
        for (auto it = r.begin(); it != r.end(); ++it)
            os << Reward(*it) << " ";
        os << "]";
        return os;
    }

inline std::ostream& operator<<(std::ostream& os, const std::unordered_map<ActionIdx, double>& v)
    {
        os << "[";
        for (const auto& val : v)
            os << val.first << ": " << val.second << ", ";
        os << "]";
        return os;
    }


template<typename Implementation>
class StateInterface {
public:

    std::shared_ptr<Implementation> execute(const JointAction &joint_action,
                                            std::vector<Reward>& rewards,
                                            EgoCosts& ego_cost) const;

    std::shared_ptr<Implementation> clone() const;

    ActionIdx get_num_actions(AgentIdx agent_idx) const;

    bool is_terminal() const;

    const std::vector<AgentIdx> get_other_agent_idx() const;

    const AgentIdx get_ego_agent_idx() const;

    const AgentIdx get_num_agents() const;

    const std::size_t get_num_costs() const;

    static const AgentIdx ego_agent_idx;

    std::string sprintf() const;

    double get_execution_step_length() const;

    ~StateInterface() {};

    static const Implementation& cast();

    CRTP_INTERFACE(Implementation)
    CRTP_CONST_INTERFACE(Implementation)

};


template<typename Implementation>
inline std::shared_ptr<Implementation> StateInterface<Implementation>::execute(const JointAction &joint_action,
                                                                               std::vector<Reward>& rewards,
                                                                               EgoCosts& ego_cost) const {
   return impl().execute(joint_action, rewards);
}

template<typename Implementation>
inline std::shared_ptr<Implementation> StateInterface<Implementation>::clone() const {
 return impl().clone();
}

template<typename Implementation>
inline ActionIdx StateInterface<Implementation>::get_num_actions(AgentIdx agent_idx) const {
    return impl().get_num_actions(agent_idx);
}

template<typename Implementation>
inline const std::size_t StateInterface<Implementation>::get_num_costs() const {
    return impl().get_num_costs();
}

template<typename Implementation>
inline bool StateInterface<Implementation>::is_terminal() const {
    return impl().is_terminal();
}

template<typename Implementation>
inline const std::vector<AgentIdx> StateInterface<Implementation>::get_other_agent_idx() const {
    return impl().get_other_agent_idx();
}

template<typename Implementation>
inline const AgentIdx StateInterface<Implementation>::get_ego_agent_idx() const {
    return impl().get_ego_agent_idx();
}

template<typename Implementation>
inline const AgentIdx StateInterface<Implementation>::get_num_agents() const {
    return impl().get_other_agent_idx().size() + 1; // num other agents + ego agent
}


template<typename Implementation>
inline std::string StateInterface<Implementation>::sprintf() const {
    return impl().sprintf();
}

template<typename Implementation>
inline double StateInterface<Implementation>::get_execution_step_length() const {
    return impl().get_execution_step_length();
}

template<typename Implementation>
const AgentIdx StateInterface<Implementation>::ego_agent_idx = 0;



} // namespace mcts

#endif