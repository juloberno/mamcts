// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef MCTS_H
#define MCTS_H


#include "stage_node.h"
#include "heuristic.h"
#include "hypothesis/hypothesis_belief_tracker.h"
#include <chrono>  // for high_resolution_clock
#include "common.h"
#include "mcts_parameters.h"
#include <string>


namespace mcts {

template<class StateTransitionInfo> 
using MctsEdgeInfo = std::tuple<AgentIdx, unsigned int, ActionIdx, ActionWeight, StateTransitionInfo>;

/*
 * @tparam S State Interface
 * @tparam SE Selection & Expandsion Strategy Ego
 * @tparam SO Selection & Expansion Strategy Others
 * @tparam SR Strategy Rollout
 */
template<class S, class SE, class SO, class H>
class Mcts {

public:
    using StageNodeSPtr = std::shared_ptr<StageNode<S,SE,SO, H>>;
    using StageNodeSPtrC = std::shared_ptr<const StageNode<S,SE,SO, H>>;
    using StageNodeWPtr = std::weak_ptr<StageNode<S,SE,SO, H>>;

    Mcts(const MctsParameters& mcts_parameters) : root_(),
                                                  num_iterations_(0),
                                                  mcts_parameters_(mcts_parameters), 
                                                  heuristic_(mcts_parameters)
                                                  {}

    ~Mcts() {}
    
    template< class Q = S>
    typename std::enable_if<std::is_base_of<RequiresHypothesis, Q>::value>::type
    search(const S& current_state, HypothesisBeliefTracker& belief_tracker);

    void search(const S& current_state);

    unsigned int numIterations();
    unsigned int searchTime();
    std::string nodeInfo();
    ActionIdx returnBestAction();
    void printTreeToDotFile(std::string filename="tree");

    H& get_heuristic_function() { return heuristic_;};
    const mcts::StageNode<S, SE, SO, H>& get_root() const { return *root_;}

    template<class StateTransitionInfo> 
    std::vector<MctsEdgeInfo<StateTransitionInfo>> visit_mcts_tree_edges(
        const std::function<StateTransitionInfo(const S& start_state, const S& end_state, const AgentIdx& agent_idx)>& edge_info_extractor,
                    unsigned int max_depth = 100);

private:

    void iterate(const StageNodeSPtr& root_node);

    StageNodeSPtr root_;

    unsigned int num_iterations_;

    unsigned int search_time_;

    const MctsParameters mcts_parameters_;
    MctsParameters iteration_parameters_;

    H heuristic_;

    std::string sprintf(const StageNodeSPtr& root_node) const;

    template<class StateTransitionInfo> 
    void visit_stage_node_edges(const StageNodeSPtr& root_node,
        const std::function<StateTransitionInfo(const S& start_state, const S& end_state, const AgentIdx& agent_idx)>& edge_info_extractor,
        std::vector<MctsEdgeInfo<StateTransitionInfo>>& edge_infos, unsigned int max_depth);

    MCTS_TEST
};

template<class S, class SE, class SO, class H>
template<class Q>
typename std::enable_if<std::is_base_of<RequiresHypothesis, Q>::value>::type
Mcts<S, SE, SO, H>::search(const S& current_state, HypothesisBeliefTracker& belief_tracker) {
    auto start = std::chrono::high_resolution_clock::now();
    StageNode<S,SE, SO, H>::reset_counter();

    const auto max_iterations = mcts_parameters_.MAX_NUMBER_OF_ITERATIONS;
    const auto max_search_time_ms = mcts_parameters_.MAX_SEARCH_TIME;

    iteration_parameters_ = mcts_parameters_;
    root_ = std::make_shared<StageNode<S,SE, SO, H>,StageNodeSPtr, std::shared_ptr<S>, const JointAction&,
            const unsigned int&> (nullptr, current_state.clone(),JointAction(),0,  iteration_parameters_);
    num_iterations_ = 0;
    while (std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::high_resolution_clock::now() - start ).count() < max_search_time_ms && num_iterations_<max_iterations) {
        NodeStatistic<SE>::update_statistic_parameters(iteration_parameters_, root_->get_ego_int_node(), num_iterations_);
        belief_tracker.sample_current_hypothesis();
        iterate(root_);
        num_iterations_ += 1;
    }
    search_time_ = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::high_resolution_clock::now() - start ).count();
}

template<class S, class SE, class SO, class H>
void Mcts<S,SE,SO,H>::search(const S& current_state)
{
    auto start = std::chrono::high_resolution_clock::now();

    StageNode<S,SE, SO, H>::reset_counter();

    const auto max_iterations = mcts_parameters_.MAX_NUMBER_OF_ITERATIONS;
    const auto max_search_time_ms = mcts_parameters_.MAX_SEARCH_TIME;

    iteration_parameters_ = mcts_parameters_;
    root_ = std::make_shared<StageNode<S,SE, SO, H>,StageNodeSPtr, std::shared_ptr<S>, const JointAction&,
            const unsigned int&> (nullptr, current_state.clone(), JointAction(),0, iteration_parameters_);
    num_iterations_ = 0;
    while (std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::high_resolution_clock::now() - start ).count() < max_search_time_ms && num_iterations_<max_iterations) {
        NodeStatistic<SE>::update_statistic_parameters(iteration_parameters_, root_->get_ego_int_node(), num_iterations_);
        iterate(root_);
        num_iterations_ += 1;
    }
    search_time_ = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::high_resolution_clock::now() - start ).count();
}

template<class S, class SE, class SO, class H>
void Mcts<S,SE,SO,H>::iterate(const StageNodeSPtr& root_node)
{
    StageNodeSPtr node = root_node;
    StageNodeSPtr node_p;

    // --------------Select & Expand  -----------------
    // We descend the tree for all joint actions already available -> last node is the newly expanded one
    std::pair<bool, bool> traversing_result(true, true);
    while(traversing_result.first) {
        traversing_result = node->select_or_expand(node);
    }

    // -------------- Heuristic Update ----------------
    // Heuristic until terminal node only if state not terminal or max depth not reached
    if(traversing_result.second) {
      const auto& heuristics = heuristic_.calculate_heuristic_values(node);
      node->update_statistics(heuristics.first, heuristics.second);
    }

    // --------------- Backpropagation ----------------
    // Backpropagate, starting from parent node of newly expanded node
    node_p = node->get_parent().lock();
    while(true)
    {
        node_p->update_statistics(node);
        if(node_p->is_root())
        {
            break;
        }
        else
        {
            node = node->get_parent().lock();
            node_p = node_p->get_parent().lock();
        }
    }

  //  std::cout << root_node->sprintf() << std::endl;

}


template<class S, class SE, class SO, class H>
std::string Mcts<S,SE,SO,H>::sprintf(const StageNodeSPtr& root_node) const
{
    std::stringstream ss;
    return ss.str();
}

template<class S, class SE, class SO, class H>
unsigned int Mcts<S,SE,SO,H>::numIterations(){
    return this->num_iterations_;
}

template<class S, class SE, class SO, class H>
unsigned int Mcts<S,SE,SO,H>::searchTime(){
    return this->search_time_;
}

template<class S, class SE, class SO, class H>
std::string Mcts<S,SE,SO,H>::nodeInfo(){
    return sprintf(root_);
}

template<class S, class SE, class SO, class H>
ActionIdx Mcts<S,SE,SO,H>::returnBestAction(){
    ActionIdx idx_max = root_->get_best_action();
    return idx_max; 
}

template<class S, class SE, class SO, class H>
void Mcts<S,SE,SO,H>::printTreeToDotFile(std::string filename){ 
    root_->printTree(filename);
}

template<class S, class SE, class SO, class H>
template<class StateTransitionInfo> 
std::vector<MctsEdgeInfo<StateTransitionInfo>> Mcts<S,SE,SO,H>::visit_mcts_tree_edges(
        const std::function<StateTransitionInfo(const S& start_state, const S& end_state, const AgentIdx& agent_idx)>& edge_info_extractor, unsigned int max_depth) {
    std::vector<MctsEdgeInfo<StateTransitionInfo>> edge_infos;
    visit_stage_node_edges(root_, edge_info_extractor, edge_infos, max_depth);
    return edge_infos;
}

template<class S, class SE, class SO, class H>
template<class StateTransitionInfo> 
void Mcts<S,SE,SO,H>::visit_stage_node_edges(const StageNodeSPtr& root_node,
        const std::function<StateTransitionInfo(const S& start_state, const S& end_state, const AgentIdx& agent_idx)>& edge_info_extractor,
            std::vector<MctsEdgeInfo<StateTransitionInfo>>& edge_infos, unsigned int max_depth) {
    if(root_node->get_children().empty()) {
        return;
    }
    const auto& depth = root_node->get_depth();
    if(depth > max_depth) {
        return;
    }
    const auto& ego_policy = root_node->get_ego_int_node().get_policy();
    std::unordered_map<AgentIdx, Policy> other_policies;
    for (const auto& other_int_node : root_node->get_other_int_nodes()) {
        other_policies[other_int_node.get_agent_idx()] = other_int_node.get_policy();
    }
    for (auto& child_node_pair : root_node->get_children()) {
        const auto& ego_agent_id = root_node->get_state()->get_ego_agent_idx();
        const auto& ego_action_id = child_node_pair.first.at(S::ego_agent_idx);
        const auto& ego_action_weight = ego_policy.at(ego_action_id);
        const auto& ego_edge_info = edge_info_extractor(*root_node->get_state(), *child_node_pair.second->get_state(), ego_agent_id);
        edge_infos.push_back(std::make_tuple(ego_agent_id, depth, ego_action_id, ego_action_weight, ego_edge_info));

        for (auto action_idx = 1; action_idx < root_node->get_other_int_nodes().size() + 1; ++action_idx) {
            const auto& other_agent_id = 
                        root_node->get_other_int_nodes().at(action_idx - 1).get_agent_idx();
            const auto& other_action_id = child_node_pair.first.at(action_idx);
            const auto& other_action_weight = other_policies.at(other_agent_id).at(other_action_id);
            const auto& other_edge_info = edge_info_extractor(*root_node->get_state(), *child_node_pair.second->get_state(), other_agent_id);
            edge_infos.push_back(std::make_tuple(other_agent_id, depth, other_action_id, other_action_weight, other_edge_info));
        }
        visit_stage_node_edges(child_node_pair.second, edge_info_extractor, edge_infos, max_depth);
    }

}


} // namespace mcts


#endif