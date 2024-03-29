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
#include <thread>


namespace mcts {

template<class StateTransitionInfo> 
using MctsEdgeInfo = std::tuple<AgentIdx, unsigned int, ActionIdx, ActionWeight, StateTransitionInfo>;
template<class StateInfo> 
// node depth, node visit count and state type specific info
using MctsStateInfo = std::tuple<unsigned int, unsigned int, StateInfo>;

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
                                                  heuristic_(mcts_parameters_),
                                                  parallel_mcts_()
                                                  {}

    ~Mcts() {}
    
    template< class Q = S>
    typename std::enable_if<std::is_base_of<RequiresHypothesis, Q>::value>::type
    search(const S& current_state, HypothesisBeliefTracker& belief_tracker);

    void search(const S& current_state);

    unsigned int numIterations();
    unsigned int numNodes();
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

    template<class StateInfo> 
    std::vector<MctsStateInfo<StateInfo>> visit_mcts_tree_nodes(
        const std::function<StateInfo(const S& state)>& state_info_extractor,
                    unsigned int max_depth = 100, bool visit_terminal_states = false);

protected:

    void iterate(const StageNodeSPtr& root_node);

    template< class Q = S>
    typename std::enable_if<std::is_base_of<RequiresHypothesis, Q>::value>::type
    single_search(const S& current_state, HypothesisBeliefTracker& belief_tracker);
    void single_search(const S& current_state);

    template< class Q = S>
    typename std::enable_if<std::is_base_of<RequiresHypothesis, Q>::value>::type
    parallel_search(const S& current_state, HypothesisBeliefTracker& belief_tracker);
    void parallel_search(const S& current_state);

    StageNodeSPtr root_;

    unsigned int num_iterations_;

    unsigned int search_time_;

    const MctsParameters mcts_parameters_;
    MctsParameters iteration_parameters_;

    H heuristic_;

    std::vector<Mcts<S, SE, SO, H>> parallel_mcts_;

    std::string sprintf(const StageNodeSPtr& root_node) const;

    template<class StateTransitionInfo> 
    void visit_stage_node_edges(const StageNodeSPtr& root_node,
        const std::function<StateTransitionInfo(const S& start_state, const S& end_state, const AgentIdx& agent_idx)>& edge_info_extractor,
        std::vector<MctsEdgeInfo<StateTransitionInfo>>& edge_infos, unsigned int max_depth);

    template<class StateInfo> 
    void visit_stage_nodes(const StageNodeSPtr& root_node,
        const std::function<StateInfo(const S& state)>& state_info_extractor,
        std::vector<MctsStateInfo<StateInfo>>& state_infos, unsigned int max_depth, bool visit_terminal_states);

    StageNodeSPtr merge_searched_trees(const std::vector<Mcts<S, SE, SO, H>>& searched_trees);

    MCTS_TEST
};

template<class S, class SE, class SO, class H>
template<class Q>
typename std::enable_if<std::is_base_of<RequiresHypothesis, Q>::value>::type
Mcts<S, SE, SO, H>::search(const S& current_state, HypothesisBeliefTracker& belief_tracker) {
    if(mcts_parameters_.NUM_PARALLEL_MCTS > 1) {
        this->parallel_search(current_state, belief_tracker);
    } else {
        this->single_search(current_state, belief_tracker);
    }
}

template<class S, class SE, class SO, class H>
void Mcts<S,SE,SO,H>::search(const S& current_state)
{
    if(mcts_parameters_.NUM_PARALLEL_MCTS > 1) {
        this->parallel_search(current_state);
    } else {
        this->single_search(current_state);
    }
}

template<class S, class SE, class SO, class H>
template<class Q>
typename std::enable_if<std::is_base_of<RequiresHypothesis, Q>::value>::type
Mcts<S, SE, SO, H>::single_search(const S& current_state, HypothesisBeliefTracker& belief_tracker) {
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
void Mcts<S,SE,SO,H>::single_search(const S& current_state)
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
void Mcts<S, SE, SO, H>::parallel_search(const S& current_state) {
    auto start = std::chrono::high_resolution_clock::now();
    parallel_mcts_.clear();

    for(unsigned i = 0; i < this->mcts_parameters_.NUM_PARALLEL_MCTS; ++i) {
        auto mcts_parameters_parallel_mcts = this->mcts_parameters_;
        parallel_mcts_.push_back(Mcts<S, SE, SO, H>(this->mcts_parameters_));
    }

    if (this->mcts_parameters_.USE_MULTI_THREADING) {
        std::vector<std::thread> threads;
        for(unsigned i = 0; i < this->mcts_parameters_.NUM_PARALLEL_MCTS; ++i) {
            const auto& cloned_state = current_state.clone();
            cloned_state->choose_random_seed(i);
            threads.push_back(std::thread([](Mcts<S, SE, SO, H>& mcts, const S& state){ 
                mcts.single_search(state);
            }, std::ref(parallel_mcts_.at(i)), *cloned_state));
        }
        bool all_joined = false;
        for(unsigned i = 0; i < this->mcts_parameters_.NUM_PARALLEL_MCTS; ++i) {
            threads.at(i).join();
        }
    } else  { 
        for (unsigned i = 0; i < this->mcts_parameters_.NUM_PARALLEL_MCTS; ++i) {
            const auto& cloned_state = current_state.clone();
            cloned_state->choose_random_seed(i);
            parallel_mcts_[i].single_search(*cloned_state);
        }
    }
    
    this->root_ = merge_searched_trees(parallel_mcts_);
    search_time_ = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::high_resolution_clock::now() - start ).count();
}

template<class S, class SE, class SO, class H>
template<class Q>
typename std::enable_if<std::is_base_of<RequiresHypothesis, Q>::value>::type
Mcts<S, SE, SO, H>::parallel_search(const S& current_state, HypothesisBeliefTracker& belief_tracker) {
    auto start = std::chrono::high_resolution_clock::now();
    
    parallel_mcts_.clear();

    for(unsigned i = 0; i < this->mcts_parameters_.NUM_PARALLEL_MCTS; ++i) {
        parallel_mcts_.push_back(Mcts<S, SE, SO, H>(this->mcts_parameters_));
    }
    unsigned num_iterations = 0;
    unsigned num_nodes = 0;
    if (this->mcts_parameters_.USE_MULTI_THREADING) {
        std::vector<std::thread> threads;
        for(unsigned i = 0; i < this->mcts_parameters_.NUM_PARALLEL_MCTS; ++i) {
            threads.push_back(std::thread([](Mcts<S, SE, SO, H>& mcts, const S& state, 
                const HypothesisBeliefTracker& belief_tracker, const unsigned& mcts_idx){ 
                HypothesisBeliefTracker local_belief_tracker = belief_tracker;
                const auto& cloned_state = 
                    state.change_belief_reference(local_belief_tracker.sample_current_hypothesis());
                cloned_state->choose_random_seed(mcts_idx);
                mcts.single_search(*cloned_state, local_belief_tracker);
            }, std::ref(parallel_mcts_.at(i)), current_state, belief_tracker, i));
        }
        bool all_joined = false;
        for(unsigned i = 0; i < this->mcts_parameters_.NUM_PARALLEL_MCTS; ++i) {
            threads.at(i).join();
        }
    } else  { 
        for (unsigned i = 0; i < this->mcts_parameters_.NUM_PARALLEL_MCTS; ++i) {
            HypothesisBeliefTracker local_belief_tracker = belief_tracker;
            const auto& cloned_state = 
                current_state.change_belief_reference(local_belief_tracker.sample_current_hypothesis());
            cloned_state->choose_random_seed(i);
            parallel_mcts_[i].single_search(*cloned_state);
            num_iterations += parallel_mcts_[i].numIterations();
            num_nodes += parallel_mcts_[i].numNodes();
        }
    }

    this->root_ = merge_searched_trees(parallel_mcts_);
    num_iterations_ = num_iterations;
    this->root_->set_num_nodes(num_nodes);
    search_time_ = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::high_resolution_clock::now() - start ).count();
}

template<class S, class SE, class SO, class H>
std::shared_ptr<StageNode<S, SE, SO, H>> Mcts<S, SE, SO, H>::merge_searched_trees(
                                            const std::vector<Mcts<S, SE, SO, H>>& searched_trees) {

    iteration_parameters_ = NodeStatistic<SE>::merge_mcts_parameters([&](){
        std::vector<MctsParameters> parameters;
        for (const auto mcts : searched_trees) {
            parameters.push_back(mcts.iteration_parameters_);
        }
        return parameters;
    }());

    auto root = std::make_shared<StageNode<S,SE, SO, H>, StageNodeSPtr, std::shared_ptr<S>, const JointAction&,
            const unsigned int&> (nullptr, searched_trees.begin()->get_root().get_state()->clone(), JointAction(), 0, iteration_parameters_);
    root->merge_node_statistics([&]() {
        std::vector<StageNode<S,SE,SO, H>> root_nodes;
        for(const auto& tree : searched_trees) {
            root_nodes.push_back(tree.get_root());
        }
        return root_nodes;
    }());
    return root;
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
    while(true && bool(node_p))
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
unsigned int Mcts<S,SE,SO,H>::numNodes(){
    return this->root_->get_num_nodes();
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
        const std::function<StateTransitionInfo(const S& start_state, const S& end_state,
                 const AgentIdx& agent_idx)>& edge_info_extractor, unsigned int max_depth) {
    std::vector<MctsEdgeInfo<StateTransitionInfo>> edge_infos;
    visit_stage_node_edges(root_, edge_info_extractor, edge_infos, max_depth);
    return edge_infos;
}

template<class S, class SE, class SO, class H>
template<class StateInfo> 
std::vector<MctsStateInfo<StateInfo>> Mcts<S,SE,SO,H>::visit_mcts_tree_nodes(
    const std::function<StateInfo(const S& state)>& state_info_extractor,
                    unsigned int max_depth, bool visits_terminal_states) {
    std::vector<MctsStateInfo<StateInfo>> state_infos;
    visit_stage_nodes(root_, state_info_extractor, state_infos, max_depth, visits_terminal_states);
    return state_infos;
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

template<class S, class SE, class SO, class H>
template<class StateInfo> 
void Mcts<S,SE,SO,H>::visit_stage_nodes(const StageNodeSPtr& root_node,
    const std::function<StateInfo(const S& state)>& state_info_extractor,
    std::vector<MctsStateInfo<StateInfo>>& state_infos, unsigned int max_depth, bool visit_terminal_states) {
    if (root_node->get_state()->is_terminal() && !visit_terminal_states) {
        return;
    }
    const auto depth = root_node->get_depth();
    const auto visit_count = root_node->get_visit_count();
    const auto& state_info = state_info_extractor(*root_node->get_state());
    state_infos.push_back(std::make_tuple(depth, visit_count, state_info));
    if(root_node->get_children().empty()) {
        return;
    }
    if(depth > max_depth) {
        return;
    }

    for (auto& child_node_pair : root_node->get_children()) {
        visit_stage_nodes(child_node_pair.second, state_info_extractor, state_infos, max_depth, visit_terminal_states);
    }
}


} // namespace mcts


#endif