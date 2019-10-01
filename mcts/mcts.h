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
#include <string>
 

namespace mcts {

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
    using StageNodeWPtr = std::weak_ptr<StageNode<S,SE,SO, H>>;

    Mcts() : root_(), num_iterations(0), heuristic_() {};

    ~Mcts() {}
    
    template< class Q = S>
    typename std::enable_if<std::is_base_of<RequiresHypothesis, Q>::value>::type
    search(const S& current_state, HypothesisBeliefTracker<S>& belief_tracker, unsigned int max_search_time_ms, unsigned int max_iterations);

    void search(const S& current_state, unsigned int max_search_time_ms, unsigned int max_iterations);
    
    int numIterations();
    std::string nodeInfo();
    ActionIdx returnBestAction();
    void printTreeToDotFile(std::string filename="tree");

    void set_heuristic_function(const H& heuristic) {heuristic_ = heuristic;}

private:

    void iterate(const StageNodeSPtr& root_node);

    StageNodeSPtr root_;

    unsigned int num_iterations;

    H heuristic_;

    std::string sprintf(const StageNodeSPtr& root_node) const;

    MCTS_TEST
};

template<class S, class SE, class SO, class H>
template<class Q>
typename std::enable_if<std::is_base_of<RequiresHypothesis, Q>::value>::type
Mcts<S, SE, SO, H>::search(const S& current_state, HypothesisBeliefTracker<S>& belief_tracker,
                                     unsigned int max_search_time_ms, unsigned int max_iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    StageNode<S,SE, SO, H>::reset_counter();

    root_ = std::make_shared<StageNode<S,SE, SO, H>,StageNodeSPtr, std::shared_ptr<S>, const JointAction&,
            const unsigned int&> (nullptr, current_state.clone(),JointAction(),0);
    num_iterations = 0;
    while (std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::high_resolution_clock::now() - start ).count() < max_search_time_ms && num_iterations<max_iterations) {
        belief_tracker.sample_current_hypothesis();
        iterate(root_);
        num_iterations += 1;
    }
}

template<class S, class SE, class SO, class H>
void Mcts<S,SE,SO,H>::search(const S& current_state, unsigned int max_search_time_ms, unsigned int max_iterations)
{
    auto start = std::chrono::high_resolution_clock::now();

    StageNode<S,SE, SO, H>::reset_counter();

    root_ = std::make_shared<StageNode<S,SE, SO, H>,StageNodeSPtr, std::shared_ptr<S>, const JointAction&,
            const unsigned int&> (nullptr, current_state.clone(), JointAction(),0);
    num_iterations = 0;
    while (std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::high_resolution_clock::now() - start ).count() < max_search_time_ms && num_iterations<max_iterations) {
        iterate(root_);
        num_iterations += 1;
    }
}

template<class S, class SE, class SO, class H>
void Mcts<S,SE,SO,H>::iterate(const StageNodeSPtr& root_node)
{
    StageNodeSPtr node = root_node;
    StageNodeSPtr node_p;

    // --------------Select & Expand  -----------------
    // We descend the tree for all joint actions already available -> last node is the newly expanded one
    while(node->select_or_expand(node));

    // -------------- Heuristic Update ----------------
    // Heuristic until terminal node
    const auto& heuristics = heuristic_.calculate_heuristic_values(node);
    node->update_statistics(heuristics.first, heuristics.second);
    
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

#ifdef PLAN_DEBUG_INFO
    sprintf(root_node);
#endif
}

template<class S, class SE, class SO, class H>
std::string Mcts<S,SE,SO,H>::sprintf(const StageNodeSPtr& root_node) const
{
    std::stringstream ss;
    return ss.str();
}

template<class S, class SE, class SO, class H>
int Mcts<S,SE,SO,H>::numIterations(){
    return this->num_iterations;
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


} // namespace mcts


#endif