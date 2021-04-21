// Copyright (c) 2021 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef PARALLEL_MCTS_H
#define PARALLEL_MCTS_H


#include "mcts.h"


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
class ParallelMcts {
    using StageNodeSPtr = std::shared_ptr<StageNode<S,SE,SO, H>>;
    using StageNodeSPtrC = std::shared_ptr<const StageNode<S,SE,SO, H>>;
    using StageNodeWPtr = std::weak_ptr<StageNode<S,SE,SO, H>>;

public:
    ParallelMcts(const MctsParameters& mcts_parameters) : root_(),
                                                  num_iterations_(0),
                                                  mcts_parameters_(mcts_parameters), 
                                                  heuristic_(mcts_parameters)
                                                  {}

    ~ParallelMcts() {}
    

    void search(const S& current_state);

private:
    StageNodeSPtr merge_searched_trees(const std::vector<Mcts<S, SE, SO, H>>& searched_trees) const;
    std::vector<Mcts<S, SE, SO, H>> parallel_mcts_;
    MCTS_TEST
};


template<class S, class SE, class SO, class H>
void ParallelMcts<S, SE, SO, H>::search_parallel_root(const S& current_state, const unsigned& num_parallel_mcts) {
    std::vector<std::thread> threads;
    parallel_mcts_.clear();

    for(unsigned i = 0; i < num_parallel_mcts; ++i) {
        parallel_mcts_.push_back(Mcts<S, SE, SO, H>(mcts_parameters_))
    }

    for(unsigned i = 0; i < num_parallel_mcts; ++i) {
        threads_.push_back(std::thread(&Mcts::search, &parallel_mcts_.at(i), current_state));
    }
    bool all_joined = false;
    for(unsigned i = 0; i < num_parallel_mcts; ++i) {
        threads_.at(i).join();
    }

    root_ = merge_parallel_trees(parallel_mcts_);
}

template<class S, class SE, class SO, class H>
std::shared_ptr<StageNode<S,SE,SO, H>> ParallelMcts<S, SE, SO, H>::merge_searched_trees(
                                            const std::vector<Mcts<S, SE, SO, H>>& searched_trees) const {

    root_ = std::make_shared<StageNode<S,SE, SO, H>,StageNodeSPtr, std::shared_ptr<S>, const JointAction&,
            const unsigned int&> (nullptr, current_state.clone(),JointAction(),0,  iteration_parameters_);
    root_->merge_node_statistics([&]() {
        std::vector<StageNode<S,SE,SO, H>> root_nodes;
        for(const& auto tree : searched_trees) {
            root_nodes.push_back(tree.get_root());
        }
        return root_nodes;
    });
}


} // namespace mcts


#endif