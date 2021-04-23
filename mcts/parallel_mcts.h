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
class ParallelMcts : public Mcts<S, SE, SO, H> {
    using StageNodeSPtr = std::shared_ptr<StageNode<S,SE,SO, H>>;
    using StageNodeSPtrC = std::shared_ptr<const StageNode<S,SE,SO, H>>;
    using StageNodeWPtr = std::weak_ptr<StageNode<S,SE,SO, H>>;

public:
    ParallelMcts(const MctsParameters& mcts_parameters) :
                 Mcts<S, SE, SO, H>(mcts_parameters) {}

    ~ParallelMcts() {}

    void search(const S& current_state);

    template< class Q = S>
    typename std::enable_if<std::is_base_of<RequiresHypothesis, Q>::value>::type
    search(const S& current_state, HypothesisBeliefTracker& belief_tracker);

private:
    StageNodeSPtr merge_searched_trees(const std::vector<Mcts<S, SE, SO, H>>& searched_trees) const;
    std::vector<Mcts<S, SE, SO, H>> parallel_mcts_;
    MCTS_TEST
};





} // namespace mcts


#endif