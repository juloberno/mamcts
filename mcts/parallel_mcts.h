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


template<class S, class SE, class SO, class H>
void ParallelMcts<S, SE, SO, H>::search(const S& current_state) {
    std::vector<std::thread> threads;
    parallel_mcts_.clear();

    for(unsigned i = 0; i < this->mcts_parameters_.NUM_PARALLEL_MCTS; ++i) {
        auto mcts_parameters_parallel_mcts = this->mcts_parameters_;
       // mcts_parameters_parallel_mcts.RANDOM_SEED = this->mcts_parameters_.RANDOM_SEED*(i+1);
        parallel_mcts_.push_back(Mcts<S, SE, SO, H>(mcts_parameters_parallel_mcts));
    }

    for(unsigned i = 0; i < this->mcts_parameters_.NUM_PARALLEL_MCTS; ++i) {
        const auto& cloned_state = current_state.clone();
        cloned_state->choose_random_seed(i);
        threads.push_back(std::thread([](Mcts<S, SE, SO, H>& mcts, const S& state){ 
            mcts.search(state);
        }, std::ref(parallel_mcts_.at(i)), *cloned_state));
    }
    bool all_joined = false;
    for(unsigned i = 0; i < this->mcts_parameters_.NUM_PARALLEL_MCTS; ++i) {
        threads.at(i).join();
    }

    this->root_ = merge_searched_trees(parallel_mcts_);
}

template<class S, class SE, class SO, class H>
template<class Q>
typename std::enable_if<std::is_base_of<RequiresHypothesis, Q>::value>::type
ParallelMcts<S, SE, SO, H>::search(const S& current_state, HypothesisBeliefTracker& belief_tracker) {
    std::vector<std::thread> threads;
    parallel_mcts_.clear();

    for(unsigned i = 0; i < this->mcts_parameters_.NUM_PARALLEL_MCTS; ++i) {
        auto mcts_parameters_parallel_mcts = this->mcts_parameters_;
        //mcts_parameters_parallel_mcts.RANDOM_SEED = this->mcts_parameters_.RANDOM_SEED*(i+1);
        parallel_mcts_.push_back(Mcts<S, SE, SO, H>(mcts_parameters_parallel_mcts));
    }

    for(unsigned i = 0; i < this->mcts_parameters_.NUM_PARALLEL_MCTS; ++i) {
        const auto& cloned_state = current_state.clone();
        cloned_state->choose_random_seed(i);
        threads.push_back(std::thread([](Mcts<S, SE, SO, H>& mcts, const S& state){ 
            mcts.search(state);
        }, std::ref(parallel_mcts_.at(i)), *cloned_state));
    }
    bool all_joined = false;
    for(unsigned i = 0; i < this->mcts_parameters_.NUM_PARALLEL_MCTS; ++i) {
        threads.at(i).join();
    }

    this->root_ = merge_searched_trees(parallel_mcts_);
}

template<class S, class SE, class SO, class H>
std::shared_ptr<StageNode<S, SE, SO, H>> ParallelMcts<S, SE, SO, H>::merge_searched_trees(
                                            const std::vector<Mcts<S, SE, SO, H>>& searched_trees) const {

    auto root = std::make_shared<StageNode<S,SE, SO, H>, StageNodeSPtr, std::shared_ptr<S>, const JointAction&,
            const unsigned int&> (nullptr, nullptr, JointAction(), 0, this->mcts_parameters_);
    root->merge_node_statistics([&]() {
        std::vector<StageNode<S,SE,SO, H>> root_nodes;
        for(const auto& tree : searched_trees) {
            root_nodes.push_back(tree.get_root());
        }
        return root_nodes;
    }());
    return root;
}


} // namespace mcts


#endif