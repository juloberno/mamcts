// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#include "mcts/state.h"

namespace mcts {

typedef unsigned int HypothesisId;

class Hypothesis {
    Hypothesis(const HypothesisId& id) : hypothesis_id_(id) {}
    HypothesisId getId() const {return hypothesis_id_;}

    template <class S>
    ActionIdx get_action(const StateInterface<S>& state);

private:
    const HypothesisId hypothesis_id_;

}

typedef std::weak_ptr<Hypothesis> HypothesisWPtr;
typedef std::shared_ptr<Hypothesis> HypothesisSPtr;

} // namespace mcts