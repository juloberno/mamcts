// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#include "mcts/mcts.h"
#include "environments/crossing_state_episode_runner.h"

template <>
const std::vector<std::string> mcts::CrossingStateEpisodeRunner<int>::EVAL_RESULT_COLUMN_DESC = {"Reward", "Cost", "Terminal", "Collision", "GoalReached", "MaxSteps", "NumSteps"};

template <>
const std::vector<std::string> mcts::CrossingStateEpisodeRunner<float>::EVAL_RESULT_COLUMN_DESC = mcts::CrossingStateEpisodeRunner<int>::EVAL_RESULT_COLUMN_DESC;