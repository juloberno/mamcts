// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef MCTSTEST_H
#define MCTSTEST_H

#include "mcts/mcts.h"
#include "mcts/heuristics/random_heuristic.h"
#include "mcts/statistics/uct_statistic.h"
#include "test/simple_state.h"

using namespace mcts;
using namespace std;


class mcts::MctsTest
{

public:
    template< class S, class SE, class SO, class H>
    void verify_uct(const Mcts<S, SE, SO, H>& mcts, unsigned int depth) {
        std::vector<UctStatistic> expected_root_statistics = verify_uct(mcts.root_, depth);
    }

    template< class S, class H>
    std::vector<UctStatistic> verify_uct(const StageNodeSPtr<S,UctStatistic,UctStatistic,H>& start_node, unsigned int depth)
    {
            if(start_node->children_.empty())
            {
               return std::vector<UctStatistic>();
            }


            const AgentIdx num_agents = start_node->state_->get_agent_idx().size();


            std::vector<UctStatistic> expected_statistics(num_agents, UctStatistic(0));

            // ----- RECURSIVE ESTIMATION OF QVALUES AND COUNTS downwards tree -----------------------
            for(auto it = start_node->children_.begin(); it != start_node->children_.end(); ++it)
            {
                std::vector<UctStatistic> expected_child_statistics = verify_uct(it->second,depth);

                // check joint actions are different
                auto it_other_child = it;
                for (std::advance(it_other_child,1); it_other_child != start_node->children_.end(); ++it_other_child)
                {
                    std::stringstream ss;
                    ss << "Equal joint action child-ids: "<<  it->second->id_ << " and "
                    << it_other_child->second->id_  << ", keys: " << static_cast<JointAction>(it->first) << " and " << static_cast<JointAction>(it_other_child->first);

                    EXPECT_TRUE( it->second->joint_action_ != it_other_child->second->joint_action_) << ss.str();
                }

                auto& child = it->second;
                std::vector<Reward> rewards;
                auto& joint_action = child->joint_action_;
                auto new_state =  start_node->state_->execute(joint_action, rewards);


                // ---------------------- Expected total node visits --------------------------
                bool is_first_child_and_not_parent_root = (it == start_node->children_.begin()) && (!start_node->is_root());
                expected_total_node_visits(it->second->ego_int_node_, S::ego_agent_idx, is_first_child_and_not_parent_root, expected_statistics);
                expected_action_count(it->second->ego_int_node_, S::ego_agent_idx, joint_action, is_first_child_and_not_parent_root, expected_statistics);
                for (auto it = child->other_int_nodes_.begin(); it != child->other_int_nodes_.end(); ++it  )
                {
                    expected_total_node_visits(*it, it->get_agent_idx(), is_first_child_and_not_parent_root, expected_statistics );
                    expected_action_count(*it, it->get_agent_idx(),joint_action, is_first_child_and_not_parent_root, expected_statistics );
                }
            }

            for(auto agent_idx = 0; agent_idx < num_agents; ++agent_idx) {
                expected_statistics[agent_idx].update_value();
            }

            // --------- COMPARE RECURSIVE ESTIMATION AGAINST EXISTING BACKPROPAGATION VALUES  ------------
            compare_expected_existing(expected_statistics,start_node->ego_int_node_,start_node->id_,start_node->depth_);

            for (auto it = start_node->other_int_nodes_.begin(); it != start_node->other_int_nodes_.end(); ++it  )
            {
                compare_expected_existing(expected_statistics,*it,start_node->id_,start_node->depth_);
            }

            return expected_statistics;

    }
private:
    // update expected ucb_stat, this functions gets called once for each agent for each child (= number agents x number childs)
    void expected_total_node_visits(const UctStatistic& child_stat, const AgentIdx& agent_idx, bool is_first_child_and_not_parent_root, std::vector<UctStatistic>& expected_statistics) {


        expected_statistics[agent_idx].total_node_visits_ += child_stat.total_node_visits_; // total count for childs + 1 (first expansion of child_stat)
        if(is_first_child_and_not_parent_root) {
            expected_statistics[agent_idx].total_node_visits_ += 1;
        }
    }

    void expected_action_count(const UctStatistic& child_stat, const AgentIdx& agent_idx, const JointAction& joint_action, bool is_first_child_and_not_parent_root, std::vector<UctStatistic>& expected_statistics) {
        // add up all child node counts which may belong to different actions of the other agents
        expected_statistics[agent_idx].ucb_statistics_[joint_action[agent_idx]].action_count_ +=  child_stat.total_node_visits_;
    }
     /*
        void expected_total_node_visits(const std::vector<UctStatistic>& expected_child_statistics, const UctStatistic& child_stat,
            const UctStatistic& parent_stat,const AgentIdx& agent_idx, const JointAction& joint_action, const std::vector<Reward> rewards, std::vector<UctStatistic>& expected_statistics, bool add)
      if(expected_statistics[agent_idx].ucb_statistics_.find(joint_action[agent_idx]) != expected_statistics[agent_idx].ucb_statistics_.end())
        {
            // first "expansion of this action idx -> add the action count the total count of the child stat
            expected_statistics[agent_idx].ucb_statistics_[joint_action[agent_idx]].action_count_ += recursive_expected_total_count;
        }
        expected_statistics[agent_idx].ucb_statistics_[joint_action[agent_idx]].action_count_ += 1; // add one more for the first expansion of this node


        double recursive_expected_value = child_stat.value_;
        if (!expected_child_statistics.empty()) {
            recursive_expected_value = expected_child_statistics[agent_idx].value_;
        }
        expected_statistics[agent_idx].ucb_statistics_[joint_action[agent_idx]].action_value_ +=
                recursive_expected_value + parent_stat.k_discount_factor * rewards[agent_idx];
            */

    template<class S, class Stats>
    void compare_expected_existing(const std::vector<UctStatistic>& expected_statistics,  const IntermediateNode<S,Stats>& inter_node,
                                   unsigned id, unsigned depth) {
        // Compare node visits
        const AgentIdx agent_idx = inter_node.get_agent_idx();
        auto recursive_node_visit = expected_statistics[agent_idx].total_node_visits_;
        auto existing_node_visit = inter_node.total_node_visits_;
        EXPECT_EQ(existing_node_visit, recursive_node_visit) << "Unexpected recursive node visits for node " << id << " at depth " << depth << " for agent " << (int)agent_idx;

        auto recursively_expected_value = expected_statistics[agent_idx].value_;
        double existing_value = inter_node.value_;
        //EXPECT_EQ(existing_value, recursively_expected_value) << "Unexpected recursive value for node " << id << " at depth " << depth << " for agent " << (int)agent_idx;


        ASSERT_EQ((int)inter_node.state_.get_num_actions(agent_idx),inter_node.ucb_statistics_.size()) << "Internode state and statistic are of unequal length";
        for (auto action_it = inter_node.ucb_statistics_.begin(); action_it != inter_node.ucb_statistics_.end(); ++action_it)
        {   
            ActionIdx action_idx = action_it->first;
            
            if(expected_statistics[agent_idx].ucb_statistics_.find(action_idx) ==  expected_statistics[agent_idx].ucb_statistics_.end()) {
                continue; // UCBStatistic class initialized map for all available actions, but during search are only some of them expanded.
                        // Only the expanded actions are recursively estimated 
            }    
            //auto recursively_expected_qvalue = expected_statistics[agent_idx].ucb_statistics_.at(action_idx).action_value_;
            //map at error is here
            /*cout << "---------------------------------------------------" << endl;
            cout << "num actions "<< (int)inter_node.state_.get_num_actions(agent_idx)<< "statistics size "<<inter_node.ucb_statistics_.size() << endl;
            cout << "node " << id << " depth " << depth << " for agent " << (int)agent_idx <<  " and action " << (int)action_idx << endl;
            cout << "expected Q_Value: " << recursively_expected_qvalue << endl;
            cout << "existing Q_Value: " << inter_node.ucb_statistics_.at(action_idx).action_value_ << endl;

            double existing_qvalue = inter_node.ucb_statistics_.at(action_idx).action_value_;
            //EXPECT_EQ(existing_qvalue, recursively_expected_qvalue) << "Unexpected recursive q-value for node " << id << " at depth " << depth << " for agent " << (int)agent_idx <<  " and action " << (int)action_idx; */

            auto recursively_expected_count = expected_statistics[agent_idx].ucb_statistics_.at(action_idx).action_count_;
            unsigned existing_count = inter_node.ucb_statistics_.at(action_idx).action_count_; 

            EXPECT_EQ(existing_count, recursively_expected_count) << "Unexpected recursive action count for node " << id << " at depth " << depth << " for agent " << (int)agent_idx  <<  " and action " << (int)action_idx;

        }
    }
};
#endif