// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef MCTS_STAGE_NODE_H
#define MCTS_STAGE_NODE_H


#include "state.h"
#include "intermediate_node.h"
#include "node_statistic.h"
#include <memory>
#include <unordered_map>
#include <boost/functional/hash.hpp>
#include <iostream>
#include "common.h"
#include <fstream> 
#include "mcts_parameters.h"
#include <string>


namespace mcts {

// hash function to use JoinAction as std::unordered map key
template <typename Container> // we can make this generic for any container [1]
struct container_hash {
    std::size_t operator()(Container const& c) const {
        return boost::hash_range(c.begin(), c.end());
    }
};


    /*
        * S: State Model
        * SE: Statistics Ego Agent
        * SO: Statistics Other Agents, e.g. probabilistic opponent models
        * H: Heuristic Model
        */
    template<class S, class SE, class SO, class H>
    class StageNode : public std::enable_shared_from_this<StageNode<S,SE, SO, H>> {
    private:
        using StageNodeSPtr = std::shared_ptr<StageNode<S,SE,SO, H>>;
        using StageNodeWPtr = std::weak_ptr<StageNode<S,SE,SO, H>>;
        typedef std::unordered_map<JointAction,StageNodeSPtr,container_hash<JointAction>> StageChildMap;

        // Environment State
        std::shared_ptr<S> state_;

        // Parents and children
        StageNodeWPtr parent_;
        StageChildMap children_;

        // Intermediate decision nodes
        IntermediateNode<S, SE> ego_int_node_;
        typedef std::vector<IntermediateNode<S, SO>> InterNodeVector;
        InterNodeVector other_int_nodes_;

        const JointAction joint_action_; // action_idx leading to this node
        const unsigned int id_;
        const unsigned int depth_;
        const unsigned int max_num_joint_actions_;
        static unsigned int num_nodes_;

    public:
        StageNode(const StageNodeSPtr& parent, std::shared_ptr<S> state,  const JointAction& joint_action, const unsigned int& depth);
        ~StageNode();
        bool select_or_expand(StageNodeSPtr& next_node);
        void update_statistics(const std::vector<SE>& heuristic_estimates);
        void update_statistics(const StageNodeSPtr& changed_child_node);
        bool each_agents_actions_expanded();
        bool each_joint_action_expanded();
        StageNodeSPtr get_shared();
        const S* get_state() const {return state_.get();}
        StageNodeWPtr get_parent() {return parent_;}
        bool is_root() const {return !parent_.lock();}
        ActionIdx get_best_action();

        std::string sprintf() const;
        void printTree(std::string filename);
        void printLayer(std::string filename);
        double getEgoAgentValue();
        int getEgoNodeVisits();
        double getActionValue(int action);

        static void reset_counter();

        MCTS_TEST
    };


    template<class S, class SE, class SO, class H>
    using StageNodeSPtr = std::shared_ptr<StageNode<S,SE, SO, H>>;


    template<class S, class SE, class SO, class H>
    StageNode<S,SE, SO, H>::StageNode(const StageNodeSPtr& parent,
                                        std::shared_ptr<S> state, const JointAction& joint_action, const unsigned int& depth ) :
    state_(state),
    parent_(parent),
    children_(),
    ego_int_node_(*state_,S::ego_agent_idx,state_->get_num_actions(S::ego_agent_idx)),
    other_int_nodes_([this]()-> InterNodeVector {
        // Initialize the intermediate nodes of other agents
        InterNodeVector vec;
        // vec.resize(state_.get_agent_idx().size()-1);
        for (AgentIdx ai = S::ego_agent_idx+1; ai < state_->get_agent_idx().size(); ++ai ) {
            vec.emplace_back(*state_,ai,state_->get_num_actions(ai));
        }
        return vec;
    }()),
    joint_action_(joint_action),
    max_num_joint_actions_([this]()-> unsigned int{
        ActionIdx num_actions(state_->get_num_actions(S::ego_agent_idx));
        for(auto ai = S::ego_agent_idx+1; ai < state_->get_agent_idx().size(); ++ai ) {
            num_actions *=state_->get_num_actions(ai);
        }
        return num_actions; }() ),
    id_(++num_nodes_),
    depth_(depth)
    {
    }

    template<class S, class SE, class SO, class H>
    StageNode<S,SE, SO, H>::~StageNode() {
    }

    template<class S, class SE, class SO, class H>
    StageNodeSPtr<S,SE, SO, H> StageNode<S,SE, SO, H>::get_shared() {
        return this->shared_from_this();
    }

    template<class S, class SE, class SO, class H>
    bool StageNode<S,SE, SO, H>::select_or_expand(StageNodeSPtr& next_node) {
        // helper function to fill rewards
        auto fill_rewards = [this](const std::vector<Reward>& reward_list, const JointAction& ja) {
            ego_int_node_.collect_reward(reward_list[S::ego_agent_idx], ja[S::ego_agent_idx]);
            for (auto it = other_int_nodes_.begin(); it != other_int_nodes_.end(); ++it)
            {
                it->collect_reward(reward_list[it->get_agent_idx()],ja[it->get_agent_idx()] );
            }
        };

        // First check if state of node is terminal
        if(this->get_state()->is_terminal()) {
            next_node = get_shared();
            return false;
        }

        // Let each agent select an action according to its statistic model -> yields joint_action
        JointAction joint_action(state_->get_agent_idx().size());
        joint_action[ego_int_node_.get_agent_idx()] = ego_int_node_.choose_next_action();
        for(auto it = other_int_nodes_.begin(); it != other_int_nodes_.end(); ++it)
        {
            joint_action[it->get_agent_idx()] = it->choose_next_action();
        }

        // Check if joint action was already expanded
        auto it = children_.find(joint_action);
        if( it != children_.end())
        {
            // SELECT EXISTING NODE
            next_node = it->second;
            std::vector<Reward> rewards;
            //recalculate the rewards
            state_->execute(joint_action, rewards);
            // set selected action indexes, reward = 0
            //todo: no not overwrite here, only for distributional rewards, another execute may be needed 
            fill_rewards(rewards, joint_action);
            return true;
        }
        else
        {   // EXPAND NEW NODE BASED ON NEW JOINT ACTION
            std::vector<Reward> rewards;
            next_node = std::make_shared<StageNode<S,SE, SO, H>,StageNodeSPtr, std::shared_ptr<S>,
                    const JointAction&, const unsigned int&> (get_shared(), state_->execute(joint_action, rewards),joint_action,depth_+1);
            children_[joint_action] = next_node;
            #ifdef PLAN_DEBUG_INFO
            //     std::cout << "expanded node state: " << state_->execute(joint_action, rewards)->sprintf();
            #endif
            // collect intermediate rewards and selected action indexes
            fill_rewards(rewards, joint_action);

            return false;
        }

    }

    template<class S, class SE, class SO, class H>
    unsigned int StageNode<S,SE, SO, H>::num_nodes_ = 0;

    template<class S, class SE, class SO, class H>
    void StageNode<S,SE, SO, H>::reset_counter() {
        StageNode<S,SE, SO, H>::num_nodes_ = 0;
    }

    template<class S, class SE, class SO, class H>
    bool StageNode<S,SE, SO, H>::each_joint_action_expanded() {
        return children_.size() == max_num_joint_actions_;
    
    }

    template<class S, class SE, class SO, class H>
    bool StageNode<S,SE, SO, H>::each_agents_actions_expanded() {
        if(!ego_int_node_.all_actions_expanded())
        {return false;}

        for(auto it = other_int_nodes_.begin(); it != other_int_nodes_.end(); ++it)
        {
            if (!*it->all_actions_expanded())
            {return false;}
        }
    }

    template<class S, class SE, class SO, class H>
    void StageNode<S,SE, SO, H>::update_statistics(const std::vector<SE>& heuristic_estimates)
    {
        // todo only update value
        ego_int_node_.update_from_heuristic(heuristic_estimates[S::ego_agent_idx]);
        for (auto it = other_int_nodes_.begin(); it != other_int_nodes_.end(); ++it)
        {
            it->update_from_heuristic(heuristic_estimates[it->get_agent_idx()]);
        }
    }

    template<class S, class SE, class SO, class H>
    void StageNode<S,SE, SO, H>::update_statistics(const StageNodeSPtr &changed_child_node) {
        ego_int_node_.update_statistic(changed_child_node->ego_int_node_);
        for (auto it = other_int_nodes_.begin(); it != other_int_nodes_.end(); ++it)
        {
            it->update_statistic(changed_child_node->other_int_nodes_[it->get_agent_idx()-1]); // -1: Ego Agent is at zero, but not contained in other int nodes
        }
    }

    template<class S, class SE, class SO, class H>
    ActionIdx StageNode<S,SE, SO, H>::get_best_action(){
        ActionIdx best = ego_int_node_.get_best_action();
        return best;
    }

    template<class S, class SE, class SO, class H>
    std::string StageNode<S,SE, SO, H>::sprintf() const
    {
        // todo std stringstream segfaults with gcc 7.3
        auto tabs = [](const unsigned int& depth) -> std::string
        { return std::string(depth,'\t'); };

            std::stringstream ss;
        std::cout << tabs(depth_) << "StageNode: ID " << id_ ;

        if(!joint_action_.empty())
        {
            std::cout << ", Joint Action " << joint_action_;
        }
        std::cout << ", " << state_->sprintf() << ", Stats: { (0) " << ego_int_node_.sprintf();
        for (int i = 0; i < other_int_nodes_.size(); ++i)
        {
            std::cout << ", (" << i+1 << ") " << other_int_nodes_[i].sprintf();
        }
        std::cout << "}" << std::endl;

        if(!children_.empty())
        {
            for (auto it = children_.begin(); it != children_.end(); ++it)
                std::cout  << it->second->sprintf() ;

        }
        return ss.str();
    };

    template<class S, class SE, class SO, class H>
    void StageNode<S,SE, SO, H>::printTree(std::string filename) 
    {      
        std::ofstream outfile (filename+".gv");
        outfile << "digraph G {" << std::endl;
        outfile << "label = \"MCTS with Exploration constant = "<< mcts::MctsParameters::EXPLORATION_CONSTANT << "\";" << std::endl;
        outfile << "labelloc = \"t\";" << std::endl;
        outfile.close();

        this->printLayer(filename);
                        
        outfile.open(filename+".gv",std::ios::app);
        outfile << "}" << std::endl;
        outfile.close();
    };

    template<class S, class SE, class SO, class H>
    void StageNode<S,SE, SO, H>::printLayer(std::string filename) {
        std::ofstream logging;
        logging.open(filename+".gv",std::ios::app);
        // DRAW SUBGRAPH FOR THIS STAGE
        logging << "subgraph cluster_node_" << this->id_<< "{" << std::endl;
        logging << "node" << this->id_ << "_" << int(ego_int_node_.get_agent_idx()) << "[label=\""<< ego_int_node_.print_node_information()
                                                            << " \n Ag." << int(ego_int_node_.get_agent_idx()) << "\"]" <<";" << std::endl;
        for (auto other_agent_it = other_int_nodes_.begin(); other_agent_it != other_int_nodes_.end(); ++other_agent_it) {
            logging << "node" << this->id_ << "_" << int(other_agent_it->get_agent_idx())  << "[label=\""<< other_agent_it->print_node_information()
                                                        << " \n Ag." << int(other_agent_it->get_agent_idx()) << "\"]" <<";" << std::endl;
        }
        logging << "label= \"ID " << this->id_ << "\";" << std::endl;
        logging << "graph[style=dotted]; }" << std::endl;

        // DRAW ARROWS FOR EACH CHILD
        for (auto child_it = this->children_.begin(); child_it != this->children_.end(); ++child_it){
            child_it->second->printLayer(filename);
            
            // ego intermediate node
            logging << "node" << this->id_ << "_" << int(ego_int_node_.get_agent_idx()) <<" -> "
                    << "node" << child_it->second->id_<< "_" << int(ego_int_node_.get_agent_idx()) <<
                    "[label=\""<< ego_int_node_.print_edge_information(ActionIdx(child_it->first[ego_int_node_.get_agent_idx()])) <<"\"]" <<";" << std::endl;
            // other intermediate nodes
            for (auto other_int_it = other_int_nodes_.begin(); other_int_it != other_int_nodes_.end(); ++other_int_it) {
                logging << "node" << this->id_ << "_" << int(other_int_it->get_agent_idx()) <<" -> "
                        << "node" << child_it->second->id_<< "_" << int(other_int_it->get_agent_idx()) <<
                        "[label=\""<< other_int_it->print_edge_information(ActionIdx(child_it->first[other_int_it->get_agent_idx()])) <<"\"]" <<";" << std::endl;

            }
        }
    };

} // namespace mcts
 
#endif