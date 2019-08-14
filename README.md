# Multi-Agent MCTS
Multi-agent Monte Carlo Tree Search implementation in C++.

## Approach
- Classical Monte Carlo Tree Search extended to the Multi-Agent MDP problem
- Selection Step: 
    - Agents choose actions simultaneusly at each stage (Stage node class contains intermediate nodes for each agent)
    - Action selection based on node statistic class ( UCTStatistic provided). An "Ego"-agent can have a different statistic class than the other agents
    - The JoinAction determines next selected stage
- Expansion Step. node or, for a leaf node, the newly expanded node via execution of the joint action in the environment. 
- Then, a random rollout policy is applied to the newly expanded state selecting joint actions randomly until reaching a terminal state (class random heuristic)
- Statistic updates of each agent separately via backpropagation


## Features
- Recursively tested consistency of UCT tree  
- Interfaces to allow easy extension with other statistics, environments, heuristics.
- Export of trees to graphviz dotfiles.
- Static polymorphic interfaces to avoid dynamic polymorphism runtime overhead (However, the effect may be subtle and was not evaluated yet)

## Installation & Test
- Install [bazel](https://docs.bazel.build/versions/master/install.html)
- Run `bazel test //...` in the WORKSPACE directory
- Under "WORKSPACE directory"/bazel-genfiles/test you find the dot file "test_tree.gy"
- Use `dot test_tree.gv -O -Tsvg` to render a svg-file


## Example

SimpleState is a 1D chain environment with two agents.
The agents move towards the goal state if they select different actions.
They remain at the current state if they select the same action.

20 iterations yield the following search tree with SimpleState.
Stage nodes are enclosed within dotted boxes and contain the statistic informations for each agent in circles.
Edges correspond to an action of one agent with action-value V and visit count N indicated.

![Search tree example](/doc/simple_state_test_tree.svg)

