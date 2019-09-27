
typedef double Belief;
typedef double Probability;

template <typename S, typename ActionType>
class HypothesisBeliefTracker {
  public:
    HypothesisBeliefTracker();
    
    void belief_update(StateInterface<S> state);

    std::unordered_map<AgentIdx, HypothesisId> sample_current_hypothesis() const; // shared across all states

private:
    std::unordered_map<AgentIdx, std::vector<Belief>> tracked_beliefs_;//< contains the beliefs for each hypothesis for each agent 

}