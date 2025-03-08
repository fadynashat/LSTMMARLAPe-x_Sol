// Deep Q-Learning Agent for Resource Allocation Decisions
public class DeepQLearningAgent
{
    // Exploration rate (epsilon) determines the probability of taking a random action
    private double _epsilon;

    // Q-table stores the estimated Q-values for each state-action pair
    private Dictionary<string, double> _qTable;

    // Random number generator for exploration decisions
    private Random _random;

    // Constructor to initialize the agent with an exploration rate
    public DeepQLearningAgent(double epsilon = 0.1)
    {
        _epsilon = epsilon;   // Set the exploration probability
        _qTable = new Dictionary<string, double>();  // Initialize an empty Q-table
        _random = new Random();  // Initialize random number generator
    }

    // Chooses an action based on epsilon-greedy policy
    public string ChooseAction(string state)
    {
        // With probability epsilon, choose a random action (exploration)
        if (_random.NextDouble() < _epsilon)
            return "Explore";  // Random action to encourage exploration

        // Otherwise, choose the best known action (exploitation)
        else
            return GetBestAction(state);
    }

    // Determines the best action based on the Q-table
    private string GetBestAction(string state)
    {
        // If the state exists in the Q-table and has a positive Q-value, choose "Exploit"
        return _qTable.ContainsKey(state) && _qTable[state] > 0 ? "Exploit" : "Explore";
    }

    // Retrieves the highest Q-value for a given state
    public double GetMaxQValue(string state)
    {
        // If the state exists in the Q-table, return its value; otherwise, return 0
        return _qTable.ContainsKey(state) ? _qTable[state] : 0;
    }

    // Updates the Q-value for a given state-action pair using a simplified Q-learning update rule
    public void UpdateQValue(string state, string action, double targetQ, string nextState)
    {
        // Get the old Q-value for the current state
        double oldQ = _qTable.ContainsKey(state) ? _qTable[state] : 0;

        // Update the Q-value using a learning rate of 0.1 (simplified update rule)
        _qTable[state] = oldQ + 0.1 * (targetQ - oldQ);
    }
}
