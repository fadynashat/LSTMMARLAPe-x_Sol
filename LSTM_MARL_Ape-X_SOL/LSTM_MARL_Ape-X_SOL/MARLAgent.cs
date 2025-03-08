// Multi-Agent Reinforcement Learning (MARL) agent with APe-X experience replay
public class MARLAgent
{
    // Q-table for storing Q-values of (state, action) pairs
    private Dictionary<string, double> _qTable = new Dictionary<string, double>();

    // APe-X replay buffer to store and sample experiences
    private APeXReplayBuffer _replayBuffer;

    // Random instance for exploration
    private Random _random = new Random();

    // Parameters for reinforcement learning
    private double _epsilon; // Exploration rate
    private double _gamma; // Discount factor for future rewards
    private double _alpha; // Learning rate
    private double _beta; // Importance sampling factor
    private double _priorityThreshold; // Minimum priority for experience selection
    private int _batchSize; // Number of experiences to sample per training step

    // Constructor to initialize agent parameters
    public MARLAgent(double epsilon = 0.1, double gamma = 0.9, double alpha = 0.1,
                    double beta = 0.5, double priorityThreshold = 0.1, int batchSize = 32)
    {
        _epsilon = epsilon;
        _gamma = gamma;
        _alpha = alpha;
        _beta = beta;
        _priorityThreshold = priorityThreshold;
        _batchSize = batchSize;
        _replayBuffer = new APeXReplayBuffer(); // Initialize replay buffer
    }

    // Store an experience in the replay buffer
    public void StoreExperience(string state, string action, double reward, string nextState)
    {
        _replayBuffer.AddExperience(state, action, reward, nextState);
    }

    // Train the agent using experiences sampled from the replay buffer
    public void Train()
    {
        // Sample a batch of experiences along with their importance weights
        var (batch, weights) = _replayBuffer.SampleBatch(_batchSize, _priorityThreshold, _beta);
        var priorityUpdates = new List<(string, string, double)>(); // List to store new priorities

        for (int i = 0; i < batch.Count; i++)
        {
            var (state, action, reward, nextState) = batch[i];
            double weight = weights[i];

            // Get current Q-value for (state, action)
            double currentQ = GetQValue(state, action);

            // Get max Q-value for the next state (used in Bellman equation)
            double maxNextQ = GetMaxNextQ(nextState);

            // Compute target Q-value using the Bellman equation
            double targetQ = reward + _gamma * maxNextQ;

            // Compute TD (Temporal Difference) error
            double tdError = Math.Abs(targetQ - currentQ);

            // Update Q-value using weighted learning rate
            double newQ = currentQ + _alpha * weight * (targetQ - currentQ);
            _qTable[$"{state}|{action}"] = newQ;

            // Update experience priority based on TD error (adding a small constant to avoid zero priority)
            priorityUpdates.Add((state, action, tdError + 1e-6));
        }

        // Update the priorities in the replay buffer based on new TD errors
        _replayBuffer.UpdatePriorities(priorityUpdates);
    }

    // Select an action based on an epsilon-greedy strategy
    public string ChooseAction(string state)
    {
        // With probability epsilon, choose a random action (exploration)
        if (_random.NextDouble() < _epsilon)
            return GetRandomAction();

        // Otherwise, choose the best action based on Q-values (exploitation)
        return GetBestAction(state);
    }

    // Return a random action from the action space
    private string GetRandomAction()
    {
        var actions = new[] { "ScaleUpCPU", "ScaleDownCPU", "ScaleUpMemory", "ScaleDownMemory" };
        return actions[_random.Next(actions.Length)];
    }

    // Return the action with the highest Q-value for the given state
    private string GetBestAction(string state)
    {
        var actions = new[] { "ScaleUpCPU", "ScaleDownCPU", "ScaleUpMemory", "ScaleDownMemory" };
        return actions.OrderByDescending(a => GetQValue(state, a)).First();
    }

    // Retrieve the Q-value for a given (state, action) pair
    private double GetQValue(string state, string action)
    {
        return _qTable.TryGetValue($"{state}|{action}", out var q) ? q : 0; // Default to 0 if not found
    }

    // Get the maximum Q-value for the next state (used in Q-learning update rule)
    private double GetMaxNextQ(string nextState)
    {
        var actions = new[] { "ScaleUpCPU", "ScaleDownCPU", "ScaleUpMemory", "ScaleDownMemory" };
        return actions.Max(a => GetQValue(nextState, a)); // Find the highest Q-value among available actions
    }
}
