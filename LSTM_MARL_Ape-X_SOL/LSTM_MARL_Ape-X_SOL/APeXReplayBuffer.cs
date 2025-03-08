// Enhanced APe-X Replay Buffer (Prioritized Experience Replay Buffer)
// This class improves reinforcement learning efficiency by prioritizing important experiences.
public class APeXReplayBuffer
{
    // List storing experience tuples: (state, action, reward, next state, priority)
    private List<(string state, string action, double reward, string nextState, double priority)> _buffer;

    // Maximum number of experiences stored in the buffer
    private int _capacity;

    // Stores the highest priority of any experience (to ensure new experiences are sampled first)
    private double _maxPriority;

    // Random number generator for sampling
    private Random _random = new Random();

    // Constructor to initialize the replay buffer with a given capacity
    public APeXReplayBuffer(int capacity = 10000)
    {
        _buffer = new List<(string, string, double, string, double)>();
        _capacity = capacity;
        _maxPriority = 1.0; // Default priority for new experiences
    }

    // Adds a new experience to the buffer with maximum priority
    public void AddExperience(string state, string action, double reward, string nextState)
    {
        // If the buffer is full, remove the oldest experience (FIFO strategy)
        if (_buffer.Count >= _capacity)
            _buffer.RemoveAt(0);

        // New experiences are given the highest priority to ensure they are sampled first
        _buffer.Add((state, action, reward, nextState, _maxPriority));
    }

    // Samples a batch of experiences based on priority
    // Returns: A batch of (state, action, reward, next state) pairs along with importance weights
    public (List<(string state, string action, double reward, string nextState)> batch, List<double> weights)
        SampleBatch(int batchSize, double threshold, double beta)
    {
        // Select experiences that have a priority higher than the given threshold
        var candidates = _buffer.Where(e => e.priority > threshold).ToList();

        // If no experiences meet the threshold, fall back to using the entire buffer
        if (candidates.Count == 0) candidates = _buffer.ToList();

        // Sort experiences by priority in descending order and take the top batchSize elements
        var sorted = candidates.OrderByDescending(e => e.priority).Take(batchSize).ToList();

        // Calculate importance sampling weights to correct bias introduced by prioritization
        double totalPriority = _buffer.Sum(e => e.priority);
        var weights = sorted.Select(e =>
            Math.Pow(1.0 / (_buffer.Count * e.priority), beta) // Importance weight formula
            ).ToList();

        return (
            sorted.Select(e => (e.state, e.action, e.reward, e.nextState)).ToList(),
            weights
        );
    }

    // Updates the priority of specific experiences in the buffer based on learning updates
    public void UpdatePriorities(List<(string state, string action, double newPriority)> updates)
    {
        foreach (var update in updates)
        {
            // Find the index of the experience that matches the given state-action pair
            var index = _buffer.FindIndex(e =>
                e.state == update.state && e.action == update.action);

            // If found, update its priority with the new value
            if (index != -1)
            {
                _buffer[index] = (
                    _buffer[index].state,
                    _buffer[index].action,
                    _buffer[index].reward,
                    _buffer[index].nextState,
                    update.newPriority
                );

                // Keep track of the highest priority for future additions
                _maxPriority = Math.Max(_maxPriority, update.newPriority);
            }
        }
    }
}
