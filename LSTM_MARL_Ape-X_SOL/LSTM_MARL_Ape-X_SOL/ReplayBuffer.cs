// Enhanced Multi-Agent Reinforcement Learning (MARL) Replay Buffer
public class ReplayBuffer
{
    // Stores experiences in the form of (state, action, reward, next state)
    private List<(string state, string action, double reward, string nextState)> buffer;

    // Maximum capacity of the replay buffer
    private int capacity;

    // Random number generator for sampling experiences
    private Random random = new Random();

    // Constructor to initialize the replay buffer with a specified capacity
    public ReplayBuffer(int capacity = 10000)
    {
        buffer = new List<(string, string, double, string)>(); // Initialize buffer
        this.capacity = capacity; // Set the maximum buffer size
    }

    // Adds a new experience (state, action, reward, next state) to the buffer
    public void AddExperience(string state, string action, double reward, string nextState)
    {
        // If the buffer is full, remove the oldest experience (FIFO strategy)
        if (buffer.Count >= capacity)
            buffer.RemoveAt(0);

        // Add the new experience to the buffer
        buffer.Add((state, action, reward, nextState));
    }

    // Samples a batch of experiences randomly from the buffer for training
    public List<(string state, string action, double reward, string nextState)> SampleBatch(int batchSize)
    {
        // Randomly shuffle the buffer and take 'batchSize' elements for training
        return buffer.OrderBy(x => random.Next()).Take(batchSize).ToList();
    }
}
