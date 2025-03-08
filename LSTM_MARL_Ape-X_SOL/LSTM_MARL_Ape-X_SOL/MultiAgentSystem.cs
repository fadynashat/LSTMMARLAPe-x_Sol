// MultiAgentSystem class manages multiple MARLAgents and coordinates resource allocation using reinforcement learning.
using LSTM_MARL_Ape_X_SOL;
using System.Diagnostics;
using System;

public class MultiAgentSystem
{
    // List of MARL agents that will learn and make decisions.
    private List<MARLAgent> agents;
    // ResourceAllocator handles actual resource scaling (CPU, Memory, etc.).
    private ResourceAllocator allocator;
    // LSTM workload predictor to forecast future workload trends.
    private LSTM workloadPredictor;
    // SLA (Service Level Agreement) threshold, used to assess performance.
    private double slaThreshold;

    // Constructor to initialize multiple agents, allocator, workload predictor, and SLA threshold.
    public MultiAgentSystem(int numAgents, ResourceAllocator allocator, LSTM lstm, double slaThreshold = 0.9)
    {
        agents = new List<MARLAgent>();

        // Create multiple MARLAgents with varying beta values (annealed over agents).
        for (int i = 0; i < numAgents; i++)
        {
            agents.Add(new MARLAgent(
                beta: Math.Min(0.4 + i * 0.1, 1.0), // Increases beta gradually for priority sampling.
                priorityThreshold: 0.2 // Sets a threshold for prioritized experience replay.
            ));
        }

        // Store dependencies.
        this.allocator = allocator;
        this.workloadPredictor = lstm;
        this.slaThreshold = slaThreshold;
    }

    // Executes a single decision-making step for all agents based on current metrics.
    public void ExecuteStep(List<DataPoint> currentMetrics)
    {
        // Convert metrics into a structured format for prediction.
        var input = currentMetrics.Select(dp => new[] {
            dp.CpuUsage, dp.MemoryUsage, dp.DiskIO,
            dp.NetworkIO, dp.RequestRate, dp.ResponseTime
        }).ToArray();

        // Predict future workload using LSTM model.
        var predictions = workloadPredictor.Predict(input);

        // Iterate through all agents to make decisions and learn.
        foreach (var agent in agents)
        {
            // Get the current system state based on real-time metrics.
            var state = GetCurrentState(currentMetrics);
            // Augment state with workload predictions for better decision-making.
            var augmentedState = $"{state}|Pred:{string.Join(",", predictions)}";

            // Choose an action based on the current state.
            var action = agent.ChooseAction(augmentedState);
            // Execute the action and get the resulting reward and next state.
            var (reward, nextState) = ExecuteAction(action, currentMetrics);

            // Store the experience for learning.
            agent.StoreExperience(augmentedState, action, reward, nextState);
            // Train the agent using replay buffer and Q-learning.
            agent.Train();
        }
    }

    // Executes the chosen action and determines the reward based on system performance.
    private (double reward, string nextState) ExecuteAction(string action, List<DataPoint> metrics)
    {
        // Convert the action string into a structured resource request.
        var request = ParseAction(action);
        // Allocate resources based on the parsed request.
        var success = allocator.AllocateResource(request);

        // Calculate the average response time from current metrics.
        double responseTime = metrics.Average(m => m.ResponseTime);
        // Determine SLA-based reward: positive if response time is below threshold, negative otherwise.
        double slaReward = responseTime < slaThreshold ? 1 : -1;

        // Final reward combines success of allocation and SLA performance.
        double reward = (success ? 0.5 : -0.5) + slaReward;

        // Return the reward and the updated system state.
        return (reward, GetCurrentState(metrics));
    }

    // Converts an action string into a structured resource request.
    private ResourceRequest ParseAction(string action)
    {
        return action switch
        {
            "ScaleUpCPU" => new ResourceRequest { Type = ResourceType.CPU, Amount = 10 },  // Increase CPU by 10 units.
            "ScaleDownCPU" => new ResourceRequest { Type = ResourceType.CPU, Amount = -10 }, // Decrease CPU by 10 units.
            "ScaleUpMemory" => new ResourceRequest { Type = ResourceType.Memory, Amount = 256 }, // Increase Memory by 256MB.
            "ScaleDownMemory" => new ResourceRequest { Type = ResourceType.Memory, Amount = -256 }, // Decrease Memory by 256MB.
            _ => throw new ArgumentException("Invalid action") // Handle invalid actions.
        };
    }

    // Generates the current state representation based on average CPU and memory usage.
    private string GetCurrentState(List<DataPoint> metrics)
    {
        var avgCpu = metrics.Average(m => m.CpuUsage); // Calculate average CPU usage.
        var avgMem = metrics.Average(m => m.MemoryUsage); // Calculate average Memory usage.
        return $"CPU:{avgCpu:F2}|MEM:{avgMem:F2}"; // Format state string for Q-learning.
    }
}


//MultiAgentSystem initialized.
//2025 - 03 - 08 13:03:32 - Starting MARL training loop with window size 10...
//2025-03-08 13:03:32 - why Processed 0 records...