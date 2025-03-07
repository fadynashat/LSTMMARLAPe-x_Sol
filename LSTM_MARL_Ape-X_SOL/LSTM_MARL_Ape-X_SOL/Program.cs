using System;
using System.Collections.Generic;
using System.Linq;

namespace ResourceAllocation
{
    // Resource Types
    public enum ResourceType { CPU, Memory, Storage }

    // Resource Allocation Request
    public class ResourceRequest
    {
        public ResourceType Type { get; set; }
        public double Amount { get; set; } // Amount of resource requested
    }

    // Resource Allocation System
    public class ResourceAllocator
    {
        private Dictionary<ResourceType, double> availableResources;

        public ResourceAllocator()
        {
            // Initialize available resources
            availableResources = new Dictionary<ResourceType, double>
            {
                { ResourceType.CPU, 100.0 },    // 100% CPU
                { ResourceType.Memory, 1024.0 }, // 1024 MB Memory
                { ResourceType.Storage, 5000.0 } // 5000 MB Storage
            };
        }

        public bool AllocateResource(ResourceRequest request)
        {
            if (availableResources[request.Type] >= request.Amount)
            {
                availableResources[request.Type] -= request.Amount;
                Console.WriteLine($"Allocated {request.Amount} of {request.Type}");
                return true;
            }
            else
            {
                Console.WriteLine($"Insufficient {request.Type} to allocate {request.Amount}");
                return false;
            }
        }

        public void ReleaseResource(ResourceRequest request)
        {
            availableResources[request.Type] += request.Amount;
            Console.WriteLine($"Released {request.Amount} of {request.Type}");
        }

        public void PrintAvailableResources()
        {
            Console.WriteLine("Available Resources:");
            foreach (var resource in availableResources)
                Console.WriteLine($"{resource.Key}: {resource.Value}");
        }
    }

    // LSTM Module for Workload Prediction
    public class LSTM
    {
        public double PredictWorkload(double[] input)
        {
            // Replace this with a real LSTM model trained on historical data
            // For now, simulate prediction using a weighted sum of input features
            double[] weights = { 0.4, 0.3, 0.1, 0.1, 0.05, 0.05 }; // Example weights
            double predictedWorkload = input.Zip(weights, (x, w) => x * w).Sum();
            return predictedWorkload;
        }
    }

    // Deep Q-Learning Agent for Resource Allocation Decisions
    public class DeepQLearningAgent
    {
        private double epsilon; // Exploration rate
        private Dictionary<string, double> QTable; // Q-values

        public DeepQLearningAgent(double epsilon = 0.1)
        {
            this.epsilon = epsilon;
            QTable = new Dictionary<string, double>();
        }

        public string ChooseAction(string state)
        {
            Random rand = new Random();
            if (rand.NextDouble() < epsilon)
                return "Explore"; // Random action
            else
                return "Exploit"; // Best action based on Q-table
        }

        public void UpdateQValue(string state, string action, double reward, string nextState)
        {
            // Simplified Deep Q-Learning update rule
            double oldQValue = QTable.ContainsKey(state) ? QTable[state] : 0;
            double maxNextQValue = QTable.ContainsKey(nextState) ? QTable[nextState] : 0;
            double newQValue = oldQValue + 0.1 * (reward + 0.9 * maxNextQValue - oldQValue);
            QTable[state] = newQValue;
        }
    }

    // Main Program
    class Program
    {
        static void Main(string[] args)
        {
            // Initialize Resource Allocator, LSTM, and Deep Q-Learning Agent
            ResourceAllocator allocator = new ResourceAllocator();
            LSTM lstm = new LSTM();
            DeepQLearningAgent agent = new DeepQLearningAgent(epsilon: 0.1);

            // Simulate resource allocation loop
            for (int episode = 0; episode < 10; episode++)
            {
                Console.WriteLine($"\n--- Episode {episode + 1} ---");

                // Simulate input data (replace with real data)
                double[] input = new double[6];
                input[0] = new Random().NextDouble() * 100; // CPU usage (%)
                input[1] = new Random().NextDouble() * 1024; // Memory usage (MB)
                input[2] = new Random().NextDouble() * 100; // Network traffic (Mbps)
                input[3] = new Random().NextDouble() * 50;  // Disk I/O (MB/s)
                input[4] = new Random().NextDouble() * 200; // Request rate (requests/second)
                input[5] = new Random().NextDouble() * 10;  // Response time (ms)

                // Predict workload using LSTM
                double predictedWorkload = lstm.PredictWorkload(input);
                Console.WriteLine($"Predicted Workload: {predictedWorkload}%");

                // Create resource request based on predicted workload
                ResourceRequest request = new ResourceRequest
                {
                    Type = ResourceType.CPU,
                    Amount = predictedWorkload // Request CPU equal to predicted workload
                };

                // Choose action using Deep Q-Learning Agent
                string action = agent.ChooseAction("CurrentState");

                // Allocate or release resources based on action
                if (action == "Exploit" && allocator.AllocateResource(request))
                {
                    Console.WriteLine("Resource allocated successfully.");
                }
                else
                {
                    Console.WriteLine("Resource allocation failed or exploration mode.");
                }

                // Simulate reward (e.g., based on resource utilization)
                double reward = allocator.AllocateResource(request) ? 1.0 : -1.0;
                agent.UpdateQValue("CurrentState", action, reward, "NextState");

                // Print available resources
                allocator.PrintAvailableResources();

                // Dynamic Resource Scaling
                if (predictedWorkload > 80) // Scale up if workload is high
                {
                    Console.WriteLine("Scaling up resources...");
                    allocator.AllocateResource(new ResourceRequest { Type = ResourceType.CPU, Amount = 20 });
                }
                else if (predictedWorkload < 20) // Scale down if workload is low
                {
                    Console.WriteLine("Scaling down resources...");
                    allocator.ReleaseResource(new ResourceRequest { Type = ResourceType.CPU, Amount = 20 });
                }
            }
        }
    }
}