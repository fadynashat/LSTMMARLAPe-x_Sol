using LSTM_MARL_Ape_X_SOL;
using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;

public class MultiAgentSystem
{
    private List<MARLAgent> agents;
    private ResourceAllocator allocator;
    private LSTM workloadPredictor;
    private double slaThreshold;
  

    public MultiAgentSystem(int numAgents, ResourceAllocator allocator, LSTM lstm, double slaThreshold = 0.9)
    {
        agents = new List<MARLAgent>();
        this.allocator = allocator;
        this.workloadPredictor = lstm;
        this.slaThreshold = slaThreshold;
    

        for (int i = 0; i < numAgents; i++)
        {
            agents.Add(new MARLAgent(
                beta: Math.Min(0.4 + i * 0.1, 1.0),
                priorityThreshold: 0.2
            ));
        }

        Logger.Log("MultiAgentSystem initialized with " + numAgents + " agents.");
    }

    public void ExecuteStep(List<DataPoint> currentMetrics)
    {
        Logger.Log("Executing decision step.");
        var input = currentMetrics.Select(dp => new[] { dp.CpuUsage, dp.MemoryUsage, dp.DiskIO, dp.NetworkIO, dp.RequestRate, dp.ResponseTime }).ToArray();
        var predictions = workloadPredictor.Predict(input);

        Logger.Log("Predicted workload: " + string.Join(", ", predictions));

        foreach (var agent in agents)
        {
            var state = GetCurrentState(currentMetrics);
            var augmentedState = $"{state}|Pred:{string.Join(",", predictions)}";

            var action = agent.ChooseAction(augmentedState);
            var (reward, nextState) = ExecuteAction(action, currentMetrics);

            agent.StoreExperience(augmentedState, action, reward, nextState);
            agent.Train();

            Logger.Log($"Agent chose action: {action}, Reward: {reward}, Next State: {nextState}");
        }
    }

    private (double reward, string nextState) ExecuteAction(string action, List<DataPoint> metrics)
    {
        var request = ParseAction(action);
        var success = allocator.AllocateResource(request);

        double responseTime = metrics.Average(m => m.ResponseTime);
        double slaReward = responseTime < slaThreshold ? 1 : -1;
        double reward = (success ? 0.5 : -0.5) + slaReward;

        Logger.Log($"Action Executed: {action}, Allocation Success: {success}, SLA Reward: {slaReward}, Total Reward: {reward}");
        return (reward, GetCurrentState(metrics));
    }

    private ResourceRequest ParseAction(string action)
    {
        return action switch
        {
            "ScaleUpCPU" => new ResourceRequest { Type = ResourceType.CPU, Amount = 10 },
            "ScaleDownCPU" => new ResourceRequest { Type = ResourceType.CPU, Amount = -10 },
            "ScaleUpMemory" => new ResourceRequest { Type = ResourceType.Memory, Amount = 256 },
            "ScaleDownMemory" => new ResourceRequest { Type = ResourceType.Memory, Amount = -256 },
            _ => throw new ArgumentException("Invalid action")
        };
    }

    private string GetCurrentState(List<DataPoint> metrics)
    {
        var avgCpu = metrics.Average(m => m.CpuUsage);
        var avgMem = metrics.Average(m => m.MemoryUsage);
        return $"CPU:{avgCpu:F2}|MEM:{avgMem:F2}";
    }
}
