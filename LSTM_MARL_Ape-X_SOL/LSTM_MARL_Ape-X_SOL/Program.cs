using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Globalization;

namespace ResourceAllocation
{
    public class DataPoint
    {
        public DateTime Timestamp { get; set; }
        public string VmId { get; set; }
        public double CpuUsage { get; set; }
        public double MemoryUsage { get; set; }
        public double DiskIO { get; set; }
        public double NetworkIO { get; set; }
        public double RequestRate { get; set; } // New property
        public double ResponseTime { get; set; } // New property
    }

    public class DataPreprocessor
    {
        // Load data from a CSV file
        public List<DataPoint> LoadData(string filePath)
        {
            var data = new List<DataPoint>();
            using (var reader = new StreamReader(filePath))
            {
                // Skip header
                reader.ReadLine();

                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();

                    // Skip empty lines
                    if (string.IsNullOrWhiteSpace(line))
                        continue;

                    var values = line.Split(',');

                    // Skip rows with incorrect number of columns
                    if (values.Length != 8) // Updated to 8 columns
                        continue;

                    var dataPoint = new DataPoint
                    {
                        Timestamp = DateTime.ParseExact(values[0], "yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture),
                        VmId = values[1],
                        CpuUsage = TryParseDouble(values[2]),
                        MemoryUsage = TryParseDouble(values[3]),
                        DiskIO = TryParseDouble(values[4]),
                        NetworkIO = TryParseDouble(values[5]),
                        RequestRate = TryParseDouble(values[6]), // Parse RequestRate
                        ResponseTime = TryParseDouble(values[7]) // Parse ResponseTime
                    };

                    data.Add(dataPoint);
                }
            }
            return data;
        }

        // Helper method to safely parse double values
        private double TryParseDouble(string value)
        {
            if (double.TryParse(value, out double result))
                return result;
            return double.NaN; // Return NaN for invalid values
        }

        // Handle missing values by filling with the mean (or default to zero)
        public void HandleMissingValues(List<DataPoint> data)
        {
            double cpuMean = data.Where(d => !double.IsNaN(d.CpuUsage)).Any() ?
                             data.Where(d => !double.IsNaN(d.CpuUsage)).Average(d => d.CpuUsage) : 0;

            double memoryMean = data.Where(d => !double.IsNaN(d.MemoryUsage)).Any() ?
                                data.Where(d => !double.IsNaN(d.MemoryUsage)).Average(d => d.MemoryUsage) : 0;

            double diskMean = data.Where(d => !double.IsNaN(d.DiskIO)).Any() ?
                              data.Where(d => !double.IsNaN(d.DiskIO)).Average(d => d.DiskIO) : 0;

            double networkMean = data.Where(d => !double.IsNaN(d.NetworkIO)).Any() ?
                                 data.Where(d => !double.IsNaN(d.NetworkIO)).Average(d => d.NetworkIO) : 0;

            double requestMean = data.Where(d => !double.IsNaN(d.RequestRate)).Any() ?
                                 data.Where(d => !double.IsNaN(d.RequestRate)).Average(d => d.RequestRate) : 0;

            double responseMean = data.Where(d => !double.IsNaN(d.ResponseTime)).Any() ?
                                  data.Where(d => !double.IsNaN(d.ResponseTime)).Average(d => d.ResponseTime) : 0;

            foreach (var point in data)
            {
                if (double.IsNaN(point.CpuUsage)) point.CpuUsage = cpuMean;
                if (double.IsNaN(point.MemoryUsage)) point.MemoryUsage = memoryMean;
                if (double.IsNaN(point.DiskIO)) point.DiskIO = diskMean;
                if (double.IsNaN(point.NetworkIO)) point.NetworkIO = networkMean;
                if (double.IsNaN(point.RequestRate)) point.RequestRate = requestMean;
                if (double.IsNaN(point.ResponseTime)) point.ResponseTime = responseMean;
            }
        }

        // Normalize data to [0, 1] range
        public void NormalizeData(List<DataPoint> data)
        {
            var cpuMax = data.Max(d => d.CpuUsage);
            var memoryMax = data.Max(d => d.MemoryUsage);
            var diskMax = data.Max(d => d.DiskIO);
            var networkMax = data.Max(d => d.NetworkIO);
            var requestMax = data.Max(d => d.RequestRate);
            var responseMax = data.Max(d => d.ResponseTime);

            // Ensure no division by zero
            cpuMax = cpuMax == 0 ? 1 : cpuMax;
            memoryMax = memoryMax == 0 ? 1 : memoryMax;
            diskMax = diskMax == 0 ? 1 : diskMax;
            networkMax = networkMax == 0 ? 1 : networkMax;
            requestMax = requestMax == 0 ? 1 : requestMax;
            responseMax = responseMax == 0 ? 1 : responseMax;

            foreach (var point in data)
            {
                point.CpuUsage /= cpuMax;
                point.MemoryUsage /= memoryMax;
                point.DiskIO /= diskMax;
                point.NetworkIO /= networkMax;
                point.RequestRate /= requestMax;
                point.ResponseTime /= responseMax;
            }
        }

        // Split data into training and test sets
        public (List<DataPoint> train, List<DataPoint> test) SplitData(List<DataPoint> data, double trainRatio = 0.8)
        {
            var trainSize = (int)(data.Count * trainRatio);
            var trainData = data.Take(trainSize).ToList();
            var testData = data.Skip(trainSize).ToList();
            return (trainData, testData);
        }

        // Save the processed data to a new CSV file
        public void SaveData(string filePath, List<DataPoint> data)
        {
            using (var writer = new StreamWriter(filePath))
            {
                // Write header
                writer.WriteLine("Timestamp,VmId,CpuUsage,MemoryUsage,DiskIO,NetworkIO,RequestRate,ResponseTime");

                // Write data rows
                foreach (var point in data)
                {
                    writer.WriteLine($"{point.Timestamp:yyyy-MM-dd HH:mm:ss},{point.VmId},{point.CpuUsage},{point.MemoryUsage},{point.DiskIO},{point.NetworkIO},{point.RequestRate},{point.ResponseTime}");
                }
            }
        }
    }

    public class ModelInput
    {
        [LoadColumn(2)] public float CpuUsage { get; set; }
        [LoadColumn(3)] public float MemoryUsage { get; set; }
        [LoadColumn(4)] public float DiskIO { get; set; }
        [LoadColumn(5)] public float NetworkIO { get; set; }
        [LoadColumn(6)] public float RequestRate { get; set; }
        [LoadColumn(7)] public float ResponseTime { get; set; } // Target variable
    }

    public class ModelOutput
    {
        [ColumnName("Score")] public float PredictedResponseTime { get; set; }
    }

    public class ModelTrainer
    {
        public static void TrainModel(MLContext mlContext, string dataPath, string modelPath)
        {
            // Load data
            IDataView dataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                path: dataPath, separatorChar: ',', hasHeader: true);

            // Split data into training and testing sets
            var trainTestSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            var trainData = trainTestSplit.TrainSet;
            var testData = trainTestSplit.TestSet;

            // Define training pipeline
            var pipeline = mlContext.Transforms.Concatenate("Features", nameof(ModelInput.CpuUsage),
                                                                  nameof(ModelInput.MemoryUsage),
                                                                  nameof(ModelInput.DiskIO),
                                                                  nameof(ModelInput.NetworkIO),
                                                                  nameof(ModelInput.RequestRate))
                .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: nameof(ModelInput.ResponseTime), featureColumnName: "Features"));

            // Train the model
            var model = pipeline.Fit(trainData);

            // Evaluate the model
            var predictions = model.Transform(testData);
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: nameof(ModelInput.ResponseTime));

            Console.WriteLine($"R² Score: {metrics.RSquared}");
            Console.WriteLine($"Mean Absolute Error: {metrics.MeanAbsoluteError}");
            Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError}");

            // Save the model
            mlContext.Model.Save(model, trainData.Schema, modelPath);

            Console.WriteLine($"Model training complete and saved to '{modelPath}'");
        }
    }

    // Resource Types
    public enum ResourceType { CPU, Memory, Storage, DiskIO, NetworkIO }

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
                { ResourceType.CPU, 100.0 },       // 100% CPU
                { ResourceType.Memory, 1024.0 },   // 1024 MB Memory
                { ResourceType.Storage, 5000.0 },  // 5000 MB Storage
                { ResourceType.DiskIO, 200.0 },    // 200 MB/s Disk I/O
                { ResourceType.NetworkIO, 1000.0 }  // 1000 Mbps Network I/O
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

    // Enhanced LSTM Module for Workload Prediction
    public class LSTM
    {
        private int inputSize;  // Size of input features
        private int hiddenSize; // Size of hidden state
        private double[] hiddenState; // Hidden state vector
        private double[] cellState;   // Cell state vector
        private double[][] weights;   // Weight matrices for input, forget, output, and cell state
        private double[] biases;      // Bias terms
        private double learningRate;  // Learning rate for gradient descent

        public LSTM(int inputSize, int hiddenSize, double learningRate = 0.01)
        {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.learningRate = learningRate;

            // Initialize hidden state and cell state
            hiddenState = new double[hiddenSize];
            cellState = new double[hiddenSize];

            // Initialize weights and biases
            weights = new double[4][]; // 4 gates: input, forget, output, cell state
            for (int i = 0; i < 4; i++)
            {
                weights[i] = new double[inputSize + hiddenSize];
                for (int j = 0; j < weights[i].Length; j++)
                {
                    weights[i][j] = new Random().NextDouble() * 0.1; // Small random weights
                }
            }

            biases = new double[4];
            for (int i = 0; i < 4; i++)
            {
                biases[i] = new Random().NextDouble() * 0.1; // Small random biases
            }
        }

        // Sigmoid activation function
        private double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        // Tanh activation function
        private double Tanh(double x)
        {
            return Math.Tanh(x);
        }

        // Forward pass for a single time step
        private (double[] hiddenState, double[] cellState) ForwardPass(double[] input, double[] prevHiddenState, double[] prevCellState)
        {
            // Concatenate input and previous hidden state
            var combinedInput = new double[input.Length + prevHiddenState.Length];
            input.CopyTo(combinedInput, 0);
            prevHiddenState.CopyTo(combinedInput, input.Length);

            // Compute gate activations
            var forgetGate = Sigmoid(DotProduct(weights[0], combinedInput) + biases[0]);
            var inputGate = Sigmoid(DotProduct(weights[1], combinedInput) + biases[1]);
            var outputGate = Sigmoid(DotProduct(weights[2], combinedInput) + biases[2]);
            var candidateCellState = Tanh(DotProduct(weights[3], combinedInput) + biases[3]);

            // Update cell state
            var cellState = new double[hiddenSize];
            for (int i = 0; i < hiddenSize; i++)
            {
                cellState[i] = forgetGate * prevCellState[i] + inputGate * candidateCellState;
            }

            // Update hidden state
            var hiddenState = new double[hiddenSize];
            for (int i = 0; i < hiddenSize; i++)
            {
                hiddenState[i] = outputGate * Tanh(cellState[i]);
            }

            return (hiddenState, cellState);
        }

        // Compute predictions for a sequence of inputs
        public double[] Predict(double[][] inputs)
        {
            var predictions = new double[inputs.Length];
            var prevHiddenState = new double[hiddenSize];
            var prevCellState = new double[hiddenSize];

            for (int t = 0; t < inputs.Length; t++)
            {
                var (hiddenState, cellState) = ForwardPass(inputs[t], prevHiddenState, prevCellState);
                predictions[t] = hiddenState.Sum(); // Simulate prediction as sum of hidden state
                prevHiddenState = hiddenState;
                prevCellState = cellState;
            }

            return predictions;
        }

        // Train the LSTM model
        public void Train(double[][] inputs, double[] targets, int epochs = 100, int batchSize = 32)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                for (int batchStart = 0; batchStart < inputs.Length; batchStart += batchSize)
                {
                    int batchEnd = Math.Min(batchStart + batchSize, inputs.Length);
                    var batchInputs = inputs[batchStart..batchEnd];
                    var batchTargets = targets[batchStart..batchEnd];

                    // Forward pass
                    var predictions = Predict(batchInputs);

                    // Compute loss (Mean Squared Error)
                    double loss = 0;
                    for (int i = 0; i < predictions.Length; i++)
                    {
                        loss += Math.Pow(predictions[i] - batchTargets[i], 2);
                    }
                    loss /= predictions.Length;

                    // Backward pass (simplified gradient descent)
                    for (int i = 0; i < weights.Length; i++)
                    {
                        for (int j = 0; j < weights[i].Length; j++)
                        {
                            // Simplified gradient update
                            weights[i][j] -= learningRate * loss;
                        }
                    }

                    for (int i = 0; i < biases.Length; i++)
                    {
                        // Simplified gradient update
                        biases[i] -= learningRate * loss;
                    }

                    Console.WriteLine($"Epoch {epoch + 1}, Batch {batchStart / batchSize + 1}, Loss: {loss}");
                }
            }
        }

        // Helper method to compute dot product
        private double DotProduct(double[] a, double[] b)
        {
            double result = 0;
            for (int i = 0; i < a.Length; i++)
            {
                result += a[i] * b[i];
            }
            return result;
        }
    }

    // Enhanced MARL Components
    public class ReplayBuffer
    {
        private List<(string state, string action, double reward, string nextState)> buffer;
        private int capacity;
        private Random random = new Random();

        public ReplayBuffer(int capacity = 10000)
        {
            buffer = new List<(string, string, double, string)>();
            this.capacity = capacity;
        }

        public void AddExperience(string state, string action, double reward, string nextState)
        {
            if (buffer.Count >= capacity) buffer.RemoveAt(0);
            buffer.Add((state, action, reward, nextState));
        }

        public List<(string state, string action, double reward, string nextState)> SampleBatch(int batchSize)
        {
            return buffer.OrderBy(x => random.Next()).Take(batchSize).ToList();
        }
    }


    // Deep Q-Learning Agent for Resource Allocation Decisions
    public class DeepQLearningAgent
    {
        private double _epsilon;
        private Dictionary<string, double> _qTable;
        private Random _random;

        public DeepQLearningAgent(double epsilon = 0.1)
        {
            _epsilon = epsilon;
            _qTable = new Dictionary<string, double>();
            _random = new Random();
        }

        public string ChooseAction(string state)
        {
            if (_random.NextDouble() < _epsilon)
                return "Explore"; // Random action
            else
                return GetBestAction(state);
        }

        private string GetBestAction(string state)
        {
            // Simplified: Assume two possible actions
            return _qTable.ContainsKey(state) && _qTable[state] > 0 ? "Exploit" : "Explore";
        }

        public double GetMaxQValue(string state)
        {
            return _qTable.ContainsKey(state) ? _qTable[state] : 0;
        }

        public void UpdateQValue(string state, string action, double targetQ, string nextState)
        {
            // Simplified Q-update rule
            double oldQ = _qTable.ContainsKey(state) ? _qTable[state] : 0;
            _qTable[state] = oldQ + 0.1 * (targetQ - oldQ);
        }
    }





    // Enhanced APe-X Replay Buffer
    public class APeXReplayBuffer
    {
        private List<(string state, string action, double reward, string nextState, double priority)> _buffer;
        private int _capacity;
        private double _maxPriority;
        private Random _random = new Random();

        public APeXReplayBuffer(int capacity = 10000)
        {
            _buffer = new List<(string, string, double, string, double)>();
            _capacity = capacity;
            _maxPriority = 1.0;
        }

        public void AddExperience(string state, string action, double reward, string nextState)
        {
            // New experiences get max priority to ensure they're sampled
            if (_buffer.Count >= _capacity)
                _buffer.RemoveAt(0);

            _buffer.Add((state, action, reward, nextState, _maxPriority));
        }

        public (List<(string state, string action, double reward, string nextState)> batch, List<double> weights)
            SampleBatch(int batchSize, double threshold, double beta)
        {
            var candidates = _buffer.Where(e => e.priority > threshold).ToList();
            if (candidates.Count == 0) candidates = _buffer.ToList();

            // Prioritized sampling
            var sorted = candidates.OrderByDescending(e => e.priority).Take(batchSize).ToList();

            // Calculate importance weights
            double totalPriority = _buffer.Sum(e => e.priority);
            var weights = sorted.Select(e =>
                Math.Pow(1.0 / (_buffer.Count * e.priority), beta)
                ).ToList();

            return (
                sorted.Select(e => (e.state, e.action, e.reward, e.nextState)).ToList(),
                weights
            );
        }

        public void UpdatePriorities(List<(string state, string action, double newPriority)> updates)
        {
            foreach (var update in updates)
            {
                var index = _buffer.FindIndex(e =>
                    e.state == update.state && e.action == update.action);

                if (index != -1)
                {
                    _buffer[index] = (
                        _buffer[index].state,
                        _buffer[index].action,
                        _buffer[index].reward,
                        _buffer[index].nextState,
                        update.newPriority
                    );
                    _maxPriority = Math.Max(_maxPriority, update.newPriority);
                }
            }
        }
    }

    // Enhanced MARL Agent with APe-X

public class MARLAgent
    {
        private Dictionary<string, double> _qTable = new Dictionary<string, double>();
        private APeXReplayBuffer _replayBuffer;
        private Random _random = new Random();
        private double _epsilon;
        private double _gamma;
        private double _alpha;
        private double _beta;
        private double _priorityThreshold;
        private int _batchSize;

        public MARLAgent(double epsilon = 0.1, double gamma = 0.9, double alpha = 0.1,
                        double beta = 0.5, double priorityThreshold = 0.1, int batchSize = 32)
        {
            _epsilon = epsilon;
            _gamma = gamma;
            _alpha = alpha;
            _beta = beta;
            _priorityThreshold = priorityThreshold;
            _batchSize = batchSize;
            _replayBuffer = new APeXReplayBuffer();
        }

        public void StoreExperience(string state, string action, double reward, string nextState)
        {
            _replayBuffer.AddExperience(state, action, reward, nextState);
        }

        public void Train()
        {
            var (batch, weights) = _replayBuffer.SampleBatch(_batchSize, _priorityThreshold, _beta);
            var priorityUpdates = new List<(string, string, double)>();

            for (int i = 0; i < batch.Count; i++)
            {
                var (state, action, reward, nextState) = batch[i];
                double weight = weights[i];

                double currentQ = GetQValue(state, action);
                double maxNextQ = GetMaxNextQ(nextState);
                double targetQ = reward + _gamma * maxNextQ;
                double tdError = Math.Abs(targetQ - currentQ);

                double newQ = currentQ + _alpha * weight * (targetQ - currentQ);
                _qTable[$"{state}|{action}"] = newQ;

                priorityUpdates.Add((state, action, tdError + 1e-6)); // Small constant to prevent zero priority
            }

            _replayBuffer.UpdatePriorities(priorityUpdates);
        }

        public string ChooseAction(string state)
        {
            if (_random.NextDouble() < _epsilon)
                return GetRandomAction();

            return GetBestAction(state);
        }

        private string GetRandomAction()
        {
            var actions = new[] { "ScaleUpCPU", "ScaleDownCPU", "ScaleUpMemory", "ScaleDownMemory" };
            return actions[_random.Next(actions.Length)];
        }

        private string GetBestAction(string state)
        {
            var actions = new[] { "ScaleUpCPU", "ScaleDownCPU", "ScaleUpMemory", "ScaleDownMemory" };
            return actions.OrderByDescending(a => GetQValue(state, a)).First();
        }

        private double GetQValue(string state, string action)
        {
            return _qTable.TryGetValue($"{state}|{action}", out var q) ? q : 0;
        }

        private double GetMaxNextQ(string nextState)
        {
            var actions = new[] { "ScaleUpCPU", "ScaleDownCPU", "ScaleUpMemory", "ScaleDownMemory" };
            return actions.Max(a => GetQValue(nextState, a));
        }
    }

    // Modified MultiAgentSystem initialization
    public class MultiAgentSystem
    {
        private List<MARLAgent> agents;
        private ResourceAllocator allocator;
        private LSTM workloadPredictor;
        private double slaThreshold;

        public MultiAgentSystem(int numAgents, ResourceAllocator allocator, LSTM lstm, double slaThreshold = 0.9)
        {
            agents = new List<MARLAgent>();
            for (int i = 0; i < numAgents; i++)
            {
                agents.Add(new MARLAgent(
                    beta: Math.Min(0.4 + i * 0.1, 1.0), // Anneal beta over agents
                    priorityThreshold: 0.2
                ));
            }
            this.allocator = allocator;
            this.workloadPredictor = lstm;
            this.slaThreshold = slaThreshold;
        }

        public void ExecuteStep(List<DataPoint> currentMetrics)
        {
            var input = currentMetrics.Select(dp => new[] {
            dp.CpuUsage, dp.MemoryUsage, dp.DiskIO,
            dp.NetworkIO, dp.RequestRate, dp.ResponseTime
        }).ToArray();

            var predictions = workloadPredictor.Predict(input);

            foreach (var agent in agents)
            {
                var state = GetCurrentState(currentMetrics);
                var augmentedState = $"{state}|Pred:{string.Join(",", predictions)}";

                var action = agent.ChooseAction(augmentedState);
                var (reward, nextState) = ExecuteAction(action, currentMetrics);

                agent.StoreExperience(augmentedState, action, reward, nextState);
                agent.Train();
            }
        }

        private (double reward, string nextState) ExecuteAction(string action, List<DataPoint> metrics)
        {
            var request = ParseAction(action);
            var success = allocator.AllocateResource(request);

            double responseTime = metrics.Average(m => m.ResponseTime);
            double slaReward = responseTime < slaThreshold ? 1 : -1;

            double reward = (success ? 0.5 : -0.5) + slaReward;

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



    // Main Program
    class Program
    {
        private static string outputFolder = Path.Combine(Directory.GetCurrentDirectory(), "Output");

        static void Main(string[] args)
        {
            // Initialize components
            Directory.CreateDirectory(outputFolder);
            var mlContext = new MLContext();
            var allocator = new ResourceAllocator();
            var lstm = new LSTM(6, 10);

            // Load and preprocess data
            var preprocessor = new DataPreprocessor();
            var rawData = preprocessor.LoadData(Path.Combine(outputFolder, "synthetic_cloud_dataset.csv"));
            preprocessor.HandleMissingValues(rawData);
            preprocessor.NormalizeData(rawData);

            // Train LSTM
            var inputs = rawData.Select(dp => new[] {
                dp.CpuUsage, dp.MemoryUsage, dp.DiskIO,
                dp.NetworkIO, dp.RequestRate, dp.ResponseTime
            }).ToArray();

            var targets = rawData.Select(dp => dp.ResponseTime).ToArray();
            lstm.Train(inputs, targets);

            // Initialize MARL system
            var multiAgentSystem = new MultiAgentSystem(
                numAgents: 3,
                allocator: allocator,
                lstm: lstm
            );

            // MARL training loop
            var windowSize = 10;
            for (int i = 0; i < rawData.Count - windowSize; i++)
            {
                var window = rawData.Skip(i).Take(windowSize).ToList();
                multiAgentSystem.ExecuteStep(window);

                if (i % 100 == 0)
                    Console.WriteLine($"Processed {i} records...");
            }

            // Final resource status
            allocator.PrintAvailableResources();
            Console.WriteLine("MARL training complete!");
        }
    }
}

