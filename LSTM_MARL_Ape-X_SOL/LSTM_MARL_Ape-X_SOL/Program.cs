using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Globalization;

namespace LSTM_MARL_Ape_X_SOL
{
    // Resource Types
    public enum ResourceType { CPU, Memory, Storage, DiskIO, NetworkIO }

    

    // Main Program
    class Program
    {
        private static string outputFolder = Path.Combine(Directory.GetCurrentDirectory(), "Output");

        static void Main(string[] args)
        {
            Logger.Log("Program started.");

            // Ensure the output directory exists
            Directory.CreateDirectory(outputFolder);

            // Initialize ML.NET context
            var mlContext = new MLContext();
            Logger.Log("MLContext initialized.");

            // Initialize Resource Allocator and LSTM model
            var allocator = new ResourceAllocator();
            var lstm = new LSTM(6, 10);
            Logger.Log("ResourceAllocator and LSTM initialized.");

            // Load and preprocess data
            var preprocessor = new DataPreprocessor();
            string dataPath = Path.Combine(outputFolder, "synthetic_cloud_dataset.csv");

            Logger.Log($"Loading data from {dataPath}...");
            var rawData = preprocessor.LoadData(dataPath);
            Logger.Log($"Loaded {rawData.Count} records.");

            // Handle missing values
            Logger.Log("Handling missing values...");
            preprocessor.HandleMissingValues(rawData);
            Logger.Log("Missing values handled.");

            // Normalize the dataset
            Logger.Log("Normalizing data...");
            preprocessor.NormalizeData(rawData);
            Logger.Log("Data normalization complete.");

            // Prepare data for LSTM training
            Logger.Log("Preparing data for LSTM training...");
            var inputs = rawData.Select(dp => new[] {
                dp.CpuUsage, dp.MemoryUsage, dp.DiskIO,
                dp.NetworkIO, dp.RequestRate, dp.ResponseTime
            }).ToArray();

            var targets = rawData.Select(dp => dp.ResponseTime).ToArray();
            Logger.Log("Data prepared for LSTM training.");

            // Train LSTM Model
            Logger.Log("Training LSTM model...");
            lstm.Train(inputs, targets);
            Logger.Log("LSTM training completed.");

            // Initialize Multi-Agent Reinforcement Learning (MARL) system
            Logger.Log("Initializing MultiAgentSystem...");
            var multiAgentSystem = new MultiAgentSystem(
                numAgents: 3,
                allocator: allocator,
                lstm: lstm
            );
            Logger.Log("MultiAgentSystem initialized.");

            // Start MARL training loop
            var windowSize = 10;
            Logger.Log($"Starting MARL training loop with window size {windowSize}...");

            for (int i = 0; i < rawData.Count - windowSize; i++)
            {
                var window = rawData.Skip(i).Take(windowSize).ToList();
                multiAgentSystem.ExecuteStep(window);

                if (i % 100 == 0)
                {
                    Logger.Log($"Processed {i} records...");
                }
            }

            // Final resource status
            Logger.Log("Printing final resource status...");
            allocator.PrintAvailableResources();

            Logger.Log("MARL training complete!");
            Logger.Log("Program execution finished.");
        }
    }
}
