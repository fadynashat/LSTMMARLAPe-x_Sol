using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Data;

namespace DataPreprocessing
{
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
        private static string dataPath = "preprocessed_data.csv";

        public static void TrainModel()
        {
            var mlContext = new MLContext();

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
            mlContext.Model.Save(model, trainData.Schema, "trained_model.zip");

            Console.WriteLine("Model training complete and saved to 'trained_model.zip'");
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            

            // Train model
            ModelTrainer.TrainModel();
        }
    }
}
