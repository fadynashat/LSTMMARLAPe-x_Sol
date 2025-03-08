using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Data;
using DataPreprocessing;

namespace CloudPerformanceModel
{
    public class ModelTrainer
    {
        private static string dataPath = "preprocessed_data.csv";
        private static string modelPath = "trained_model.zip";

        public static void TrainModel()
        {
            var mlContext = new MLContext();

            // Load data
            IDataView dataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                path: dataPath, separatorChar: ',', hasHeader: true);

            // Split data into training and testing sets (80% train, 20% test)
            var trainTestSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            var trainData = trainTestSplit.TrainSet;
            var testData = trainTestSplit.TestSet;

            // Define model pipeline
            var pipeline = mlContext.Transforms.Concatenate("Features", nameof(ModelInput.CpuUsage),
                                                                  nameof(ModelInput.MemoryUsage),
                                                                  nameof(ModelInput.DiskIO),
                                                                  nameof(ModelInput.NetworkIO),
                                                                  nameof(ModelInput.RequestRate))
                .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: nameof(ModelInput.ResponseTime), featureColumnName: "Features"));

            // Train the model
            var model = pipeline.Fit(trainData);

            // Evaluate the model
            EvaluateModel(mlContext, model, testData);

            // Save the trained model
            mlContext.Model.Save(model, trainData.Schema, modelPath);
            Console.WriteLine($"✅ Model training complete. Saved as {modelPath}");
        }

        private static void EvaluateModel(MLContext mlContext, ITransformer model, IDataView testData)
        {
            Console.WriteLine("\n🔍 Evaluating Model Performance...");

            var predictions = model.Transform(testData);
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: nameof(ModelInput.ResponseTime));

            Console.WriteLine($"📊 R² Score: {metrics.RSquared:F4}");
            Console.WriteLine($"📉 Mean Absolute Error (MAE): {metrics.MeanAbsoluteError:F4}");
            Console.WriteLine($"📉 Root Mean Squared Error (RMSE): {metrics.RootMeanSquaredError:F4}");
        }

        static void Main(string[] args)
        {
            Console.WriteLine("🚀 Starting Model Training...");

            // Call the TrainModel method
            ModelTrainer.TrainModel();

            Console.WriteLine("✅ Training and Evaluation Complete!");
            Console.ReadLine();
        }
    }
}