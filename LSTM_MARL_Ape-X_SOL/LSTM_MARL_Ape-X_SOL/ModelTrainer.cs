
using Microsoft.ML;

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
