using LSTM_MARL_Ape_X_SOL;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

/// <summary>
/// A class responsible for preprocessing data, including loading, handling missing values, normalizing, and saving.
/// </summary>
public class DataPreprocessor
{
    /// <summary>
    /// Loads data from a CSV file into a list of DataPoint objects.
    /// </summary>
    /// <param name="filePath">The file path of the CSV file.</param>
    /// <returns>A list of DataPoint objects.</returns>
    public List<DataPoint> LoadData(string filePath)
    {
        var data = new List<DataPoint>();

        // Open the file for reading
        using (var reader = new StreamReader(filePath))
        {
            // Skip the first line (header)
            reader.ReadLine();

            // Read each line until the end of the file
            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine();

                // Skip empty lines
                if (string.IsNullOrWhiteSpace(line))
                    continue;

                var values = line.Split(',');

                // Ensure the row has exactly 8 columns before processing
                if (values.Length != 8) // Updated to match the number of properties
                    continue;

                // Parse the data into a DataPoint object
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

                // Add the parsed data point to the list
                data.Add(dataPoint);
            }
        }

        return data;
    }

    /// <summary>
    /// Safely parses a string to a double value.
    /// Returns NaN if parsing fails.
    /// </summary>
    /// <param name="value">The string representation of a numeric value.</param>
    /// <returns>A double value or NaN if invalid.</returns>
    private double TryParseDouble(string value)
    {
        if (double.TryParse(value, out double result))
            return result;
        return double.NaN; // Return NaN for missing or invalid values
    }

    /// <summary>
    /// Handles missing values by replacing them with the mean of the respective column.
    /// If no valid data exists for a column, it defaults to zero.
    /// </summary>
    /// <param name="data">The list of DataPoint objects to process.</param>
    public void HandleMissingValues(List<DataPoint> data)
    {
        // Compute the mean of each metric, excluding NaN values
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

        // Replace NaN values with the computed mean
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

    /// <summary>
    /// Normalizes data to a [0, 1] range by dividing each value by the maximum in its column.
    /// </summary>
    /// <param name="data">The list of DataPoint objects to normalize.</param>
    public void NormalizeData(List<DataPoint> data)
    {
        // Find the maximum values for each metric
        var cpuMax = data.Max(d => d.CpuUsage);
        var memoryMax = data.Max(d => d.MemoryUsage);
        var diskMax = data.Max(d => d.DiskIO);
        var networkMax = data.Max(d => d.NetworkIO);
        var requestMax = data.Max(d => d.RequestRate);
        var responseMax = data.Max(d => d.ResponseTime);

        // Avoid division by zero by setting zero max values to 1
        cpuMax = cpuMax == 0 ? 1 : cpuMax;
        memoryMax = memoryMax == 0 ? 1 : memoryMax;
        diskMax = diskMax == 0 ? 1 : diskMax;
        networkMax = networkMax == 0 ? 1 : networkMax;
        requestMax = requestMax == 0 ? 1 : requestMax;
        responseMax = responseMax == 0 ? 1 : responseMax;

        // Normalize each data point
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

    /// <summary>
    /// Splits the dataset into training and testing sets based on a given ratio.
    /// </summary>
    /// <param name="data">The list of DataPoint objects.</param>
    /// <param name="trainRatio">The proportion of data to be used for training (default is 80%).</param>
    /// <returns>A tuple containing the training and test sets.</returns>
    public (List<DataPoint> train, List<DataPoint> test) SplitData(List<DataPoint> data, double trainRatio = 0.8)
    {
        var trainSize = (int)(data.Count * trainRatio);

        // Take the first 'trainSize' elements for training
        var trainData = data.Take(trainSize).ToList();

        // The remaining elements are used for testing
        var testData = data.Skip(trainSize).ToList();

        return (trainData, testData);
    }

    /// <summary>
    /// Saves the processed data to a CSV file.
    /// </summary>
    /// <param name="filePath">The file path where the data should be saved.</param>
    /// <param name="data">The list of DataPoint objects to save.</param>
    public void SaveData(string filePath, List<DataPoint> data)
    {
        using (var writer = new StreamWriter(filePath))
        {
            // Write CSV header
            writer.WriteLine("Timestamp,VmId,CpuUsage,MemoryUsage,DiskIO,NetworkIO,RequestRate,ResponseTime");

            // Write each data point as a CSV row
            foreach (var point in data)
            {
                writer.WriteLine($"{point.Timestamp:yyyy-MM-dd HH:mm:ss},{point.VmId},{point.CpuUsage},{point.MemoryUsage},{point.DiskIO},{point.NetworkIO},{point.RequestRate},{point.ResponseTime}");
            }
        }
    }
}
