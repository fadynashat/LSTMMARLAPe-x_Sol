using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace DataPreprocessing
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

    class Program
    {
        static void Main(string[] args)
        {
            // Initialize preprocessor
            var preprocessor = new DataPreprocessor();

            // Load data from CSV
            var data = preprocessor.LoadData("synthetic_cloud_dataset.csv");

            // Handle missing values
            preprocessor.HandleMissingValues(data);

            // Normalize data
            preprocessor.NormalizeData(data);

            // Split data into training and test sets
            var (trainData, testData) = preprocessor.SplitData(data);

            // Save preprocessed data
            preprocessor.SaveData("preprocessed_data.csv", data);

            // Print preprocessed data
            Console.WriteLine("Preprocessed data saved to 'preprocessed_data.csv'.");

             }
    }
}
