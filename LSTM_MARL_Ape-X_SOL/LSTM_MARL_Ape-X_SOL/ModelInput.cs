using Microsoft.ML.Data;

public class ModelInput
{
    // Attribute that specifies which column in the CSV file should be loaded into this property.
    // The index starts from 0, so column 2 in the CSV corresponds to "CpuUsage".
    [LoadColumn(2)]
    public float CpuUsage { get; set; }

    // Column 3 in the CSV corresponds to "MemoryUsage".
    [LoadColumn(3)]
    public float MemoryUsage { get; set; }

    // Column 4 in the CSV corresponds to "DiskIO".
    [LoadColumn(4)]
    public float DiskIO { get; set; }

    // Column 5 in the CSV corresponds to "NetworkIO".
    [LoadColumn(5)]
    public float NetworkIO { get; set; }

    // Column 6 in the CSV corresponds to "RequestRate".
    [LoadColumn(6)]
    public float RequestRate { get; set; }

    // Column 7 in the CSV corresponds to "ResponseTime".
    // This is the **target variable** (what the model will try to predict).
    [LoadColumn(7)]
    public float ResponseTime { get; set; }
}
