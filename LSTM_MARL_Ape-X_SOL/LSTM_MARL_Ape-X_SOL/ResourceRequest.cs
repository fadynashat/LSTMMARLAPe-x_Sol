// Represents a request for allocating a specific type and amount of resource.
using LSTM_MARL_Ape_X_SOL;

public class ResourceRequest
{
    // The type of resource being requested (e.g., CPU, Memory, Disk, Network).
    public ResourceType Type { get; set; }

    // The amount of the resource requested (e.g., 2.5 CPU cores, 8GB RAM).
    public double Amount { get; set; }
}
