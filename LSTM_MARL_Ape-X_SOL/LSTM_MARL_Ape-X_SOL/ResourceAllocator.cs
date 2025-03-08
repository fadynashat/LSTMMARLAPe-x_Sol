
// Resource Allocation System
using LSTM_MARL_Ape_X_SOL;

public class ResourceAllocator
    {
        // Dictionary to track available resources for different types (CPU, Memory, etc.)
        private Dictionary<ResourceType, double> availableResources;

        // Constructor initializes the available resources with predefined values.
        public ResourceAllocator()
        {
            availableResources = new Dictionary<ResourceType, double>
        {
            { ResourceType.CPU, 100.0 },       // 100% total CPU available
            { ResourceType.Memory, 1024.0 },   // 1024 MB of available memory
            { ResourceType.Storage, 5000.0 },  // 5000 MB of storage
            { ResourceType.DiskIO, 200.0 },    // 200 MB/s disk input/output speed
            { ResourceType.NetworkIO, 1000.0 } // 1000 Mbps network bandwidth
        };
        }

        // Allocates a requested resource if sufficient amount is available.
        public bool AllocateResource(ResourceRequest request)
        {
            // Check if the requested amount of the resource is available
            if (availableResources[request.Type] >= request.Amount)
            {
                // Deduct the allocated amount from the available resources
                availableResources[request.Type] -= request.Amount;
                Console.WriteLine($"Allocated {request.Amount} of {request.Type}");
                return true; // Allocation successful
            }
            else
            {
                Console.WriteLine($"Insufficient {request.Type} to allocate {request.Amount}");
                return false; // Allocation failed due to insufficient resources
            }
        }

        // Releases a previously allocated resource, making it available again.
        public void ReleaseResource(ResourceRequest request)
        {
            // Add back the released amount to the available resources
            availableResources[request.Type] += request.Amount;
            Console.WriteLine($"Released {request.Amount} of {request.Type}");
        }

        // Prints the current availability of all resources in the system.
        public void PrintAvailableResources()
        {
            Console.WriteLine("Available Resources:");
            foreach (var resource in availableResources)
                Console.WriteLine($"{resource.Key}: {resource.Value}");
        }
    }


