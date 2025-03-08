using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LSTM_MARL_Ape_X_SOL
{
    /// <summary>
    /// Represents a single data point containing system performance metrics.
    /// </summary>
    public class DataPoint
    {
        /// <summary>
        /// The timestamp indicating when this data point was recorded.
        /// </summary>
        public DateTime Timestamp { get; set; }

        /// <summary>
        /// The unique identifier of the virtual machine (VM) associated with this data point.
        /// </summary>
        public string VmId { get; set; }

        /// <summary>
        /// The CPU utilization of the VM, typically represented as a percentage (e.g., 75.3%).
        /// </summary>
        public double CpuUsage { get; set; }

        /// <summary>
        /// The memory usage of the VM, either in megabytes (MB) or as a percentage.
        /// </summary>
        public double MemoryUsage { get; set; }

        /// <summary>
        /// The disk input/output (I/O) operations, usually measured in MB/s (megabytes per second) or IOPS (input/output operations per second).
        /// </summary>
        public double DiskIO { get; set; }

        /// <summary>
        /// The network traffic in terms of data sent and received, typically measured in Mbps (megabits per second).
        /// </summary>
        public double NetworkIO { get; set; }

        /// <summary>
        /// The number of incoming requests per second to the VM.
        /// This helps in analyzing the system load and determining if scaling is needed.
        /// </summary>
        public double RequestRate { get; set; }

        /// <summary>
        /// The average response time of the VM when handling requests, measured in milliseconds (ms).
        /// Ensuring this value remains low is crucial for maintaining SLA compliance.
        /// </summary>
        public double ResponseTime { get; set; }
    }

}
