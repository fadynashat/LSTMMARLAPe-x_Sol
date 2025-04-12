import os
import requests
import pandas as pd
import gzip
from tqdm import tqdm

class GoogleClusterData2011:
    def __init__(self):
        self.base_url = "https://storage.googleapis.com/clusterdata-2011-2"
        self.valid_files = {
            'machine_events': 'machine_events/part-00000-of-00001.csv.gz',
            'instance_events': 'instance_events/part-00000-of-00500.csv.gz',
            'instance_usage': 'instance_usage/part-00000-of-00500.csv.gz'
        }
        self.schema = {
            'machine_events': ['timestamp', 'machine_id', 'event_type', 'platform_id', 'cpu', 'memory'],
            'instance_events': ['timestamp', 'missing_info', 'job_id', 'task_index', 'machine_id', 'event_type',
                               'user', 'scheduling_class', 'priority', 'cpu_request', 'memory_request',
                               'disk_request', 'different_machine_constraint'],
            'instance_usage': ['start_time', 'end_time', 'job_id', 'task_index', 'machine_id', 'mean_cpu',
                              'canonical_mem', 'assigned_mem', 'unmapped_cache', 'total_cache',
                              'max_mem', 'mean_disk', 'mean_disk_space', 'max_cpu', 'max_disk',
                              'cpi', 'mai', 'sample_portion', 'aggregation_type', 'sampled_cpu']
        }
    
    def download_file(self, file_type):
        """Download a specific file type with validation"""
        if file_type not in self.valid_files:
            raise ValueError(f"Invalid file type. Choose from: {list(self.valid_files.keys())}")
        
        filename = self.valid_files[file_type]
        url = f"{self.base_url}/{filename}"
        local_path = f"clusterdata_2011/{os.path.basename(filename)}"
        
        os.makedirs("clusterdata_2011", exist_ok=True)
        
        try:
            print(f"Downloading {file_type}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            with open(local_path, 'wb') as f, tqdm(
                desc=os.path.basename(filename),
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            
            return local_path
            
        except Exception as e:
            print(f"Download failed: {str(e)}")
            if os.path.exists(local_path):
                os.remove(local_path)
            return None

    def load_data(self, file_type, sample_fraction=1.0):
        """Load and process data from downloaded files"""
        local_path = self.download_file(file_type)
        if not local_path:
            return None
            
        try:
            with gzip.open(local_path, 'rt') as f:
                # Read in chunks for large files
                chunks = pd.read_csv(f, header=None, names=self.schema[file_type],
                                    chunksize=100000)
                df = pd.concat(chunk for chunk in chunks)
                
                # Apply sampling if needed
                if sample_fraction < 1.0:
                    df = df.sample(frac=sample_fraction)
                
                # Convert timestamps to datetime
                if 'timestamp' in df.columns:
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='us', origin='unix')
                elif 'start_time' in df.columns:
                    df['start_datetime'] = pd.to_datetime(df['start_time'], unit='us', origin='unix')
                    df['end_datetime'] = pd.to_datetime(df['end_time'], unit='us', origin='unix')
                
                print(f"Loaded {len(df)} records from {file_type}")
                return df
                
        except Exception as e:
            print(f"Failed to process {file_type}: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    processor = GoogleClusterData2011()
    
    # Download and process machine events
    machine_df = processor.load_data('machine_events')
    if machine_df is not None:
        print(machine_df.head())
    
    # Download and process instance events (sampling 10%)
    instance_df = processor.load_data('instance_events', sample_fraction=0.1)
    if instance_df is not None:
        print(instance_df.head())
    
    # Download and process usage data
    usage_df = processor.load_data('instance_usage')
    if usage_df is not None:
        print(usage_df.head())