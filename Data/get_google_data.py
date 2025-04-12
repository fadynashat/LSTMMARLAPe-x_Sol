import requests
import os
from urllib.parse import urljoin
import xml.etree.ElementTree as ET
import hashlib
import pandas as pd

class GoogleClusterDataDownloader:
    def __init__(self):
        self.base_url = "https://commondatastorage.googleapis.com/clusterdata-2011-2/"
        self.schema = {
            # ... (keep your existing schema definition) ...
        }
        self.file_metadata = {}
        self.download_dir = "google_cluster_data"
        os.makedirs(self.download_dir, exist_ok=True)

    def fetch_all_bucket_listings(self):
        """Fetch all bucket listings using pagination"""
        marker = ""
        total_files = 0
        
        while True:
            url = urljoin(self.base_url, f"?max-keys=1000&marker={marker}")
            try:
                response = requests.get(url)
                response.raise_for_status()
                root = ET.fromstring(response.content)
                ns = {'s3': 'http://doc.s3.amazonaws.com/2006-03-01'}
                
                # Process contents
                for content in root.findall('s3:Contents', ns):
                    key = content.find('s3:Key', ns).text
                    etag = content.find('s3:ETag', ns).text.strip('"')
                    size = int(content.find('s3:Size', ns).text)
                    
                    self.file_metadata[key] = {
                        'etag': etag,
                        'size': size,
                        'last_modified': content.find('s3:LastModified', ns).text
                    }
                    total_files += 1
                
                # Check if we need to paginate
                is_truncated = root.find('s3:IsTruncated', ns).text
                if is_truncated.lower() != 'true':
                    break
                    
                # Get next marker
                next_marker = root.find('s3:NextMarker', ns)
                if next_marker is not None:
                    marker = next_marker.text
                else:
                    # If no NextMarker, use last key
                    last_key = root.findall('s3:Contents', ns)[-1].find('s3:Key', ns).text
                    marker = last_key
                    
            except Exception as e:
                print(f"Error fetching bucket listing: {e}")
                return False
        
        print(f"Found metadata for {total_files} files")
        return True

    def download_file(self, filename):
        """Download a single file with verification"""
        if filename not in self.file_metadata:
            print(f"File {filename} not found in metadata")
            return False
            
        file_url = urljoin(self.base_url, filename)
        file_path = os.path.join(self.download_dir, os.path.basename(filename))
        
        # Create subdirectories if needed
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Check existing file
        if os.path.exists(file_path):
            if self.verify_file(file_path, filename):
                print(f"File {filename} already exists and is valid")
                return True
            print(f"Existing file {filename} is invalid, re-downloading")
            os.remove(file_path)
        
        print(f"Downloading {filename}...")
        try:
            with requests.get(file_url, stream=True) as r:
                r.raise_for_status()
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            if not self.verify_file(file_path, filename):
                raise Exception("File verification failed")
                
            print(f"Successfully downloaded {filename}")
            return True
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            if os.path.exists(file_path):
                os.remove(file_path)
            return False

    def verify_file(self, filepath, filename):
        """Verify file using ETag and size"""
        meta = self.file_metadata.get(filename)
        if not meta:
            return False
            
        # Check size first (faster than hash)
        if os.path.getsize(filepath) != meta['size']:
            return False
            
        # Check ETag (MD5 hash)
        md5 = hashlib.md5()
        with open(filepath, 'rb') as f:
            while True:
                data = f.read(65536)
                if not data:
                    break
                md5.update(data)
        return md5.hexdigest() == meta['etag']

    def download_by_type(self, data_type):
        """Download all files for a specific data type"""
        if not self.file_metadata:
            if not self.fetch_all_bucket_listings():
                return False
                
        prefix = f"{data_type}/"
        matching_files = [k for k in self.file_metadata if k.startswith(prefix)]
        
        if not matching_files:
            print(f"No files found for {data_type}")
            return False
            
        print(f"Found {len(matching_files)} files for {data_type}")
        success = True
        for filename in sorted(matching_files):
            if not self.download_file(filename):
                success = False
                
        return success

    def load_data(self, data_type):
        """Load and combine all files for a data type"""
        if data_type not in self.schema:
            print(f"Unknown data type: {data_type}")
            return None
            
        prefix = f"{data_type}/"
        matching_files = [k for k in self.file_metadata if k.startswith(prefix)]
        
        if not matching_files:
            print(f"No files found for {data_type}")
            return None
            
        dfs = []
        for filename in sorted(matching_files):
            local_path = os.path.join(self.download_dir, os.path.basename(filename))
            
            if not os.path.exists(local_path):
                print(f"File not found locally: {filename}")
                if not self.download_file(filename):
                    continue
                    
            try:
                df = pd.read_csv(
                    local_path,
                    compression='gzip' if filename.endswith('.gz') else None,
                    header=None,
                    names=self.schema[data_type]['columns']
                )
                dfs.append(df)
                print(f"Loaded {len(df)} records from {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                
        if not dfs:
            return None
            
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Combined {len(combined_df)} records for {data_type}")
        return combined_df

# Example usage
if __name__ == "__main__":
    downloader = GoogleClusterDataDownloader()
    
    # Fetch all metadata (this may take a while)
    downloader.fetch_all_bucket_listings()
    
    # Download all task_events files
    downloader.download_by_type("task_events")
    
    # Load and combine all task events
    task_events_df = downloader.load_data("task_events")
    if task_events_df is not None:
        print(task_events_df.head())