import numpy as np
import torch
from collections import deque
import zstandard as zstd

class DualPriorityReplayBuffer:
    def __init__(self, capacity=1e6, alpha=0.6, beta=0.5):
        self.capacity = int(capacity)
        self.alpha = alpha
        self.beta = beta
        self.buffer = deque(maxlen=self.capacity)
        self.priorities = deque(maxlen=self.capacity)
        self.compressor = zstd.ZstdCompressor(level=3)
        
    def add(self, experience, td_error, attention):
        priority = (abs(td_error) + self.alpha * attention) ** self.beta
        compressed = self.compressor.compress(pickle.dumps(experience))
        self.buffer.append(compressed)
        self.priorities.append(priority)
        
    def sample(self, batch_size):
        probs = np.array(self.priorities) ** self.beta
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [pickle.loads(self.compressor.decompress(self.buffer[i])) 
                  for i in indices]
        
        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        return samples, indices, weights
    
    def update_priorities(self, indices, new_priorities):
        for idx, priority in zip(indices, new_priorities):
            self.priorities[idx] = priority