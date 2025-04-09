#!/usr/bin/env python3
import os
import time
import torch
import numpy as np
import ray
from ray.util.queue import Queue
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from models.temporal_model import TemporalFusion
from models.marl_agent import HierarchicalAgent
from utils.replay_buffer import DualPriorityReplayBuffer
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
from mpi4py import MPI  # For multi-node communication

class DistributedTrainer:
    def __init__(self, config_path='configs/default.yaml'):
        self.config = self.load_config(config_path)
        self.setup_distributed()
        self.initialize_components()
        self.setup_ray()
        
    def load_config(self, config_path):
        """Load training configuration"""
        with open(config_path) as f:
            config = json.load(f)
        return config['training']

    def setup_distributed(self):
        """Initialize distributed training backend"""
        if 'OMPI_COMM_WORLD_SIZE' in os.environ:
            # MPI initialization for multi-node
            comm = MPI.COMM_WORLD
            self.world_size = comm.Get_size()
            self.rank = comm.Get_rank()
            os.environ['MASTER_ADDR'] = self.config['master_address']
            os.environ['MASTER_PORT'] = str(self.config['master_port'])
            dist.init_process_group(
                backend='nccl',
                world_size=self.world_size,
                rank=self.rank
            )
        else:
            # Single-node multi-GPU
            dist.init_process_group(backend='nccl')
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()

        torch.cuda.set_device(self.rank)
        self.device = torch.device(f'cuda:{self.rank}')

    def initialize_components(self):
        """Initialize models, optimizers, and replay buffer"""
        # Temporal Model
        self.temporal_model = TemporalFusion(
            input_dim=self.config['input_dim'],
            lstm_dim=self.config['lstm_dim'],
            mamba_dim=self.config['mamba_dim']
        ).to(self.device)
        self.temporal_model = DDP(self.temporal_model, device_ids=[self.rank])

        # Hierarchical Agents
        self.agents = {
            'leaf': HierarchicalAgent(
                input_dim=self.config['input_dim'],
                action_dim=self.config['action_dim'],
                agent_type='leaf'
            ).to(self.device),
            'rack': HierarchicalAgent(
                input_dim=self.config['input_dim'],
                action_dim=self.config['action_dim'],
                agent_type='rack'
            ).to(self.device),
            'cluster': HierarchicalAgent(
                input_dim=self.config['input_dim'],
                action_dim=self.config['action_dim'],
                agent_type='cluster'
            ).to(self.device)
        }
        self.agents = {k: DDP(v, device_ids=[self.rank]) for k, v in self.agents.items()}

        # Optimizers
        self.optimizers = {
            'temporal': torch.optim.Adam(
                self.temporal_model.parameters(),
                lr=self.config['lr_temporal'],
                weight_decay=1e-5
            ),
            'leaf': torch.optim.Adam(
                self.agents['leaf'].parameters(),
                lr=self.config['lr_leaf'],
                weight_decay=1e-5
            ),
            'rack': torch.optim.Adam(
                self.agents['rack'].parameters(),
                lr=self.config['lr_rack'],
                weight_decay=1e-5
            ),
            'cluster': torch.optim.Adam(
                self.agents['cluster'].parameters(),
                lr=self.config['lr_cluster'],
                weight_decay=1e-5
            )
        }

        # Schedulers
        self.schedulers = {
            name: CosineAnnealingLR(opt, T_max=self.config['epochs'])
            for name, opt in self.optimizers.items()
        }

        # Replay Buffer (shared across processes)
        if self.rank == 0:
            self.replay_buffer = DualPriorityReplayBuffer(
                capacity=int(self.config['replay_capacity']),
                alpha=self.config['alpha'],
                beta=self.config['beta_init']
            )
        else:
            self.replay_buffer = None

    def setup_ray(self):
        """Initialize Ray for distributed experience collection"""
        if self.rank == 0:
            ray.init(
                address=self.config['ray_address'],
                runtime_env={"working_dir": "."}
            )
            self.experience_queue = Queue(maxsize=10000)
            
            # Create 64 remote actors
            self.data_actors = [
                DataActor.remote(self.config) 
                for _ in range(self.config['num_actors'])
            ]
            
            # Start experience collection
            self.collection_tasks = [
                actor.collect_experience.remote(self.experience_queue)
                for actor in self.data_actors
            ]

    def train(self):
        """Main training loop"""
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            
            # Synchronize replay buffer across processes
            self.sync_replay_buffer()
            
            # Train temporal model
            temporal_loss = self.train_temporal_model()
            
            # Train MARL agents
            marl_losses = self.train_marl_agents()
            
            # Update priorities
            self.update_priorities()
            
            # Logging and checkpointing
            if self.rank == 0:
                self.log_metrics(epoch, temporal_loss, marl_losses)
                if epoch % self.config['checkpoint_freq'] == 0:
                    self.save_checkpoint(epoch)
            
            # Synchronize all processes
            dist.barrier()
            
            epoch_time = time.time() - epoch_start
            if self.rank == 0:
                print(f"Epoch {epoch} completed in {epoch_time:.2f}s")

    def train_temporal_model(self):
        """Train LSTM-Mamba hybrid model"""
        self.temporal_model.train()
        total_loss = 0
        
        # Create distributed sampler
        sampler = DistributedSampler(
            self.temporal_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )
        
        loader = torch.utils.data.DataLoader(
            self.temporal_dataset,
            batch_size=self.config['batch_size'],
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
        
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizers['temporal'].zero_grad()
            y_pred = self.temporal_model(x)
            loss = torch.nn.functional.mse_loss(y_pred, y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.temporal_model.parameters(),
                max_norm=5.0
            )
            
            self.optimizers['temporal'].step()
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                if self.rank == 0:
                    print(f"Temporal Batch {batch_idx}: Loss {loss.item()/self.world_size:.4f}")
        
        self.schedulers['temporal'].step()
        return total_loss / len(loader)

    def train_marl_agents(self):
        """Train hierarchical MARL agents"""
        losses = {agent_type: 0 for agent_type in self.agents}
        
        # Sample batch from replay buffer
        samples, indices, weights = self.replay_buffer.sample(
            self.config['batch_size'])
        
        # Convert to tensors
        states = torch.stack([s['state'] for s in samples]).to(self.device)
        actions = {k: torch.stack([s['actions'][k] for s in samples]).to(self.device)}
        rewards = {k: torch.stack([s['rewards'][k] for s in samples]).to(self.device)}
        next_states = torch.stack([s['next_state'] for s in samples]).to(self.device)
        dones = torch.stack([s['done'] for s in samples]).to(self.device)
        weights = torch.tensor(weights, device=self.device)
        
        # Train each agent type
        for agent_type, agent in self.agents.items():
            agent.train()
            self.optimizers[agent_type].zero_grad()
            
            # Current Q values
            q_values = agent(states)
            current_q = q_values.gather(1, actions[agent_type].unsqueeze(1)).squeeze(1)
            
            # Target Q values
            with torch.no_grad():
                next_q_values = agent(next_states)
                next_q = next_q_values.max(1)[0]
                target_q = rewards[agent_type] + (1 - dones.float()) * self.config['gamma'] * next_q
            
            # Compute loss
            td_error = current_q - target_q
            loss = (weights * td_error.pow(2)).mean()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                agent.parameters(),
                max_norm=5.0
            )
            
            self.optimizers[agent_type].step()
            losses[agent_type] = loss.item()
            
            # Update priorities
            new_priorities = abs(td_error.detach().cpu().numpy()) + 1e-6
            self.replay_buffer.update_priorities(indices, new_priorities)
        
        # Step schedulers
        for scheduler in self.schedulers.values():
            scheduler.step()
            
        return losses

    def sync_replay_buffer(self):
        """Synchronize replay buffer across processes"""
        if self.world_size <= 1:
            return
            
        if self.rank == 0:
            # Gather experiences from all processes
            all_experiences = []
            for _ in range(1, self.world_size):
                exp = self.experience_queue.get()
                all_experiences.extend(exp)
            
            # Add to shared buffer
            for exp in all_experiences:
                self.replay_buffer.add(
                    exp['experience'],
                    exp['td_error'],
                    exp['attention']
                )
            
            # Broadcast buffer size
            buffer_size = torch.tensor(len(self.replay_buffer), device=self.device)
            dist.broadcast(buffer_size, src=0)
        else:
            # Send experiences to rank 0
            local_experiences = []  # Collect your local experiences
            if local_experiences:
                self.experience_queue.put(local_experiences)
            
            # Receive buffer size
            buffer_size = torch.tensor(0, device=self.device)
            dist.broadcast(buffer_size, src=0)
            
            # Sync models if buffer was updated
            if buffer_size > 0:
                self.sync_models()

    def sync_models(self):
        """Synchronize model parameters across processes"""
        for model in [self.temporal_model] + list(self.agents.values()):
            for param in model.parameters():
                dist.broadcast(param.data, src=0)

    def update_priorities(self):
        """Update priorities in the replay buffer"""
        if self.rank != 0:
            return
            
        # Meta-learning for priority parameters
        new_alpha = self.config['alpha'] * np.exp(-0.0001 * self.epoch)
        new_beta = min(self.config['beta_init'] + 0.0002 * self.epoch, 1.0)
        
        self.replay_buffer.alpha = new_alpha
        self.replay_buffer.beta = new_beta

    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'temporal_model_state_dict': self.temporal_model.state_dict(),
            'leaf_agent_state_dict': self.agents['leaf'].state_dict(),
            'rack_agent_state_dict': self.agents['rack'].state_dict(),
            'cluster_agent_state_dict': self.agents['cluster'].state_dict(),
            'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()},
            'replay_buffer': self.replay_buffer.state_dict() if self.rank == 0 else None,
            'config': self.config
        }
        
        torch.save(checkpoint, f"checkpoints/checkpoint_epoch_{epoch}.pt")
        print(f"Saved checkpoint for epoch {epoch}")

    def log_metrics(self, epoch, temporal_loss, marl_losses):
        """Log training metrics"""
        metrics = {
            'epoch': epoch,
            'temporal_loss': temporal_loss,
            'marl_losses': marl_losses,
            'replay_buffer_size': len(self.replay_buffer) if self.rank == 0 else 0,
            'timestamp': time.time()
        }
        
        with open('training_logs.json', 'a') as f:
            json.dump(metrics, f)
            f.write('\n')

@ray.remote(num_gpus=0.25)
class DataActor:
    def __init__(self, config):
        self.config = config
        self.env = self.create_environment()
        self.local_buffer = []
        
    def create_environment(self):
        """Initialize cloud environment simulator"""
        # Implement your environment here
        pass
        
    def collect_experience(self, experience_queue):
        """Generate and send experiences to the central queue"""
        while True:
            state = self.env.reset()
            done = False
            
            while not done:
                # Get actions from current policy (would need model reference)
                actions = self.sample_actions(state)
                
                # Step environment
                next_state, rewards, done, info = self.env.step(actions)
                
                # Calculate TD-error and attention (simplified)
                td_error = np.random.random()  # Replace with actual TD-error
                attention = info.get('attention', 0.5)
                
                # Store experience
                experience = {
                    'state': state,
                    'actions': actions,
                    'rewards': rewards,
                    'next_state': next_state,
                    'done': done,
                    'td_error': td_error,
                    'attention': attention
                }
                
                self.local_buffer.append(experience)
                state = next_state
                
                # Send batch periodically
                if len(self.local_buffer) >= self.config['send_batch_size']:
                    experience_queue.put(self.local_buffer)
                    self.local_buffer = []
                    
            time.sleep(0.1)  # Prevent overwhelming the queue

if __name__ == "__main__":
    trainer = DistributedTrainer()
    trainer.train()