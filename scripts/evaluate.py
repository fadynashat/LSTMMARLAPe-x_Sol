#!/usr/bin/env python3
import torch
import numpy as np
from datetime import datetime
from models.temporal_model import TemporalFusion
from models.marl_agent import HierarchicalAgent
from utils.replay_buffer import DualPriorityReplayBuffer
import matplotlib.pyplot as plt
import json
import os
from scipy.stats import ttest_ind

class Evaluator:
    def __init__(self, config_path='configs/default.yaml'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_config(config_path)
        self.setup_models()
        self.metrics = {
            'sla_compliance': [],
            'energy_reduction': [],
            'decision_latency': [],
            'training_steps': [],
            'periodicity_detection': []
        }
        
    def load_config(self, config_path):
        """Load evaluation configuration"""
        with open(config_path) as f:
            self.config = json.load(f)['evaluation']
        
    def setup_models(self):
        """Initialize models with pretrained weights"""
        self.temporal_model = TemporalFusion(
            input_dim=self.config['input_dim'],
            lstm_dim=self.config['lstm_dim'],
            mamba_dim=self.config['mamba_dim']
        ).to(self.device)
        
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
        
        # Load pretrained weights
        self.load_model_weights()
        
    def load_model_weights(self):
        """Load pretrained model weights"""
        try:
            checkpoint = torch.load(self.config['model_checkpoint'])
            self.temporal_model.load_state_dict(checkpoint['temporal_model'])
            for agent_type in self.agents:
                self.agents[agent_type].load_state_dict(
                    checkpoint[f'{agent_type}_agent'])
            print("Successfully loaded pretrained weights")
        except Exception as e:
            print(f"Error loading weights: {e}")
            exit(1)
            
    def evaluate_temporal_model(self, test_loader):
        """Evaluate LSTM-Mamba hybrid performance"""
        self.temporal_model.eval()
        total_mae, total_dtw = 0, 0
        latencies = []
        
        with torch.no_grad():
            for batch in test_loader:
                x, y_true = batch
                x, y_true = x.to(self.device), y_true.to(self.device)
                
                # Measure inference latency
                start_time = datetime.now()
                y_pred = self.temporal_model(x)
                latency = (datetime.now() - start_time).total_seconds() * 1000  # ms
                latencies.append(latency)
                
                # Calculate metrics
                mae = torch.mean(torch.abs(y_pred - y_true)).item()
                dtw = self.dynamic_time_warping(y_pred, y_true)
                total_mae += mae
                total_dtw += dtw
                
        avg_mae = total_mae / len(test_loader)
        avg_dtw = total_dtw / len(test_loader)
        avg_latency = np.mean(latencies)
        p99_latency = np.percentile(latencies, 99)
        
        self.metrics['temporal_mae'] = avg_mae
        self.metrics['temporal_dtw'] = avg_dtw
        self.metrics['inference_latency'] = {
            'mean': avg_latency,
            'p99': p99_latency
        }
        
        print(f"Temporal Model Evaluation:")
        print(f"- MAE: {avg_mae:.4f}")
        print(f"- DTW: {avg_dtw:.4f}")
        print(f"- Latency: {avg_latency:.2f}ms (p99: {p99_latency:.2f}ms)")
        
    def evaluate_marl_system(self, env, num_episodes=100):
        """Evaluate hierarchical MARL performance"""
        sla_violations = []
        decision_latencies = []
        energy_consumptions = []
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_sla = 0
            episode_energy = 0
            episode_steps = 0
            
            while not env.done:
                # Measure decision latency
                start_time = datetime.now()
                actions = {}
                for agent_type, agent in self.agents.items():
                    agent_input = self.prepare_agent_input(state, agent_type)
                    actions[agent_type] = agent(agent_input)
                latency = (datetime.now() - start_time).total_seconds() * 1000
                decision_latencies.append(latency)
                
                # Execute actions
                next_state, reward, done, info = env.step(actions)
                
                # Accumulate metrics
                episode_sla += info['sla_violation']
                episode_energy += info['energy_consumption']
                episode_steps += 1
                state = next_state
                
            # Normalize metrics
            sla_violations.append(episode_sla / episode_steps)
            energy_consumptions.append(episode_energy / episode_steps)
            
        # Calculate statistics
        avg_sla = np.mean(sla_violations) * 100  # percentage
        avg_energy = np.mean(energy_consumptions)
        avg_latency = np.mean(decision_latencies)
        p99_latency = np.percentile(decision_latencies, 99)
        
        self.metrics['sla_compliance'] = 100 - avg_sla
        self.metrics['energy_reduction'] = self.calculate_energy_reduction(avg_energy)
        self.metrics['decision_latency'] = {
            'mean': avg_latency,
            'p99': p99_latency
        }
        
        print("\nMARL System Evaluation:")
        print(f"- SLA Compliance: {100 - avg_sla:.2f}%")
        print(f"- Energy Reduction: {self.metrics['energy_reduction']:.2f}%")
        print(f"- Decision Latency: {avg_latency:.2f}ms (p99: {p99_latency:.2f}ms)")
        
    def calculate_energy_reduction(self, current_energy):
        """Calculate % reduction from baseline"""
        baseline = self.config['energy_baseline']
        return ((baseline - current_energy) / baseline) * 100
        
    def dynamic_time_warping(self, pred, true):
        """Calculate Dynamic Time Warping distance"""
        # Implementation of DTW algorithm
        n, m = pred.shape[0], true.shape[0]
        dtw_matrix = np.zeros((n+1, m+1))
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = torch.abs(pred[i-1] - true[j-1]).sum()
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],    # insertion
                    dtw_matrix[i, j-1],    # deletion
                    dtw_matrix[i-1, j-1]   # match
                )
                
        return dtw_matrix[n, m] / (n + m)
        
    def statistical_significance(self, baseline_metrics):
        """Perform t-tests against baseline metrics"""
        print("\nStatistical Significance Testing:")
        
        comparisons = [
            ('sla_compliance', 'higher'),
            ('energy_reduction', 'higher'),
            ('decision_latency.mean', 'lower'),
            ('temporal_mae', 'lower')
        ]
        
        for metric, direction in comparisons:
            our_data = self.flatten_metric(self.metrics, metric)
            baseline_data = self.flatten_metric(baseline_metrics, metric)
            
            t_stat, p_value = ttest_ind(our_data, baseline_data, 
                                       equal_var=False)
            
            print(f"{metric}: p-value = {p_value:.4f}", end=" ")
            if direction == 'higher':
                significant = (p_value < 0.01) and (np.mean(our_data) > np.mean(baseline_data))
            else:
                significant = (p_value < 0.01) and (np.mean(our_data) < np.mean(baseline_data))
                
            print("(Significant)" if significant else "(Not Significant)")
            
    def flatten_metric(self, metrics, path):
        """Extract nested metric values"""
        keys = path.split('.')
        value = metrics
        for key in keys:
            value = value[key]
        return np.array([value])  # Wrap single values for t-test
        
    def save_results(self, filename='evaluation_results.json'):
        """Save evaluation metrics to file"""
        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"\nResults saved to {filename}")
        
    def plot_metrics(self, baseline_metrics=None):
        """Generate comparative performance plots"""
        metrics_to_plot = [
            ('sla_compliance', 'SLA Compliance (%)', 'higher'),
            ('energy_reduction', 'Energy Reduction (%)', 'higher'),
            ('decision_latency.mean', 'Decision Latency (ms)', 'lower'),
            ('temporal_mae', 'Temporal MAE', 'lower')
        ]
        
        plt.figure(figsize=(15, 10))
        
        for i, (metric, title, _) in enumerate(metrics_to_plot, 1):
            plt.subplot(2, 2, i)
            
            our_value = self.get_nested_value(self.metrics, metric)
            plt.bar(['Our Method'], [our_value], color='orange')
            
            if baseline_metrics:
                baseline_value = self.get_nested_value(baseline_metrics, metric)
                plt.bar(['Baseline'], [baseline_value], color='blue')
                
            plt.title(title)
            plt.grid(True, linestyle='--', alpha=0.6)
            
        plt.tight_layout()
        plt.savefig('evaluation_metrics.png')
        print("Saved metrics plot to evaluation_metrics.png")
        
    def get_nested_value(self, d, path):
        keys = path.split('.')
        for key in keys:
            d = d[key]
        return d

if __name__ == "__main__":
    evaluator = Evaluator()
    
    # Load your test data and environment
    # test_loader = ...  # PyTorch DataLoader for temporal evaluation
    # test_env = ...     # Your cloud environment simulator
    
    # Run evaluations
    # evaluator.evaluate_temporal_model(test_loader)
    # evaluator.evaluate_marl_system(test_env)
    
    # Compare with baseline (example)
    baseline = {
        'sla_compliance': 88.1,
        'energy_reduction': 18.1,
        'decision_latency': {'mean': 41.0},
        'temporal_mae': 6.89
    }
    
    evaluator.statistical_significance(baseline)
    evaluator.plot_metrics(baseline)
    evaluator.save_results()