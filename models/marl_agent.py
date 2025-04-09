import torch
import torch.nn as nn
import torch_geometric.nn as geom_nn

class HierarchicalAgent(nn.Module):
    def __init__(self, input_dim, action_dim, agent_type='leaf'):
        super().__init__()
        self.agent_type = agent_type
        
        # Temporal encoder
        self.temporal_enc = TemporalFusion(input_dim)
        
        # Graph attention communication
        self.gat = geom_nn.GATConv(input_dim, 64, heads=2)
        
        # Q-network
        q_input_dim = 128 if agent_type == 'leaf' else 192
        self.q_net = nn.Sequential(
            nn.Linear(q_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def communicate(self, node_features, edge_index):
        return self.gat(node_features, edge_index)
    
    def forward(self, x, neighbor_msgs=None):
        temporal_feat = self.temporal_enc(x)
        
        if neighbor_msgs is not None:
            combined = torch.cat([temporal_feat, neighbor_msgs], dim=-1)
        else:
            combined = temporal_feat
            
        return self.q_net(combined)