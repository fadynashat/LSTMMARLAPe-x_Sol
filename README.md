# LSTM-MARL-Ape-X: Scalable Resource Allocation for Cloud Computing

![Framework Architecture](docs/architecture.png)

A hybrid AI framework combining multi-scale temporal modeling with hierarchical multi-agent reinforcement learning for optimal cloud resource allocation.

## Key Innovations

| Component | Improvement Over SOTA | Technical Highlights |
|-----------|-----------------------|----------------------|
| **LSTM-Mamba Hybrid** | 41.3% lower MAE (5.23 vs 8.91) | • Bidirectional LSTM (128 units) <br> • Mamba SSM (64-dim state) <br> • Dynamic fusion gating |
| **Hierarchical MARL** | 72% fewer SLA violations | • 16-agent hierarchy <br> • Variance-regularized credit assignment <br> • Graph attention communication |
| **Ape-X Replay** | 3.2× faster convergence | • Dual prioritization (TD+attention) <br> • Meta-learned hyperparameters <br> • 60GB distributed buffer |
| **Production Framework** | 2.7-month ROI | • Kubernetes operator <br> • Carbon-aware scheduling <br> • 3s fault recovery |

## Installation

### Prerequisites
- NVIDIA GPU (A100 recommended)
- CUDA 11.7+
- Python 3.9+

```bash
# Clone repository
git clone https://github.com/fadynashat/LSTMMARLAPe-x_Sol.git
cd LSTMMARLAPe-x_Sol

#install conda
from url  https://www.anaconda.com/docs/main
direct link for windows 
https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Windows-x86_64.exe


# Create conda environment
conda create -n marl python=3.9
conda activate marl

# Install with PyTorch
pip install torch==2.0.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt

# Install MPI for multi-node training
conda install -c conda-forge mpi4py openmpi

#Dataset Preparation
#Google Cluster Trace:
cd data
python get_google_data.py
#Azure VM Workloads:
cd data
python get_azure_data.py

#bitbrains VM Workloads:
cd data
python get_bitbrains_data.py

Training
Single Node (8 GPUs)

torchrun --nnodes=1 --nproc_per_node=8  scripts/train.py  --config configs/default.yaml  --dataset data/google_processed  --log-dir logs/

Multi-Node Cluster


# On each node
mpirun -np 8 --hostfile hostfile python scripts/train.py --config configs/cluster.yaml -dataset /shared/storage/google_processed --log-dir /shared/logs/
Key Training Parameters:


# configs/default.yaml
training:
  epochs: 1000
  batch_size: 1024
  lr_temporal: 3e-4    # LSTM-Mamba learning rate
  lr_leaf: 1e-3        # Leaf agent LR
  lr_rack: 5e-4        # Rack agent LR  
  lr_cluster: 1e-4     # Cluster agent LR
  gamma: 0.95          # Discount factor
  alpha: 0.6           # Priority weight
  beta_init: 0.5       # Initial IS weight
  
replay_buffer:
  capacity: 60000000   # 60GB
  compression: zstd    # 3.8:1 compression ratio
Evaluation


python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --dataset data/azure_processed \
    --output results/ \
    --metrics sla_compliance energy_reduction decision_latency
