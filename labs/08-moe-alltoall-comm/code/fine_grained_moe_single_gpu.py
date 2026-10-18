"""
Fine-Grained Expert Pool Demo - Single GPU Version
For testing on single GPU before multi-GPU deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from dataclasses import dataclass
from typing import List, Tuple

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class MoEConfig:
    """DeepSeek-style MoE Configuration"""
    d_model: int = 2048           # Model dimension (reduced for demo)
    d_expert: int = 1024          # Expert FFN intermediate dimension
    n_shared_experts: int = 2     # Number of shared experts
    n_routed_experts: int = 32    # Number of routed experts (fine-grained)
    n_activated: int = 4          # Number of activated experts per token
    top_k: int = 4                # Top-k routing
    aux_loss_free: bool = True    # Use auxiliary-loss-free balancing
    bias_update_speed: float = 0.001


# ============================================================================
# Fine-Grained Expert
# ============================================================================

class FineGrainedExpert(nn.Module):
    """Single fine-grained expert with SwiGLU activation"""
    
    def __init__(self, d_model: int, d_expert: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_expert, bias=False)
        self.up_proj = nn.Linear(d_model, d_expert, bias=False)
        self.down_proj = nn.Linear(d_expert, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ============================================================================
# DeepSeek MoE Layer
# ============================================================================

class DeepSeekMoELayer(nn.Module):
    """
    DeepSeek-style MoE Layer with:
    1. Shared experts (always active)
    2. Fine-grained routed experts
    3. Auxiliary-loss-free load balancing
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # Shared experts (always active)
        self.shared_experts = nn.ModuleList([
            FineGrainedExpert(config.d_model, config.d_expert)
            for _ in range(config.n_shared_experts)
        ])
        
        # Routed experts (fine-grained)
        self.routed_experts = nn.ModuleList([
            FineGrainedExpert(config.d_model, config.d_expert)
            for _ in range(config.n_routed_experts)
        ])
        
        # Router
        self.router = nn.Linear(config.d_model, config.n_routed_experts, bias=False)
        
        # Auxiliary-loss-free balancing
        if config.aux_loss_free:
            self.expert_bias = nn.Parameter(torch.zeros(config.n_routed_experts))
        
        # Statistics
        self.expert_load_counts = torch.zeros(config.n_routed_experts)
        self.n_forward_calls = 0
    
    def _compute_routing(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute routing scores and select top-k experts"""
        
        # Router logits
        router_logits = self.router(x)  # [batch, seq, n_routed_experts]
        
        # Sigmoid affinity (DeepSeek-V3 style)
        affinity = torch.sigmoid(router_logits)
        
        # Add bias for load balancing
        if self.config.aux_loss_free:
            routing_scores = affinity + self.expert_bias
        else:
            routing_scores = affinity
        
        # Top-k selection
        top_k_scores, top_k_indices = torch.topk(
            routing_scores, self.config.top_k, dim=-1
        )
        
        # Normalize gating values
        top_k_affinity = torch.gather(affinity, -1, top_k_indices)
        routing_weights = top_k_affinity / top_k_affinity.sum(dim=-1, keepdim=True)
        
        return routing_weights, top_k_indices
    
    def _update_statistics(self, selected_experts: torch.Tensor):
        """Update expert load statistics"""
        with torch.no_grad():
            for i in range(self.config.n_routed_experts):
                count = (selected_experts == i).sum().item()
                self.expert_load_counts[i] += count
        self.n_forward_calls += 1
    
    def _update_bias(self):
        """Update expert bias for load balancing"""
        if not self.config.aux_loss_free:
            return
        
        avg_load = self.expert_load_counts.mean()
        with torch.no_grad():
            for i in range(self.config.n_routed_experts):
                if self.expert_load_counts[i] > avg_load:
                    self.expert_bias[i] -= self.config.bias_update_speed
                else:
                    self.expert_bias[i] += self.config.bias_update_speed
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Step 1: Compute routing
        routing_weights, selected_experts = self._compute_routing(x)
        
        # Step 2: Shared expert computation
        shared_output = torch.zeros_like(x)
        for expert in self.shared_experts:
            shared_output += expert(x)
        shared_output /= len(self.shared_experts)
        
        # Step 3: Routed expert computation (simplified loop)
        routed_output = torch.zeros_like(x)
        
        # Flatten for processing
        x_flat = x.reshape(-1, d_model)  # [batch*seq, d_model]
        routing_weights_flat = routing_weights.reshape(-1, self.config.top_k)
        selected_experts_flat = selected_experts.reshape(-1, self.config.top_k)
        routed_output_flat = routed_output.reshape(-1, d_model)
        
        # Process each token-expert pair
        for i in range(x_flat.shape[0]):
            for k in range(self.config.top_k):
                expert_idx = selected_experts_flat[i, k].item()
                weight = routing_weights_flat[i, k].item()
                # Reshape for expert: [1, 1, d_model]
                token_input = x_flat[i].unsqueeze(0).unsqueeze(0)
                expert_out = self.routed_experts[expert_idx](token_input)
                routed_output_flat[i] += weight * expert_out.squeeze(0).squeeze(0)
        
        # Reshape back
        routed_output = routed_output_flat.reshape(batch_size, seq_len, d_model)
        
        # Step 4: Combine
        output = shared_output + routed_output
        
        # Step 5: Update statistics
        self._update_statistics(selected_experts)
        if self.n_forward_calls % 100 == 0:
            self._update_bias()
        
        return output
    
    def get_load_balance_stats(self) -> dict:
        """Get load balancing statistics"""
        if self.n_forward_calls == 0:
            return {}
        
        avg_load = self.expert_load_counts.mean().item()
        max_load = self.expert_load_counts.max().item()
        min_load = self.expert_load_counts.min().item()
        
        # Coefficient of variation (lower is better)
        cv = self.expert_load_counts.std().item() / avg_load if avg_load > 0 else 0
        
        return {
            'avg_load': avg_load,
            'max_load': max_load,
            'min_load': min_load,
            'cv': cv,  # Coefficient of variation
            'imbalance_ratio': max_load / min_load if min_load > 0 else float('inf')
        }


# ============================================================================
# Model with MoE
# ============================================================================

class DeepSeekMoEModel(nn.Module):
    """Simple model with multiple MoE layers"""
    
    def __init__(self, config: MoEConfig, n_layers: int = 4, vocab_size: int = 10000):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        
        self.layers = nn.ModuleList([
            DeepSeekMoELayer(config)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.RMSNorm(config.d_model)
        self.output = nn.Linear(config.d_model, vocab_size, bias=False)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = x + layer(x)
        
        x = self.norm(x)
        return self.output(x)
    
    def get_all_stats(self) -> List[dict]:
        """Get load balance stats from all MoE layers"""
        return [layer.get_load_balance_stats() for layer in self.layers]


# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_inference(model: nn.Module, config: MoEConfig, device: torch.device):
    """Benchmark inference latency"""
    
    print("\n" + "=" * 80)
    print("Inference Benchmark")
    print("=" * 80)
    
    model.eval()
    seq_lengths = [64, 128, 256, 512]
    batch_size = 1
    
    results = []
    
    for seq_len in seq_lengths:
        input_ids = torch.randint(0, 10000, (batch_size, seq_len), device=device)
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(input_ids)
        torch.cuda.synchronize()
        
        # Benchmark
        latencies = []
        for _ in range(20):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(input_ids)
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        
        mean_lat = sum(latencies) / len(latencies)
        std_lat = (sum((l - mean_lat)**2 for l in latencies) / len(latencies)) ** 0.5
        
        results.append({
            'seq_len': seq_len,
            'mean_ms': mean_lat,
            'std_ms': std_lat
        })
        
        print(f"  Seq Len {seq_len:>4}: {mean_lat:>8.2f} ms (±{std_lat:.2f})")
    
    return results


def benchmark_expert_counts(device: torch.device):
    """Benchmark with different expert counts"""
    
    print("\n" + "=" * 80)
    print("Expert Count Scaling Benchmark")
    print("=" * 80)
    
    expert_counts = [8, 16, 32, 64, 128]
    seq_len = 256
    batch_size = 1
    d_model = 2048
    
    results = []
    
    for n_experts in expert_counts:
        config = MoEConfig(
            d_model=d_model,
            d_expert=d_model // 2,
            n_shared_experts=2,
            n_routed_experts=n_experts,
            n_activated=min(4, n_experts),
            top_k=min(4, n_experts)
        )
        
        layer = DeepSeekMoELayer(config).to(device).half()
        x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float16)
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = layer(x)
        torch.cuda.synchronize()
        
        # Benchmark
        latencies = []
        for _ in range(10):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = layer(x)
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        
        mean_lat = sum(latencies) / len(latencies)
        n_params = sum(p.numel() for p in layer.parameters())
        
        results.append({
            'n_experts': n_experts,
            'mean_ms': mean_lat,
            'n_params_m': n_params / 1e6
        })
        
        print(f"  Experts {n_experts:>3}: {mean_lat:>8.2f} ms, {n_params/1e6:>6.1f}M params")
        
        del layer
        torch.cuda.empty_cache()
    
    return results


def benchmark_load_balancing(config: MoEConfig, device: torch.device):
    """Benchmark load balancing"""
    
    print("\n" + "=" * 80)
    print("Load Balancing Benchmark")
    print("=" * 80)
    
    layer = DeepSeekMoELayer(config).to(device)
    seq_len = 512
    batch_size = 4
    n_steps = 50
    
    # Run multiple forward passes
    for step in range(n_steps):
        x = torch.randn(batch_size, seq_len, config.d_model, device=device)
        _ = layer(x)
    
    # Get statistics
    stats = layer.get_load_balance_stats()
    
    print(f"\n  After {n_steps} steps:")
    print(f"    Average load: {stats['avg_load']:.1f}")
    print(f"    Max load: {stats['max_load']:.1f}")
    print(f"    Min load: {stats['min_load']:.1f}")
    print(f"    Imbalance ratio: {stats['imbalance_ratio']:.2f}")
    print(f"    Coefficient of variation: {stats['cv']:.4f}")
    
    # Print expert load distribution
    print(f"\n  Expert Load Distribution:")
    load_counts = layer.expert_load_counts.numpy()
    for i in range(0, len(load_counts), 8):
        row = load_counts[i:i+8]
        print(f"    Experts {i:>3}-{i+7:>3}: [{', '.join(f'{x:>6.0f}' for x in row)}]")
    
    return stats


# ============================================================================
# Main Demo
# ============================================================================

def main():
    """Main demo function"""
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 80)
    print("DeepSeek Fine-Grained Expert Pool Demo (Single GPU)")
    print("=" * 80)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80)
    
    # Configuration
    config = MoEConfig(
        d_model=2048,
        d_expert=1024,
        n_shared_experts=2,
        n_routed_experts=32,
        n_activated=4,
        top_k=4,
        aux_loss_free=True
    )
    
    print(f"\nMoE Configuration:")
    print(f"  Model dimension: {config.d_model}")
    print(f"  Expert FFN dimension: {config.d_expert}")
    print(f"  Shared experts: {config.n_shared_experts}")
    print(f"  Routed experts: {config.n_routed_experts}")
    print(f"  Activated per token: {config.n_activated} (2 shared + 4 routed)")
    print(f"  Auxiliary-loss-free: {config.aux_loss_free}")
    
    # Create model
    model = DeepSeekMoEModel(config, n_layers=4).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Parameters: {n_params/1e6:.1f}M")
    
    # Benchmark 1: Inference
    inference_results = benchmark_inference(model, config, device)
    
    # Benchmark 2: Expert scaling
    scaling_results = benchmark_expert_counts(device)
    
    # Benchmark 3: Load balancing
    load_stats = benchmark_load_balancing(config, device)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: Fine-Grained Expert Pool Key Points")
    print("=" * 80)
    print("""
1. FINE-GRAINED EXPERTS:
   - Many small experts (32 experts × 1024 dim) vs few large (4 experts × 8192 dim)
   - Same total capacity, but better specialization
   - Each expert focuses on specific patterns

2. SHARED + ROUTED DESIGN:
   - Shared experts: Always active, capture common patterns
   - Routed experts: Selected per token, specialized knowledge
   - Total activation: 2 shared + 4 routed = 6 experts per token

3. AUXILIARY-LOSS-FREE BALANCING:
   - No auxiliary loss that degrades model quality
   - Dynamic bias adjustment based on expert load
   - Maintains balanced utilization naturally

4. SCALING TO 8xH20:
   - 64+ experts distributed across 8 GPUs
   - Each GPU handles 8+ experts
   - NVLink enables efficient all-to-all communication
   - Fine-grained experts reduce memory per GPU
""")
    
    print("=" * 80)
    print("COMPARISON: Fine-Grained vs Coarse-Grained")
    print("=" * 80)
    
    # Create comparison table
    print("""
┌─────────────────────┬─────────────────────┬─────────────────────┐
│ Metric              │ Coarse (4 experts)  │ Fine (32 experts)   │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Expert size         │ 8192                │ 1024                │
│ Total experts       │ 4                   │ 32                  │
│ Activated/token     │ 2                   │ 4 (routed) + 2 (shared) │
│ Experts per GPU (8) │ 0.5                 │ 4                   │
│ Specialization      │ Low                 │ High                │
│ Load balance        │ Harder              │ Easier              │
│ Memory per expert   │ High                │ Low                 │
└─────────────────────┴─────────────────────┴─────────────────────┘
""")
    
    # Multi-GPU instructions
    print("=" * 80)
    print("Multi-GPU Deployment (8xH20)")
    print("=" * 80)
    print("""
To run on 8xH20 with NVLink:

1. Save this script as fine_grained_moe_demo.py

2. Run with torchrun:
   torchrun --nproc_per_node=8 fine_grained_moe_demo.py

3. Or use the multi-GPU version:
   python fine_grained_moe_demo_multigpu.py

Key optimizations for H20:
- Use NVLink for all-to-all communication
- Batch expert dispatch/combine operations
- Use FP16/BF16 for all computations
- Overlap computation with communication
""")


if __name__ == "__main__":
    main()
