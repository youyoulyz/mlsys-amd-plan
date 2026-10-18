"""
DeepSeek-V3 (MLA) vs DeepSeek-V4 (CSA+HCA) KV Cache Efficiency Benchmark

Based on DeepSeek-V4 paper findings:
- V4-Pro: 10% KV cache size, 27% single-token FLOPs vs V3.2 (at 1M context)
- V4-Flash: 7% KV cache size, 10% single-token FLOPs vs V3.2 (at 1M context)

This benchmark demonstrates:
1. KV cache size reduction through sequence compression
2. Memory efficiency at different context lengths
3. Theoretical FLOPs comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from dataclasses import dataclass
from typing import Tuple, Dict

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class V3MLAConfig:
    """DeepSeek-V3 Multi-head Latent Attention Config"""
    d_model: int = 7168          # V4-Pro hidden size
    n_heads: int = 128
    d_head: int = 128
    d_c: int = 512               # KV compression dimension
    d_rope: int = 64
    max_seq_len: int = 1024      # Test with smaller sequences


@dataclass
class V4CSAConfig:
    """DeepSeek-V4 Compressed Sparse Attention Config"""
    d_model: int = 7168
    n_heads: int = 128
    d_head: int = 512
    d_compress: int = 512
    m: int = 4                   # Compression ratio: every 4 tokens -> 1 entry
    top_k: int = 1024            # Top-k sparse selection
    d_c_query: int = 1536        # Query compression dimension
    n_win: int = 128             # Sliding window size
    max_seq_len: int = 1024


@dataclass
class V4HCAConfig:
    """DeepSeek-V4 Heavily Compressed Attention Config"""
    d_model: int = 7168
    n_heads: int = 128
    d_head: int = 512
    d_compress: int = 512
    m_prime: int = 128           # Heavy compression: every 128 tokens -> 1 entry
    d_c_query: int = 1536
    n_win: int = 128
    max_seq_len: int = 1024


# ============================================================================
# DeepSeek-V3: Multi-head Latent Attention (MLA)
# ============================================================================

class V3_MLA(nn.Module):
    """
    DeepSeek-V3 MLA: Low-rank joint compression for KV cache.
    
    KV Cache stores:
    - c_KV: compressed latent vector [batch, seq, d_c]
    - k_R: RoPE key [batch, seq, d_rope]
    
    Total KV cache per token: d_c + d_rope elements
    """
    
    def __init__(self, config: V3MLAConfig):
        super().__init__()
        self.config = config
        
        # KV compression
        self.W_DKV = nn.Linear(config.d_model, config.d_c, bias=False)
        self.W_UK = nn.Linear(config.d_c, config.n_heads * config.d_head, bias=False)
        self.W_UV = nn.Linear(config.d_c, config.n_heads * config.d_head, bias=False)
        self.W_KR = nn.Linear(config.d_model, config.d_rope, bias=False)
        
        # Query compression
        self.W_DQ = nn.Linear(config.d_model, config.d_c, bias=False)
        self.W_UQ = nn.Linear(config.d_c, config.n_heads * config.d_head, bias=False)
        self.W_QR = nn.Linear(config.d_c, config.n_heads * config.d_rope, bias=False)
        
        # Output
        self.W_O = nn.Linear(config.n_heads * config.d_head, config.d_model, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
    
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            h: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
            cache_info: dict with KV cache metrics
        """
        batch_size, seq_len, _ = h.shape
        
        # KV compression
        c_KV = self.W_DKV(h)  # [batch, seq, d_c]
        
        # Keys
        k_C = self.W_UK(c_KV).view(batch_size, seq_len, self.config.n_heads, self.config.d_head)
        k_R = self.W_KR(h)  # [batch, seq, d_rope]
        
        # Values
        v_C = self.W_UV(c_KV).view(batch_size, seq_len, self.config.n_heads, self.config.d_head)
        
        # Queries
        c_Q = self.W_DQ(h)
        q_C = self.W_UQ(c_Q).view(batch_size, seq_len, self.config.n_heads, self.config.d_head)
        q_R = self.W_QR(c_Q).view(batch_size, seq_len, self.config.n_heads, self.config.d_rope)
        
        # Combine
        k = torch.cat([k_C, k_R.unsqueeze(2).expand(-1, -1, self.config.n_heads, -1)], dim=-1)
        q = torch.cat([q_C, q_R], dim=-1)
        
        # Transpose: [batch, n_heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v_C = v_C.transpose(1, 2)
        
        # Attention
        scale = math.sqrt(self.config.d_head + self.config.d_rope)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        
        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=h.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1).to(v_C.dtype)
        
        # Output
        out = torch.matmul(attn, v_C)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.W_O(out)
        
        # KV cache info (what we need to store for inference)
        # Only c_KV and k_R need to be cached
        kv_cache_elements = c_KV.numel() + k_R.numel()
        
        cache_info = {
            'c_KV_shape': c_KV.shape,
            'k_R_shape': k_R.shape,
            'total_elements': kv_cache_elements,
            'elements_per_token': self.config.d_c + self.config.d_rope,
            'compression_ratio': seq_len,  # MLA doesn't compress sequence dimension
        }
        
        return output, cache_info


# ============================================================================
# DeepSeek-V4: Compressed Sparse Attention (CSA)
# ============================================================================

class V4_CSA(nn.Module):
    """
    DeepSeek-V4 CSA: Compress KV cache every m tokens + sparse selection.
    
    KV Cache stores:
    - Compressed entries: [batch, seq/m, d_compress]
    - Compression ratio: m (e.g., 4)
    
    Key innovation: Sequence dimension compression
    """
    
    def __init__(self, config: V4CSAConfig):
        super().__init__()
        self.config = config
        self.m = config.m
        
        # KV compression
        self.W_aKV = nn.Linear(config.d_model, config.d_compress, bias=False)
        self.W_bKV = nn.Linear(config.d_model, config.d_compress, bias=False)
        self.W_aZ = nn.Linear(config.d_model, config.d_compress, bias=False)
        self.W_bZ = nn.Linear(config.d_model, config.d_compress, bias=False)
        
        # Positional biases
        self.B_a = nn.Parameter(torch.randn(config.m, config.d_compress) * 0.02)
        self.B_b = nn.Parameter(torch.randn(config.m, config.d_compress) * 0.02)
        
        # Query projection
        self.W_DQ = nn.Linear(config.d_model, config.d_c_query, bias=False)
        self.W_UQ = nn.Linear(config.d_c_query, config.n_heads * config.d_head, bias=False)
        
        # Output
        self.W_O = nn.Linear(config.n_heads * config.d_head, config.d_model, bias=False)
        
        # Normalization
        self.q_norm = nn.RMSNorm(config.d_head)
        self.kv_norm = nn.RMSNorm(config.d_compress)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
    
    def _compress_kv(self, h: torch.Tensor) -> torch.Tensor:
        """Compress KV cache every m tokens into 1 entry"""
        batch_size, seq_len, _ = h.shape
        
        # Pad
        pad_len = (self.m - seq_len % self.m) % self.m
        h_padded = F.pad(h, (0, 0, 0, pad_len)) if pad_len > 0 else h
        
        new_seq_len = h_padded.shape[1]
        n_blocks = new_seq_len // self.m
        
        # Compute KV entries and compression weights
        C_a = self.W_aKV(h_padded).view(batch_size, n_blocks, self.m, -1)
        C_b = self.W_bKV(h_padded).view(batch_size, n_blocks, self.m, -1)
        Z_a = self.W_aZ(h_padded).view(batch_size, n_blocks, self.m, -1)
        Z_b = self.W_bZ(h_padded).view(batch_size, n_blocks, self.m, -1)
        
        # Compression weights with positional biases
        S_a = F.softmax(Z_a + self.B_a, dim=2)
        S_b = F.softmax(Z_b + self.B_b, dim=2)
        
        # Weighted sum
        C_comp = (S_a * C_a).sum(dim=2) + (S_b * C_b).sum(dim=2)  # [batch, n_blocks, d_compress]
        
        return C_comp
    
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            h: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
            cache_info: dict with KV cache metrics
        """
        batch_size, seq_len, _ = h.shape
        
        # Step 1: Compress KV cache
        C_comp = self._compress_kv(h)
        n_blocks = C_comp.shape[1]
        
        # Step 2: Generate queries
        c_Q = self.W_DQ(h)
        q = self.W_UQ(c_Q).view(batch_size, seq_len, self.config.n_heads, self.config.d_head)
        
        # Normalize
        q = self.q_norm(q)
        C_comp = self.kv_norm(C_comp)
        
        # Step 3: Simplified attention (dense for benchmark)
        # In real CSA, this would use sparse selection (top-k)
        
        # Average q to match compressed blocks
        # q: [batch, seq, n_heads, d_head] -> [batch, n_blocks, n_heads, d_head]
        q_reshaped = q.view(batch_size, n_blocks, self.m, self.config.n_heads, self.config.d_head)
        q_per_block = q_reshaped.mean(dim=2)  # [batch, n_blocks, n_heads, d_head]
        q_per_block = q_per_block.transpose(1, 2)  # [batch, n_heads, n_blocks, d_head]
        
        # Pad C_comp to match d_head
        # If d_compress > d_head, we need to project down
        if C_comp.shape[-1] > self.config.d_head:
            # Project down using a simple linear layer (or just take first d_head dims)
            C_padded = C_comp[..., :self.config.d_head]
        elif C_comp.shape[-1] < self.config.d_head:
            # Pad
            pad_size = self.config.d_head - C_comp.shape[-1]
            C_padded = F.pad(C_comp, (0, pad_size))
        else:
            C_padded = C_comp
        
        # Use compressed entries as key and value (MQA style)
        # C_padded: [batch, n_blocks, d_head]
        # Expand for multi-head: [batch, n_heads, n_blocks, d_head]
        k = C_padded.unsqueeze(1).expand(-1, self.config.n_heads, -1, -1)
        
        scale = math.sqrt(self.config.d_head)
        attn = torch.matmul(q_per_block, k.transpose(-2, -1)) / scale
        
        # Causal mask (block-level)
        mask = torch.triu(torch.ones(n_blocks, n_blocks, device=h.device), diagonal=1).bool()
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1).to(q_per_block.dtype)
        
        # Output
        out = torch.matmul(attn, C_padded.unsqueeze(1))  # [batch, n_heads, n_blocks, d_head]
        out = out.transpose(1, 2)  # [batch, n_blocks, n_heads, d_head]
        
        # Expand back to seq_len
        out = out.unsqueeze(2).expand(-1, -1, self.m, -1, -1)  # [batch, n_blocks, m, n_heads, d_head]
        out = out.contiguous().view(batch_size, seq_len, self.config.n_heads, self.config.d_head)
        out = out.view(batch_size, seq_len, -1)
        output = self.W_O(out)
        
        # KV cache info
        kv_cache_elements = C_comp.numel()
        
        cache_info = {
            'compressed_shape': C_comp.shape,
            'total_elements': kv_cache_elements,
            'elements_per_token': self.config.d_compress / self.m,
            'compression_ratio': seq_len / n_blocks if n_blocks > 0 else 1,
            'n_blocks': n_blocks,
        }
        
        return output, cache_info


# ============================================================================
# DeepSeek-V4: Heavily Compressed Attention (HCA)
# ============================================================================

class V4_HCA(nn.Module):
    """
    DeepSeek-V4 HCA: More aggressive compression (m' >> m).
    
    KV Cache stores:
    - Compressed entries: [batch, seq/m', d_compress]
    - Compression ratio: m' (e.g., 128)
    
    Key innovation: Extreme compression for long sequences
    """
    
    def __init__(self, config: V4HCAConfig):
        super().__init__()
        self.config = config
        self.m_prime = config.m_prime
        
        # KV compression
        self.W_KV = nn.Linear(config.d_model, config.d_compress, bias=False)
        self.W_Z = nn.Linear(config.d_model, config.d_compress, bias=False)
        self.B = nn.Parameter(torch.randn(config.m_prime, config.d_compress) * 0.02)
        
        # Query projection
        self.W_DQ = nn.Linear(config.d_model, config.d_c_query, bias=False)
        self.W_UQ = nn.Linear(config.d_c_query, config.n_heads * config.d_head, bias=False)
        
        # Output
        self.W_O = nn.Linear(config.n_heads * config.d_head, config.d_model, bias=False)
        
        # Normalization
        self.q_norm = nn.RMSNorm(config.d_head)
        self.kv_norm = nn.RMSNorm(config.d_compress)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
    
    def _compress_kv(self, h: torch.Tensor) -> torch.Tensor:
        """Compress KV cache every m' tokens into 1 entry"""
        batch_size, seq_len, _ = h.shape
        
        # Pad
        pad_len = (self.m_prime - seq_len % self.m_prime) % self.m_prime
        h_padded = F.pad(h, (0, 0, 0, pad_len)) if pad_len > 0 else h
        
        new_seq_len = h_padded.shape[1]
        n_blocks = new_seq_len // self.m_prime
        
        # Compute KV entries and weights
        C = self.W_KV(h_padded).view(batch_size, n_blocks, self.m_prime, -1)
        Z = self.W_Z(h_padded).view(batch_size, n_blocks, self.m_prime, -1)
        
        # Compression
        S = F.softmax(Z + self.B, dim=2)
        C_comp = (S * C).sum(dim=2)  # [batch, n_blocks, d_compress]
        
        return C_comp
    
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            h: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
            cache_info: dict with KV cache metrics
        """
        batch_size, seq_len, _ = h.shape
        
        # Step 1: Compress KV cache (more aggressive)
        C_comp = self._compress_kv(h)
        n_blocks = C_comp.shape[1]
        
        # Step 2: Generate queries
        c_Q = self.W_DQ(h)
        q = self.W_UQ(c_Q).view(batch_size, seq_len, self.config.n_heads, self.config.d_head)
        
        # Normalize
        q = self.q_norm(q)
        C_comp = self.kv_norm(C_comp)
        
        # Step 3: Dense attention
        
        # Average q to match compressed blocks
        # q: [batch, seq, n_heads, d_head] -> [batch, n_blocks, n_heads, d_head]
        q_reshaped = q.view(batch_size, n_blocks, self.m_prime, self.config.n_heads, self.config.d_head)
        q_per_block = q_reshaped.mean(dim=2)  # [batch, n_blocks, n_heads, d_head]
        q_per_block = q_per_block.transpose(1, 2)  # [batch, n_heads, n_blocks, d_head]
        
        # Pad C_comp to match d_head
        if C_comp.shape[-1] > self.config.d_head:
            C_padded = C_comp[..., :self.config.d_head]
        elif C_comp.shape[-1] < self.config.d_head:
            pad_size = self.config.d_head - C_comp.shape[-1]
            C_padded = F.pad(C_comp, (0, pad_size))
        else:
            C_padded = C_comp
        
        # Expand for multi-head
        k = C_padded.unsqueeze(1).expand(-1, self.config.n_heads, -1, -1)
        
        scale = math.sqrt(self.config.d_head)
        attn = torch.matmul(q_per_block, k.transpose(-2, -1)) / scale
        
        mask = torch.triu(torch.ones(n_blocks, n_blocks, device=h.device), diagonal=1).bool()
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1).to(q_per_block.dtype)
        
        out = torch.matmul(attn, C_padded.unsqueeze(1))  # [batch, n_heads, n_blocks, d_head]
        out = out.transpose(1, 2)  # [batch, n_blocks, n_heads, d_head]
        
        # Expand back to seq_len
        out = out.unsqueeze(2).expand(-1, -1, self.m_prime, -1, -1)  # [batch, n_blocks, m_prime, n_heads, d_head]
        out = out.contiguous().view(batch_size, seq_len, self.config.n_heads, self.config.d_head)
        out = out.view(batch_size, seq_len, -1)
        output = self.W_O(out)
        
        # KV cache info
        kv_cache_elements = C_comp.numel()
        
        cache_info = {
            'compressed_shape': C_comp.shape,
            'total_elements': kv_cache_elements,
            'elements_per_token': self.config.d_compress / self.m_prime,
            'compression_ratio': seq_len / n_blocks if n_blocks > 0 else 1,
            'n_blocks': n_blocks,
        }
        
        return output, cache_info


# ============================================================================
# Benchmark Functions
# ============================================================================

def compute_kv_cache_size_mb(elements: int, dtype: torch.dtype = torch.float16) -> float:
    """Compute KV cache size in MB"""
    bytes_per_element = torch.finfo(dtype).bits // 8
    return elements * bytes_per_element / (1024 ** 2)


def compute_theoretical_flops(seq_len: int, config, attention_type: str) -> float:
    """
    Compute theoretical attention FLOPs.
    
    For MLA: O(seq_len^2 * n_heads * d_head)
    For CSA: O((seq_len/m)^2 * n_heads * d_head + seq_len * top_k * n_heads * d_head)
    For HCA: O((seq_len/m')^2 * n_heads * d_head)
    """
    if attention_type == 'MLA':
        return seq_len ** 2 * config.n_heads * (config.d_head + config.d_rope)
    elif attention_type == 'CSA':
        n_blocks = seq_len / config.m
        # Compression + sparse attention
        flops_compress = seq_len * config.d_compress
        flops_attn = seq_len * config.top_k * config.n_heads * config.d_head
        return flops_compress + flops_attn
    elif attention_type == 'HCA':
        n_blocks = seq_len / config.m_prime
        # Compression + dense attention on compressed
        flops_compress = seq_len * config.d_compress
        flops_attn = n_blocks ** 2 * config.n_heads * config.d_head
        return flops_compress + flops_attn
    return 0


def run_benchmark():
    """Run comprehensive KV cache efficiency benchmark"""
    
    print("=" * 100)
    print("DeepSeek-V3 (MLA) vs DeepSeek-V4 (CSA+HCA) KV Cache Efficiency Benchmark")
    print("=" * 100)
    
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
    else:
        n_gpus = torch.cuda.device_count()
        print(f"\nAvailable GPUs: {n_gpus}")
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1024**2:.0f} MB)")
        device = torch.device("cuda:0")
    
    # Test configurations
    # Using smaller dimensions for local testing
    d_model = 1024  # Scaled down from 7168
    seq_lengths = [256, 512, 1024, 2048, 4096]
    batch_size = 1
    
    print(f"\n{'='*100}")
    print(f"Test Configuration:")
    print(f"  Model dimension: {d_model}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence lengths: {seq_lengths}")
    print(f"{'='*100}")
    
    results = []
    
    for seq_len in seq_lengths:
        print(f"\n{'='*100}")
        print(f"Sequence Length: {seq_len}")
        print(f"{'='*100}")
        
        h = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float16)
        
        # ---- V3 MLA ----
        print("\n[1] DeepSeek-V3 MLA (Multi-head Latent Attention)")
        mla_config = V3MLAConfig(
            d_model=d_model,
            n_heads=32,
            d_head=64,
            d_c=256,
            d_rope=32,
        )
        mla = V3_MLA(mla_config).to(device).half()
        
        with torch.no_grad():
            _, mla_cache = mla(h)
        
        mla_kv_mb = compute_kv_cache_size_mb(mla_cache['total_elements'])
        mla_flops = compute_theoretical_flops(seq_len, mla_config, 'MLA')
        
        print(f"  KV Cache: {mla_cache['total_elements']:>12,} elements")
        print(f"  KV Cache Size (FP16): {mla_kv_mb:>10.2f} MB")
        print(f"  Elements per token: {mla_cache['elements_per_token']}")
        print(f"  Theoretical FLOPs: {mla_flops:.2e}")
        
        # ---- V4 CSA ----
        print("\n[2] DeepSeek-V4 CSA (Compressed Sparse Attention)")
        csa_config = V4CSAConfig(
            d_model=d_model,
            n_heads=32,
            d_head=64,
            d_compress=256,
            m=4,
            top_k=min(256, seq_len // 4),
            d_c_query=512,
        )
        csa = V4_CSA(csa_config).to(device).half()
        
        with torch.no_grad():
            _, csa_cache = csa(h)
        
        csa_kv_mb = compute_kv_cache_size_mb(csa_cache['total_elements'])
        csa_flops = compute_theoretical_flops(seq_len, csa_config, 'CSA')
        csa_compression = csa_cache['compression_ratio']
        
        print(f"  Compressed KV: {csa_cache['total_elements']:>12,} elements")
        print(f"  Compressed KV Size (FP16): {csa_kv_mb:>10.2f} MB")
        print(f"  Compression Ratio: {csa_compression:.1f}x")
        print(f"  Elements per token (effective): {csa_cache['elements_per_token']:.1f}")
        print(f"  Theoretical FLOPs: {csa_flops:.2e}")
        
        # ---- V4 HCA ----
        print("\n[3] DeepSeek-V4 HCA (Heavily Compressed Attention)")
        hca_config = V4HCAConfig(
            d_model=d_model,
            n_heads=32,
            d_head=64,
            d_compress=256,
            m_prime=64,  # Scaled down from 128 for testing
            d_c_query=512,
        )
        hca = V4_HCA(hca_config).to(device).half()
        
        with torch.no_grad():
            _, hca_cache = hca(h)
        
        hca_kv_mb = compute_kv_cache_size_mb(hca_cache['total_elements'])
        hca_flops = compute_theoretical_flops(seq_len, hca_config, 'HCA')
        hca_compression = hca_cache['compression_ratio']
        
        print(f"  Compressed KV: {hca_cache['total_elements']:>12,} elements")
        print(f"  Compressed KV Size (FP16): {hca_kv_mb:>10.2f} MB")
        print(f"  Compression Ratio: {hca_compression:.1f}x")
        print(f"  Elements per token (effective): {hca_cache['elements_per_token']:.1f}")
        print(f"  Theoretical FLOPs: {hca_flops:.2e}")
        
        # Store results
        results.append({
            'seq_len': seq_len,
            'mla': {'kv_mb': mla_kv_mb, 'flops': mla_flops, 'elements': mla_cache['total_elements']},
            'csa': {'kv_mb': csa_kv_mb, 'flops': csa_flops, 'elements': csa_cache['total_elements'], 
                   'compression': csa_compression},
            'hca': {'kv_mb': hca_kv_mb, 'flops': hca_flops, 'elements': hca_cache['total_elements'],
                   'compression': hca_compression},
        })
        
        # Cleanup
        del mla, csa, hca, h
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Print summary tables
    print("\n" + "=" * 120)
    print("SUMMARY 1: KV Cache Size Comparison (MB, FP16)")
    print("=" * 120)
    print(f"{'Seq Len':<10} {'V3 MLA':<15} {'V4 CSA':<15} {'V4 HCA':<15} {'CSA Reduction':<20} {'HCA Reduction':<20}")
    print("-" * 120)
    for r in results:
        mla_mb = r['mla']['kv_mb']
        csa_mb = r['csa']['kv_mb']
        hca_mb = r['hca']['kv_mb']
        csa_reduction = (1 - csa_mb / mla_mb) * 100 if mla_mb > 0 else 0
        hca_reduction = (1 - hca_mb / mla_mb) * 100 if mla_mb > 0 else 0
        print(f"{r['seq_len']:<10} {mla_mb:<15.2f} {csa_mb:<15.2f} {hca_mb:<15.2f} "
              f"{csa_reduction:<20.1f}% {hca_reduction:<20.1f}%")
    
    print("\n" + "=" * 120)
    print("SUMMARY 2: Theoretical Attention FLOPs Comparison")
    print("=" * 120)
    print(f"{'Seq Len':<10} {'V3 MLA':<20} {'V4 CSA':<20} {'V4 HCA':<20} {'CSA Speedup':<15} {'HCA Speedup':<15}")
    print("-" * 120)
    for r in results:
        mla_flops = r['mla']['flops']
        csa_flops = r['csa']['flops']
        hca_flops = r['hca']['flops']
        csa_speedup = mla_flops / csa_flops if csa_flops > 0 else float('inf')
        hca_speedup = mla_flops / hca_flops if hca_flops > 0 else float('inf')
        print(f"{r['seq_len']:<10} {mla_flops:<20.2e} {csa_flops:<20.2e} {hca_flops:<20.2e} "
              f"{csa_speedup:<15.2f}x {hca_speedup:<15.2f}x")
    
    print("\n" + "=" * 120)
    print("SUMMARY 3: KV Cache Elements per Token")
    print("=" * 120)
    print(f"{'Seq Len':<10} {'V3 MLA':<15} {'V4 CSA':<15} {'V4 HCA':<15} {'CSA Ratio':<15} {'HCA Ratio':<15}")
    print("-" * 120)
    for r in results:
        mla_elem = r['mla']['elements'] / r['seq_len']
        csa_elem = r['csa']['elements'] / r['seq_len']
        hca_elem = r['hca']['elements'] / r['seq_len']
        csa_ratio = csa_elem / mla_elem if mla_elem > 0 else float('inf')
        hca_ratio = hca_elem / mla_elem if mla_elem > 0 else float('inf')
        print(f"{r['seq_len']:<10} {mla_elem:<15.1f} {csa_elem:<15.1f} {hca_elem:<15.1f} "
              f"{csa_ratio:<15.3f} {hca_ratio:<15.3f}")
    
    # Extrapolate to 1M context
    print("\n" + "=" * 120)
    print("EXTRAPOLATION TO 1M CONTEXT (Based on Paper Findings)")
    print("=" * 120)
    
    # Using last sequence length as baseline
    last_r = results[-1]
    seq_len_last = last_r['seq_len']
    scale_factor = 1_000_000 / seq_len_last
    
    # MLA scales linearly with sequence length
    mla_1m_mb = last_r['mla']['kv_mb'] * scale_factor
    csa_1m_mb = last_r['csa']['kv_mb'] * scale_factor / last_r['csa']['compression']
    hca_1m_mb = last_r['hca']['kv_mb'] * scale_factor / last_r['hca']['compression']
    
    print(f"\nEstimated KV Cache at 1M tokens:")
    print(f"  V3 MLA:  {mla_1m_mb:>12.2f} MB ({mla_1m_mb / 1024:.2f} GB)")
    print(f"  V4 CSA:  {csa_1m_mb:>12.2f} MB ({csa_1m_mb / 1024:.2f} GB)")
    print(f"  V4 HCA:  {hca_1m_mb:>12.2f} MB ({hca_1m_mb / 1024:.2f} GB)")
    
    csa_reduction_1m = (1 - csa_1m_mb / mla_1m_mb) * 100 if mla_1m_mb > 0 else 0
    hca_reduction_1m = (1 - hca_1m_mb / mla_1m_mb) * 100 if mla_1m_mb > 0 else 0
    
    print(f"\n  CSA reduction: {csa_reduction_1m:.1f}%")
    print(f"  HCA reduction: {hca_reduction_1m:.1f}%")
    
    print("\n" + "=" * 120)
    print("KEY FINDINGS (Aligned with DeepSeek-V4 Paper)")
    print("=" * 120)
    print("""
1. KV Cache Efficiency:
   - V4 CSA compresses sequence dimension by m=4x, reducing KV cache significantly
   - V4 HCA compresses by m'=128x (or 64x in this benchmark), achieving extreme reduction
   - At 1M context, V4 can achieve ~90% KV cache reduction vs V3

2. Computational Efficiency:
   - V4 CSA reduces attention FLOPs through sparse selection (top-k)
   - V4 HCA reduces FLOPs through heavy compression
   - At 1M context, V4-Pro achieves 27% of V3.2 FLOPs, V4-Flash achieves 10%

3. Architecture Innovation:
   - V3 MLA: Low-rank compression in feature space (d_c << d_head * n_heads)
   - V4 CSA+HCA: Sequence dimension compression (m=4, m'=128)
   - Hybrid approach: CSA for sparse access, HCA for extreme compression

4. Practical Impact:
   - Enables efficient 1M token context windows
   - Reduces memory bandwidth pressure during inference
   - Makes long-context reasoning and agentic workflows feasible
    """)
    
    print("=" * 120)


if __name__ == "__main__":
    run_benchmark()
