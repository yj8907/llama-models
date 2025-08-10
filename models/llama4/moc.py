import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple

class MemoryBank(nn.Module):
    def __init__(self, dim: int, memory_size: int = 512, tokens_per_memory: int = 256, 
        top_k: int = 8, dtype=torch.float32):
        super().__init__()
        self.memory_size = memory_size
        self.dim = dim
        self.top_k = min(top_k, memory_size)
        self.tokens_per_memory = tokens_per_memory

        # Memory storage - learnable memory tokens (similar to expert parameters)
        self.memory_tokens = nn.Parameter(torch.empty(memory_size, self.tokens_per_memory, dim, dtype=dtype))
        
        # Memory router (similar to MoE router)
        self.memory_router = nn.Parameter(torch.empty(dim, memory_size, dtype=dtype))
        
        # Memory projection and gating
        self.memory_gate = nn.Linear(dim, 1)
        self.memory_combine = nn.Linear(dim * (self.top_k + 1), dim)  # +1 for original input
        
        # Memory update mechanism
        self.memory_update_gate = nn.Linear(dim * 2, dim)
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.memory_tokens)
        nn.init.xavier_uniform_(self.memory_router)
        
    def forward(self, x: torch.Tensor, update_memory: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [batch_size, seq_len, dim]. First 1+self.tokens_per_memory tokens are used for memory/contexts.
        """
        batch_size, seq_len, dim = x.shape
        
        # Only get the first token for routing. First token is used to summarize the context. 
        x_flat = x[:,0, :].view(-1, dim)
        x_a = x_flat.shape[0]
        
        # Compute router scores: [batch * seq_len, memory_size]
        router_scores = torch.matmul(x_flat, self.memory_router)
        
        # Get top-k memory tokens for each position
        router_scores_topk, router_indices_topk = torch.topk(router_scores, self.top_k, dim=1)
        # router_scores_topk: [batch * seq_len, top_k]
        # router_indices_topk: [batch * seq_len, top_k]
        
        # Apply softmax to top-k scores for weighted combination
        router_weights = torch.softmax(router_scores_topk, dim=1)  # [batch * seq_len, top_k]
        
        # Gather top-k memory tokens
        # router_indices_topk: [batch * seq_len, top_k] -> [batch * seq_len * top_k]
        indices_flat = router_indices_topk.view(-1)
        # memory_tokens: [memory_size, dim] -> gathered: [batch * seq_len * top_k, dim]
        gathered_memory = self.memory_tokens[indices_flat]
        # Reshape back: [batch * seq_len, top_k, tokens_per_memory,dim]
        gathered_memory = gathered_memory.view(x_a, self.top_k, self.tokens_per_memory, dim)
        
        # Weight and combine memory tokens
        # router_weights: [batch * seq_len, top_k, 1, 1] * gathered_memory: [batch * seq_len, top_k, tokens_per_memory, dim]
        weighted_memory = router_weights.unsqueeze(-1).unsqueeze(-1) * gathered_memory
        
        #TODO: what's best way to handle this? 
        # # Sum across top_k dimension: [batch * seq_len, tokens_per_memory, dim]
        combined_memory = torch.sum(weighted_memory, dim=1)
        
        # Memory gating - decide how much to use memory
        # memory_gate = torch.sigmoid(self.memory_gate(x_flat))  # [batch * seq_len, 1]
        # gated_memory = memory_gate * combined_memory  # [batch * seq_len, dim]
        
        combined_memory = combined_memory.view(batch_size, self.tokens_per_memory, dim)
        x[:, 1: self.tokens_per_memory+1, :] = combined_memory
        
        # Reshape back to original shape
        # memory_output = memory_output.view(batch_size, seq_len, dim)
        # router_weights = router_weights.view(batch_size, seq_len, self.top_k)
        # router_indices_topk = router_indices_topk.view(batch_size, seq_len, self.top_k)
        
        # Update memory if requested (typically during training)
        if update_memory and self.training:
            # Compute update signals for selected memory tokens
            with torch.no_grad():
                # Create update mask for selected memory tokens
                update_counts = torch.zeros(self.memory_size, device=x.device)
                update_values = torch.zeros(self.memory_size, dim, device=x.device)
                
                # Accumulate updates for each memory token
                for i in range(self.top_k):
                    indices = router_indices_topk[:, :, i].flatten()  # [batch * seq_len]
                    weights = router_weights[:, :, i].flatten().unsqueeze(-1)  # [batch * seq_len, 1]
                    values = x_flat * weights  # [batch * seq_len, dim]
                    
                    # Scatter add the weighted updates
                    update_values.index_add_(0, indices, values)
                    update_counts.index_add_(0, indices, weights.squeeze(-1))
                
                # Apply updates with exponential moving average
                # Avoid division by zero
                update_counts = torch.clamp(update_counts, min=1e-8)
                avg_updates = update_values / update_counts.unsqueeze(-1)
                
                # Update memory tokens with exponential moving average
                alpha = 0.01  # Learning rate for memory updates
                mask = update_counts > 0
                self.memory_tokens.data[mask] = (
                    (1 - alpha) * self.memory_tokens.data[mask] + 
                    alpha * avg_updates[mask]
                )
        
        return x