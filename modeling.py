import os, re, math, json, torch
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.checkpoint import checkpoint
from safetensors.torch import save_file, load_file
from torch import nn
from tqdm import tqdm
import numpy as np

# === Utility Functions ===
def save_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_texts(file_path):
    with open(file_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# === Configuration Class ===
class ChatConfig:
    def __init__(self, vocab_size, max_seq_length, hidden_size=256, num_layers=4, num_heads=8, 
                 rope_dim=32, feed_forward_dim=640, window_size=512, dropout=0.1, 
                 num_experts=4, expert_loss_weight=0.01, num_kv_heads=None, 
                 use_sliding_window=False, use_flash_attn=False, use_rmsnorm=True,
                 use_rotary=True, rotary_base=10000.0, rope_scaling=None,
                 use_gqa=False, use_moe=True, attention_bias=False,
                 use_parallel_residual=False, norm_eps=1e-5):
        
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.rope_dim = rope_dim
        self.feed_forward_dim = feed_forward_dim
        self.window_size = window_size
        self.dropout = dropout
        self.num_experts = num_experts
        self.expert_loss_weight = expert_loss_weight
        
        # New advanced parameters
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.use_sliding_window = use_sliding_window
        self.use_flash_attn = use_flash_attn
        self.use_rmsnorm = use_rmsnorm
        self.use_rotary = use_rotary
        self.rotary_base = rotary_base
        self.rope_scaling = rope_scaling
        self.use_gqa = use_gqa
        self.use_moe = use_moe
        self.attention_bias = attention_bias
        self.use_parallel_residual = use_parallel_residual
        self.norm_eps = norm_eps

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        save_json(self.__dict__, os.path.join(path, "config.json"))

    @classmethod
    def from_pretrained(cls, path):
        return cls(**load_json(os.path.join(path, "config.json")))

# === RMSNorm Module ===
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# === Rotary Embedding Functions ===
def precompute_freqs_cis(dim, max_seq_len, base=10000.0, scaling_factor=None):
    """Precompute the frequency tensor for complex exponentials (cis) with rotary embeddings"""
    # Scaling is only applied if explicitly specified
    if scaling_factor is not None:
        base = base * scaling_factor ** (dim / (dim - 2))
        
    # Create the theta parameter
    theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    seq_idx = torch.arange(max_seq_len).float()
    
    # Compute the complex exponential (cis)
    idx_theta = torch.outer(seq_idx, theta)
    freqs_cos = torch.cos(idx_theta)
    freqs_sin = torch.sin(idx_theta)
    
    return freqs_cos, freqs_sin

def apply_rotary_emb(x, freqs_cos, freqs_sin, seq_dimension=1):
    """Apply rotary embeddings to the given tensor"""
    # Extract shapes
    batch, seq_len, n_heads, head_dim = x.shape
    dim = head_dim // 2
    
    # Reshape for the rotation
    x = x.view(batch, seq_len, n_heads, 2, dim)
    x1, x2 = x[..., 0, :], x[..., 1, :]
    
    # Get the appropriate frequencies for this sequence
    cos = freqs_cos[:seq_len].to(x.device)
    sin = freqs_sin[:seq_len].to(x.device)
    
    # Apply rotation - this is elementwise multiplication
    # followed by swapping the odd/even dimensions and changing signs
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos
    
    # Recombine the rotated values with the original order of dimensions
    result = torch.stack([rotated_x1, rotated_x2], dim=-2)
    return result.view(batch, seq_len, n_heads, head_dim)

# === Attention Mechanisms ===
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads if config.use_gqa else config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        
        # For GQA, we have different numbers of q and kv heads
        self.kv_repeats = self.num_heads // self.num_kv_heads if config.use_gqa else 1
        
        # Q, K, V projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Precompute freqs for rotary embeddings if used
        if config.use_rotary:
            rope_scaling = None
            if config.rope_scaling is not None:
                rope_scaling = config.rope_scaling
            self.freqs_cos, self.freqs_sin = precompute_freqs_cis(
                self.head_dim, 
                config.max_seq_length * 2,  # 2x to handle longer sequences
                config.rotary_base,
                rope_scaling
            )
        else:
            self.freqs_cos, self.freqs_sin = None, None
        
        # Flash attention support
        self.use_flash_attn = config.use_flash_attn and hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x, attention_mask=None, past_kv=None, use_cache=False):
        batch_size, seq_length, _ = x.shape
        
        # Compute query, key, value vectors
        q = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        
        # Apply rotary embeddings if using RoPE
        if self.config.use_rotary:
            q = apply_rotary_emb(q, self.freqs_cos, self.freqs_sin)
            k = apply_rotary_emb(k, self.freqs_cos, self.freqs_sin)
        
        # Handle GQA key-value repetitions
        if self.kv_repeats > 1:
            k = k.repeat_interleave(self.kv_repeats, dim=2)
            v = v.repeat_interleave(self.kv_repeats, dim=2)
        
        # Combine with past key-values if provided
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        
        # Store key-values for potential future use
        current_kv = (k, v) if use_cache else None
        kv_seq_length = k.shape[1]
        
        # Prepare for attention computation
        q = q.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)  # (batch, num_heads, kv_seq_len, head_dim)
        v = v.transpose(1, 2)  # (batch, num_heads, kv_seq_len, head_dim)
        
        # Apply attention mechanism
        if self.use_flash_attn and attention_mask is None:
            # Use Flash Attention if available and no explicit mask is needed
            attn_output = F.scaled_dot_product_attention(
                q, k, v, 
                dropout_p=self.config.dropout if self.training else 0.0
            )
        else:
            # Traditional attention calculation
            attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            
            # Apply attention mask if provided
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask
                
            # Apply sliding window attention if configured
            if self.config.use_sliding_window and kv_seq_length > self.config.window_size:
                window_mask = torch.triu(
                    torch.ones((kv_seq_length, kv_seq_length), dtype=torch.bool, device=q.device),
                    diagonal=self.config.window_size + 1
                ).transpose(0, 1)
                window_mask = window_mask.expand(batch_size, self.num_heads, kv_seq_length, kv_seq_length)
                window_mask = window_mask[:, :, -seq_length:, :]
                attn_scores = attn_scores.masked_fill(window_mask, float("-inf"))
                
            # Compute attention probabilities and apply dropout
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            
            # Compute attention output
            attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        attn_output = self.resid_dropout(self.out_proj(attn_output))
        
        return attn_output, current_kv

# === GEGLU Module ===
class GEGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size, bias=True):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x):
        gate = F.gelu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

# === MoE Feed Forward ===
class MoEFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.experts = nn.ModuleList([
            GEGLU(
                config.hidden_size, 
                config.feed_forward_dim, 
                bias=True
            ) for _ in range(config.num_experts)
        ])
        self.gate = nn.Linear(config.hidden_size, config.num_experts)
        self.dropout = nn.Dropout(config.dropout)
        self.num_experts = config.num_experts
        self.expert_capacity = 2  # Number of tokens per expert (can be adjusted)

    def forward(self, x):
        orig_shape = x.shape
        x = x.view(-1, x.shape[-1])  # Reshape to (batch*seq_len, hidden_size)
        
        # Calculate routing probabilities
        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Find top-k experts for each token
        routing_weights, indices = torch.topk(routing_weights, k=self.expert_capacity, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Create expert masks
        expert_mask = torch.zeros(
            x.shape[0], self.num_experts, device=x.device, dtype=torch.bool
        )
        expert_mask.scatter_(1, indices, True)
        
        # Process tokens through their assigned experts
        final_output = torch.zeros_like(x)
        expert_utilization = torch.zeros(self.num_experts, device=x.device)
        
        for expert_idx, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            token_indices = expert_mask[:, expert_idx].nonzero().squeeze(-1)
            if token_indices.shape[0] == 0:
                continue
            
            # Get expert weights for these tokens
            token_weights = routing_weights[torch.arange(routing_weights.shape[0]), 
                                           torch.where(indices == expert_idx)[1]]
            
            # Process tokens through this expert
            expert_output = expert(x[token_indices])
            
            # Add weighted output to the final result
            final_output[token_indices] += expert_output * token_weights.unsqueeze(-1)
            
            # Track expert utilization
            expert_utilization[expert_idx] = token_indices.shape[0] / x.shape[0]
        
        # Calculate auxiliary load balancing loss
        # Ideal: uniform utilization across experts
        expected_utilization = 1.0 / self.num_experts
        balance_loss = torch.sum((expert_utilization - expected_utilization) ** 2)
        
        # Apply dropout
        final_output = self.dropout(final_output)
        
        return final_output.view(orig_shape), balance_loss

# === Standard Feed Forward ===
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.geglu = GEGLU(config.hidden_size, config.feed_forward_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.geglu(x)), 0.0  # Return 0 loss to match MoE interface

# === Transformer Block ===
class TransformerBlock(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.use_parallel_residual = config.use_parallel_residual
        
        # Choose normalization type
        norm_class = RMSNorm if config.use_rmsnorm else nn.LayerNorm
        norm_kwargs = {"dim": config.hidden_size} if config.use_rmsnorm else {"normalized_shape": config.hidden_size, "eps": config.norm_eps}
        
        # Normalization layers
        self.attn_norm = norm_class(**norm_kwargs)
        self.ffn_norm = norm_class(**norm_kwargs) if not self.use_parallel_residual else None
        
        # Attention layer
        self.self_attn = MultiHeadAttention(config)
        
        # Feed-forward layer (MoE or standard)
        if config.use_moe:
            self.ffn = MoEFeedForward(config)
        else:
            self.ffn = FeedForward(config)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def _forward_impl(self, x, attention_mask=None, past_kv=None, use_cache=False):
        # Normalize for attention (pre-norm architecture)
        attn_input = self.attn_norm(x)
        
        # Apply attention
        attn_output, current_kv = self.self_attn(
            attn_input, 
            attention_mask=attention_mask,
            past_kv=past_kv,
            use_cache=use_cache
        )
        
        if self.use_parallel_residual:
            # For parallel residual, we normalize input once and apply parallel paths
            ffn_output, moe_loss = self.ffn(attn_input)
            output = x + self.dropout(attn_output + ffn_output)
        else:
            # Apply first residual connection
            hidden_states = x + attn_output
            
            # Apply normalization before FFN
            ffn_input = self.ffn_norm(hidden_states)
            
            # Apply FFN and second residual
            ffn_output, moe_loss = self.ffn(ffn_input)
            output = hidden_states + self.dropout(ffn_output)
        
        return output, current_kv, moe_loss

    def forward(self, x, attention_mask=None, past_kv=None, use_cache=False):
        if self.training:
            # Use torch.utils.checkpoint during training for memory efficiency
            output, current_kv, moe_loss = checkpoint(
                self._forward_impl, 
                x, attention_mask, past_kv, use_cache,
                use_reentrant=False
            )
        else:
            output, current_kv, moe_loss = self._forward_impl(
                x, attention_mask, past_kv, use_cache
            )
        
        return output, current_kv, moe_loss

# === Embedding and Positional Encoding ===
class EmbeddingLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Scale embeddings 
        self.emb_scale = math.sqrt(config.hidden_size)

    def forward(self, input_ids):
        embeddings = self.token_embedding(input_ids) * self.emb_scale
        return self.dropout(embeddings)

# === Main Model Class ===
class ChatModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.embed = EmbeddingLayer(config)
        
        # Transformer layers
        self.blocks = nn.ModuleList([
            TransformerBlock(config, layer_idx=i) 
            for i in range(config.num_layers)
        ])
        
        # Final layer normalization
        if config.use_rmsnorm:
            self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        else:
            self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights between embedding and output layer
        self.lm_head.weight = self.embed.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            module.bias.data.zero_() if hasattr(module, 'bias') else None
            module.weight.data.fill_(1.0)

    def forward(
        self, 
        input_ids, 
        attention_mask=None, 
        labels=None, 
        past_key_values=None,
        use_cache=False
    ):
        batch_size, seq_length = input_ids.shape
        
        # Generate causal attention mask
        if attention_mask is not None:
            # Convert attention mask from 0/1 to -inf/0
            attention_mask_float = attention_mask.to(dtype=torch.float32)
            attention_mask_float = (1.0 - attention_mask_float) * -10000.0
            attention_mask_float = attention_mask_float.unsqueeze(1).unsqueeze(2)
        else:
            attention_mask_float = None
        
        # Create position-based causal mask
        if past_key_values is None:
            # Full causal mask when no past keys
            causal_mask = torch.triu(
                torch.full((seq_length, seq_length), float("-inf"), device=input_ids.device), 
                diagonal=1
            )
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        else:
            # Causal mask for current sequence only
            past_length = past_key_values[0][0].size(1)
            causal_mask = torch.zeros(
                (seq_length, past_length + seq_length), 
                device=input_ids.device
            )
            causal_mask = torch.triu(causal_mask, diagonal=1 + past_length)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
            causal_mask = causal_mask * float("-inf")
            
        # Combine masks if needed
        if attention_mask_float is not None:
            combined_mask = attention_mask_float
            if causal_mask is not None:
                combined_mask = combined_mask + causal_mask
        else:
            combined_mask = causal_mask
        
        # Apply embedding
        x = self.embed(input_ids)
        
        # Process through transformer blocks
        moe_losses = 0
        new_key_values = () if use_cache else None
        
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            
            x, current_kv, moe_loss = block(
                x, 
                attention_mask=combined_mask,
                past_kv=past_kv,
                use_cache=use_cache
            )
            
            moe_losses += moe_loss
            
            if use_cache:
                new_key_values += (current_kv,)
        
        # Normalize output
        x = self.norm(x)
        
        # Compute logits
        logits = self.lm_head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Compute cross entropy loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                labels.view(-1), 
                ignore_index=-100
            )
            
            # Add MoE auxiliary loss if applicable
            if self.config.use_moe:
                moe_losses /= len(self.blocks)
                loss += self.config.expert_loss_weight * moe_losses
        
        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": new_key_values,
            "moe_loss": moe_losses / len(self.blocks) if self.config.use_moe else 0.0
        }

    def generate(
        self, 
        input_ids, 
        attention_mask=None,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.2,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        use_cache=True,
        stopping_criteria=None,
        logits_processor=None,
        typical_p=None,
        streaming=False,
        callback=None
    ):
        """Generate text from prompt with advanced decoding methods"""
        
        batch_size, input_seq_len = input_ids.shape
        device = input_ids.device
        
        # Expand inputs for batched generation
        if input_ids.shape[0] != batch_size:
            input_ids = input_ids.expand(batch_size, -1)
            
        if attention_mask is not None and attention_mask.shape[0] != batch_size:
            attention_mask = attention_mask.expand(batch_size, -1)
            
        # Set up stopping criteria
        if stopping_criteria is None:
            stopping_criteria = []
            
        # Default stopping criteria: end token or max length
        generated_tokens = []
        past_key_values = None
        
        # Keep track of token positions for repetition penalty
        if repetition_penalty > 1.0:
            token_history = set()
                
        # Start generation loop
        for i in range(max_new_tokens):
            with torch.no_grad():
                if past_key_values is None or not use_cache:
                    outputs = self(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=use_cache
                    )
                else:
                    # Only process the last token with cached keys/values
                    outputs = self(
                        input_ids=input_ids[:, -1:],
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=use_cache
                    )
                    
                logits = outputs["logits"][:, -1, :]
                past_key_values = outputs.get("past_key_values", None)
                
                # Apply temperature scaling
                if temperature > 0:
                    logits = logits / temperature
                    
                # Apply repetition penalty
                if repetition_penalty > 1.0:
                    for b in range(batch_size):
                        for token_id in token_history:
                            logits[b, token_id] /= repetition_penalty
                            
                # Apply presence and frequency penalties (like in OpenAI API)
                if presence_penalty != 0.0 or frequency_penalty != 0.0:
                    token_counts = Counter(input_ids[0].tolist())
                    for token_id, count in token_counts.items():
                        if presence_penalty != 0.0:
                            logits[0, token_id] -= presence_penalty
                        if frequency_penalty != 0.0:
                            logits[0, token_id] -= count * frequency_penalty
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                    
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Keep the first token above the threshold
                    sorted_indices_to_remove[..., 0] = False
                    
                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=-1, 
                        index=sorted_indices, 
                        src=sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')
                    
                # Apply typical_p sampling if specified
                if typical_p is not None and typical_p < 1.0:
                    temp_logits = logits.clone()
                    # Calculate entropy
                    probs = F.softmax(temp_logits, dim=-1)
                    log_probs = F.log_softmax(temp_logits, dim=-1)
                    entropy = -(probs * log_probs).sum(-1, keepdim=True)
                    
                    # Calculate the expected "surprisal" of each token
                    neg_entropy = -entropy
                    shifted_scores = log_probs - neg_entropy
                    
                    # Apply typical sampling with mass concentrated in the center
                    sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
                    sorted_probs = probs.gather(-1, sorted_indices)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    # Remove tokens below threshold
                    sorted_indices_to_remove = cumulative_probs < (1 - typical_p)
                    indices_to_remove = torch.scatter(
                        torch.zeros_like(logits, dtype=torch.bool),
                        dim=-1,
                        index=sorted_indices,
                        src=sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')
                
                # Convert to probabilities
                probs = F.softmax(logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Add to generated output
                generated_tokens.append(next_token)
                
                # For repetition penalty
                if repetition_penalty > 1.0:
                    token_history.update(next_token[0].tolist())
                    
                # Append new token to input_ids
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Update attention mask if needed
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones_like(next_token)
                    ], dim=-1)
                    
                # Handle callback for streaming
                if streaming and callback is not None:
                    callback(next_token)
                    
                # Check stopping criteria
                for criterion in stopping_criteria:
                    if criterion(input_ids):
                        return input_ids
                        
        return input_ids

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        # Save the model in safetensors format
        model_to_save = self.module if hasattr(self, 'module') else self
        save_file(model_to_save.state_dict(), os.path.join(path, "model.safetensors"))
        # Save the config
        self.config.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, model_path, config=None):
        if config is None:
            config = ChatConfig.from_pretrained(model_path)
        state_dict = load_file(os.path.join(model_path, "model.safetensors"))
        new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        model = cls(config)
        model.load_state_dict(new_state_dict)
        return model

# === Tokenization ===
class ChatTokenizer:
    def __init__(self, vocab_path=None):
        self.vocab = {
            "<|padding|>": 0, "<|unknown|>": 1, "<|user|>": 2,
            "<|think|>": 3, "<|assistant|>": 4, "<|end|>": 5, 
            " ": 6, "\\n": 7, "<|system|>": 8, "<|tool|>": 9
        }
        self.special_tokens = sorted(self.vocab.keys(), key=lambda k: self.vocab[k])
        self.pattern = re.compile(
            f'({"|".join(map(re.escape, self.special_tokens))})'
            r'|([\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF\U00020000-\U0002EBEF])'
            r'|([a-zA-Z]+)' r'|([0-9])' r'|(\s+)' r'|(_)' r'|([^\s])'
            r'|([!@#$%^&*()\-+=\[\]{}\\|;:\'",.<>/?`~])', re.UNICODE)
        if vocab_path and os.path.exists(vocab_path):
            self.load(vocab_path)

    def tokenize(self, text):
        return [m.group() for m in self.pattern.finditer(text)]

    def convert_tokens_to_ids(self, tokens, update_vocab=True):
        return [self.vocab.setdefault(token, len(self.vocab)) if token not in self.vocab and update_vocab
            else self.vocab.get(token, self.vocab["<|unknown|>"]) for token in tokens]

    def build_vocab(self, texts, max_vocab_size=None):
        token_freq = Counter(token for text in texts for token in self.tokenize(text) if token not in self.special_tokens)
        sorted_tokens = sorted(token_freq.items(), key=lambda x: (-x[1], x[0]))
        if max_vocab_size:
            sorted_tokens = sorted_tokens[:max_vocab_size - len(self.special_tokens)]
        self.vocab.update({token: idx + len(self.special_tokens) for idx, (token, _) in enumerate(sorted_tokens)})

    def __call__(self, text, max_length, truncation=True, padding="max_length", update_vocab=False):
        ids = self.convert_tokens_to_ids(self.tokenize(text), update_vocab)
        ids = ids[:max_length] if truncation else ids
        ids += [self.vocab["<|padding|>"]] * (max_length - len(ids))
        mask = [1 if i != self.vocab["<|padding|>"] else 0 for i in ids]
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([mask])}

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        save_json({"vocab": self.vocab}, os.path.join(path, "tokenizer.json"))

    def load(self, path):
        self.vocab = load_json(os.path.join(path, "tokenizer.json"))["vocab"]

    @property
    def reverse_vocab(self):
        return {i: token for token, i in self.vocab.items()}
        
    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
            
        if isinstance(ids[0], list):
            return [self.decode(id_list, skip_special_tokens) for id_list in ids]
            
        tokens = []
        for id in ids:
            token = self.reverse_vocab.get(id, "<|unknown|>")
            if skip_special_tokens and token in self.special_tokens and token not in [" ", "\\n"]:
                continue
            tokens.append(token)
            
        return "".join(tokens)

    @classmethod
    def from_pretrained(cls, path):
        tokenizer = cls()
        tokenizer.load(path)
        return tokenizer
        
# === Dataset Classes ===
class ChatDataset(Dataset):
    def __init__(self, tokenizer, max_length, texts):
        self.data = [text for text in texts if text and len(tokenizer.tokenize(text)) > 1]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.data[idx], self.max_length, update_vocab=False)
        input_ids = encoding["input_ids"].squeeze()
        return {
            "input_ids": input_ids[:-1],
            "attention_mask": encoding["attention_mask"].squeeze()[:-1],
            "labels": input_ids[1:].masked_fill(input_ids[1:] == self.tokenizer.vocab["<|padding|>"], -100)
        }

class JsonlChatDataset(Dataset):
    def __init__(self, tokenizer, max_length, file_path):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load data from JSONL file
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Format the conversation in the expected format
        formatted_text = self._format_conversation(item)
        
        # Tokenize and prepare
        encoding = self.tokenizer(formatted_text, self.max_length, update_vocab=False)
        input_ids = encoding["input_ids"].squeeze()
        
        return {
            "input_ids": input_ids[:-1],
            "attention_mask": encoding["attention_mask"].squeeze()[:-1],
            "labels": input_ids[1:].masked_fill(input_ids[1:] == self.tokenizer.vocab["<|padding|>"], -100)
        }
    
    def _format_conversation(self, item):
        # Different formats can be supported here
        if isinstance(item, dict):
            if "messages" in item:
                # ChatML-like format
                formatted = ""
                for msg in item["messages"]:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    
                    if role == "system":
                        formatted += f"<|system|>{content}"
                    elif role == "user":
                        formatted += f"<|user|>{content}"
                    elif role == "assistant":
                        if "thinking" in msg:
                            formatted += f"<|think|>{msg['thinking']}"
                        formatted += f"<|assistant|>{content}<|end|>"
                    elif role == "tool":
                        formatted += f"<|tool|>{content}"
                
                return formatted
            
            # Simple Q&A format
            elif "input" in item and "output" in item:
                return f"<|user|>{item['input']}<|assistant|>{item['output']}<|end|>"
                
        # Fallback: treat as raw text
        return item