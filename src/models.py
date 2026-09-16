"""
Simple Transformer model for autoregressive prediction tasks.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension to implement RoPE."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE) applied to Q and K in attention.

    Unlike additive encodings, RoPE rotates query/key vectors by position-
    dependent angles, encoding relative position implicitly in dot products.

    Args:
        head_dim: Dimension of each attention head (must be even)
        max_seq_len: Maximum sequence length to cache
        base: Base for the geometric frequency sequence (default 10000)
    """

    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)          # (seq_len, head_dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1)        # (seq_len, head_dim)
        # (1, 1, seq_len, head_dim) for broadcasting over batch and heads
        self.register_buffer('cos_cached', emb.cos()[None, None])
        self.register_buffer('sin_cached', emb.sin()[None, None])

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:, :, :seq_len], self.sin_cached[:, :, :seq_len]


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with einops for clarity."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 rotary_emb: RotaryPositionalEmbedding = None):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.rotary_emb = rotary_emb

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b n (three h d) -> three b h n d',
                           three=3, h=self.n_heads)

        # Apply rotary embeddings to Q and K (not V)
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(seq_len)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin

        # Attention scores with causal mask
        attn = einsum(q, k, 'b h i d, b h j d -> b h i j') * self.scale

        # Causal mask: prevent attending to future tokens
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        attn = attn.masked_fill(causal_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.out_proj(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int = None,
                 dropout: float = 0.1, attention_only: bool = False,
                 rotary_emb: RotaryPositionalEmbedding = None):
        super().__init__()
        self.attention_only = attention_only
        self.attention_only = attention_only
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, rotary_emb=rotary_emb)
        if not attention_only:
            self.ln2 = nn.LayerNorm(d_model)
            self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        if not self.attention_only:
            x = x + self.ff(self.ln2(x))
        if not self.attention_only:
            x = x + self.ff(self.ln2(x))
        return x


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encodings."""

    def __init__(self, d_model: int, max_seq_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:, :x.size(1)]


class Transformer(nn.Module):
    """
    Autoregressive Transformer model.

    Args:
        vocab_size: Size of the vocabulary
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer blocks
        max_seq_len: Maximum sequence length
        d_ff: Feed-forward hidden dimension (default: 4 * d_model)
        dropout: Dropout rate
        use_learned_pe: Use learned positional embeddings instead of sinusoidal
        attention_only: If True, remove FFN sublayers (attention-only transformer)
        use_rotary_pe: Use rotary positional embeddings (RoPE). Mutually
            exclusive with use_learned_pe; no additive PE is added to tokens.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        max_seq_len: int = 512,
        d_ff: int = None,
        dropout: float = 0.1,
        use_learned_pe: bool = False,
        attention_only: bool = False,
        use_rotary_pe: bool = False,
    ):
        super().__init__()
        assert not (use_learned_pe and use_rotary_pe), \
            "use_learned_pe and use_rotary_pe are mutually exclusive"
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.attention_only = attention_only
        self.use_learned_pe = use_learned_pe
        self.use_rotary_pe = use_rotary_pe

        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        if use_rotary_pe:
            # RoPE is applied inside attention; no additive PE on token embeddings.
            # One shared RoPE module is passed to every attention layer.
            head_dim = d_model // n_heads
            rotary_emb = RotaryPositionalEmbedding(head_dim, max_seq_len)
            self.rotary_emb = rotary_emb
            self.pos_enc = None
        elif use_learned_pe:
            self.pos_enc = nn.Embedding(max_seq_len, d_model)
            rotary_emb = None
        else:
            self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len)
            rotary_emb = None

        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout,
                             attention_only=attention_only, rotary_emb=rotary_emb)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)

        # Output projection (tied with token embeddings by default)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # self.lm_head.weight = self.token_emb.weight  # Weight tying
        # self.lm_head.weight = self.token_emb.weight  # Weight tying

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input token indices (batch, seq_len)

        Returns:
            Logits over vocabulary (batch, seq_len, vocab_size)
        """
        batch, seq_len = x.shape
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"

        # Token + positional embeddings
        tok_emb = self.token_emb(x)

        if self.use_rotary_pe:
            # RoPE is applied inside each attention layer; no additive PE here.
            x = tok_emb
        elif self.use_learned_pe:
            positions = torch.arange(seq_len, device=x.device)
            pos_emb = self.pos_enc(positions)
            x = tok_emb + pos_emb
        else:
            x = self.pos_enc(tok_emb)

        x = self.dropout(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        # Project to vocabulary
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            prompt: Starting tokens (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top k tokens

        Returns:
            Generated sequence including prompt (batch, seq_len + max_new_tokens)
        """
        self.eval()
        x = prompt

        for _ in range(max_new_tokens):
            # Crop to max sequence length
            x_cond = x if x.size(1) <= self.max_seq_len else x[:, -self.max_seq_len:]

            # Get predictions
            logits = self(x_cond)
            logits = logits[:, -1, :] / temperature

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            x = torch.cat([x, next_token], dim=1)

        return x

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerClassifier(nn.Module):
    """Transformer encoder with an n-class classification head at the last token position.

    forward() returns logits of shape (batch, seq_len, n_classes) so it is compatible with
    the existing train() function, which picks logits[:, -1, :] for scalar-label batches.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 32,
        d_ff: int = None,
        dropout: float = 0.1,
        n_classes: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.classifier_head = nn.Linear(d_model, n_classes)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) long for hard tokens, or
               (batch, seq_len, vocab_size) float for soft tokens (e.g. blur mode)
        Returns:
            logits: (batch, seq_len, 2)
        """
        if x.dim() == 3:  # soft inputs: weighted combination of embedding rows
            tok_emb = x @ self.token_emb.weight
        else:              # hard inputs: standard integer lookup
            tok_emb = self.token_emb(x)
        x = self.pos_enc(tok_emb)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.classifier_head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ContextGatedLinearNetwork(nn.Module):
    """Two-layer gated linear network where context tokens gate the task representation.

    Replicates the following JAX functional model in PyTorch:

        x_task    = x[:-n_contexts]
        x_context = x[-n_contexts:]
        h = (W @ x_task + b) * (G @ x_context)           # gated hidden layer
        y = A @ h + c                                    # output layer

    I got rid of the softmax so that not every node has to be active.

    Args:
        input_dim:  Total input size (task + context dimensions).
        hidden_dim: Width of the gated hidden layer.
        output_dim: Output size (default 1 for scalar regression).
        n_contexts: Number of trailing input dimensions used as context (default 2).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        n_contexts: int = 2,
        bias: bool = False,
    ):
        super().__init__()
        task_dim = input_dim - n_contexts
        assert task_dim > 0, "input_dim must be greater than n_contexts"
        self.n_contexts = n_contexts

        self.task_linear = nn.Linear(task_dim, hidden_dim, bias=bias)     # W
        self.gate_linear = nn.Linear(n_contexts, hidden_dim, bias=False)  # G (no bias)
        self.output_linear = nn.Linear(hidden_dim, output_dim, bias=bias) # A

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., input_dim) float tensor
        Returns:
            (...) tensor — last dim squeezed when output_dim == 1
        """
        x_task    = x[..., :-self.n_contexts]
        x_context = x[..., -self.n_contexts:]

        h = self.task_linear(x_task) * self.gate_linear(x_context)
        out = self.output_linear(h)
        return out.squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ContextGatedLinearSoftmaxNetwork(ContextGatedLinearNetwork):
    """Variant of ContextGatedLinearNetwork where the gate weights are softmax-normalized.

    The gate weight matrix rows are passed through softmax before the linear transform,
    so the effective gate weights always form a convex combination of the context inputs
    (non-negative, summing to 1) by construction — no regularization needed.

        gate_weights = softmax(G, dim=-1)          # (hidden_dim, n_contexts)
        h = (W @ x_task) * (gate_weights @ x_context)

    The stored parameters G are unconstrained logits; softmax is applied at forward time.
    All other behaviour (bias, init, parameter count) is identical to the base class.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_task    = x[..., :-self.n_contexts]
        x_context = x[..., -self.n_contexts:]

        gate_weights = F.softmax(self.gate_linear.weight, dim=-1)  # (hidden_dim, n_contexts)
        gate = x_context @ gate_weights.T                          # (..., hidden_dim)
        h = self.task_linear(x_task) * gate
        out = self.output_linear(h)
        return out.squeeze(-1)


class FixedContextGatedLinearNetwork(ContextGatedLinearNetwork):
    """Variant of ContextGatedLinearNetwork where the gate weights are fixed (non-trainable).

    Each row of the gate weight matrix is randomly assigned to one of [1, 0], [0, 1],
    or [0.5, 0.5] at initialisation time.
    These weights are registered as a buffer so they are not updated by the optimiser
    but are still moved to the correct device with the model.

    All other behaviour (bias, forward pass, parameter count) is identical to the base class.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        n_contexts: int = 2,
        bias: bool = False,
    ):
        super().__init__(input_dim, hidden_dim, output_dim, n_contexts, bias)

        # Build fixed gate weights: each row randomly drawn from {[1,0], [0,1], [0.5,0.5]}
        candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        row_indices = torch.randint(0, len(candidates), (hidden_dim,))
        fixed_weight = candidates[row_indices]

        # Replace trainable parameter with a non-trainable buffer
        del self.gate_linear
        self.register_buffer("gate_weight", fixed_weight)  # (hidden_dim, n_contexts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_task    = x[..., :-self.n_contexts]
        x_context = x[..., -self.n_contexts:]

        gate = x_context @ self.gate_weight.T   # (..., hidden_dim)
        h = self.task_linear(x_task) * gate
        out = self.output_linear(h)
        return out.squeeze(-1)


class FixedContextGatedNetwork(FixedContextGatedLinearNetwork):
    """Variant of FixedContextGatedLinearNetwork with an optional nonlinear activation.

    Applies an activation function (default: ReLU) to the task linear projection
    before the gate is applied:

        h = activation(W @ x_task + b) * gate(x_context)

    Args:
        bias: Whether to add bias to task and output linear layers (default: True — bias
              is essential with ReLU to prevent dead neurons on zero-input patterns).
        activation: Nonlinearity applied after the task linear layer (default: nn.ReLU()).
                    Pass nn.Identity() to recover linear behaviour.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        n_contexts: int = 2,
        bias: bool = True,
        activation: nn.Module = None,
    ):
        super().__init__(input_dim, hidden_dim, output_dim, n_contexts, bias)
        self.activation = activation if activation is not None else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_task    = x[..., :-self.n_contexts]
        x_context = x[..., -self.n_contexts:]

        gate = x_context @ self.gate_weight.T           # (..., hidden_dim)
        h = self.activation(self.task_linear(x_task)) * gate
        out = self.output_linear(h)
        return out.squeeze(-1)


class FullyConnectedNetwork(nn.Module):
    """MLP with configurable hidden layer sizes and activation.

    Args:
        input_dim:   Input feature size.
        hidden_dims: List of hidden layer widths. An empty list gives a single
                     linear layer (no nonlinearity). e.g. [128, 64] produces
                     Linear(input) -> ReLU -> Linear(128) -> ReLU -> Linear(64) -> Linear(output).
        output_dim:  Output size (default 1 for scalar regression).
        activation:  Activation applied between hidden layers (default ReLU).
                     Pass nn.Identity() for a fully linear network.
        bias:        Whether to add bias terms to all linear layers (default True).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int = 1,
        activation: nn.Module = None,
        bias: bool = True,
    ):
        super().__init__()
        activation = activation or nn.ReLU()

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h, bias=bias), activation]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim, bias=bias))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., input_dim) float tensor
        Returns:
            (...) tensor — last dim squeezed when output_dim == 1
        """
        return self.net(x).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimpleAttentionNetwork(nn.Module):
    """Simplified attention network: an abstraction of a single-layer Transformer.

    Architecture:
      - Token embeddings of size d_model concatenated with one-hot positional
        encodings of size len_sequence, giving d_input = d_model + len_sequence.
        When no_learned_embedding=True, a fixed one-hot token encoding of size
        vocab_size is used instead, giving d_input = vocab_size + len_sequence.
      - Single attention head with softmax attention weights
      - Readout: linear projection of the representation at position 0

    Args:
        vocab_size:           Size of the vocabulary.
        d_model:              Token embedding dimension (ignored when no_learned_embedding=True).
        len_sequence:         Fixed sequence length (also sets positional encoding size).
        output_dim:           Output dimension (defaults to vocab_size).
        attention_type:       Either ``"softmax"`` (default) or ``"linear"`` (raw dot-product
                              without softmax normalization).
        no_learned_embedding: If True, replace the learned token embedding with a fixed
                              one-hot encoding (d_input = vocab_size + len_sequence).
        no_learned_Q:         If True, fix Q projection to the identity (not trained).
        no_learned_K:         If True, fix K projection to the identity (not trained).
        no_learned_V:         If True, fix V projection to the identity (not trained).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        len_sequence: int,
        output_dim: int = None,
        attention_type: str = "softmax",
        no_learned_embedding: bool = False,
        no_learned_Q: bool = False,
        no_learned_K: bool = False,
        no_learned_V: bool = False,
    ):
        super().__init__()
        self.len_sequence = len_sequence
        self.no_learned_embedding = no_learned_embedding
        output_dim = output_dim or vocab_size

        assert attention_type in ("softmax", "linear"), \
            f"attention_type must be 'softmax' or 'linear', got '{attention_type}'"
        self.attention_type = attention_type

        if no_learned_embedding:
            self.register_buffer('tok_enc', torch.eye(vocab_size))  # fixed one-hot token enc
            d_input = vocab_size + len_sequence
        else:
            self.token_emb = nn.Embedding(vocab_size, d_model)
            d_input = d_model + len_sequence
        self.register_buffer('pos_enc', torch.eye(len_sequence))  # fixed one-hot PE

        self.scale = d_input ** -0.5
        self.q_proj = self._make_proj(d_input, fixed=no_learned_Q)
        self.k_proj = self._make_proj(d_input, fixed=no_learned_K)
        self.v_proj = self._make_proj(d_input, fixed=no_learned_V)

        self.readout = nn.Linear(d_input, output_dim)

    @staticmethod
    def _make_proj(d_input: int, fixed: bool) -> nn.Linear:
        proj = nn.Linear(d_input, d_input, bias=False)
        if fixed:
            nn.init.eye_(proj.weight)
            proj.weight.requires_grad_(False)
        return proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) token indices — seq_len must equal len_sequence
        Returns:
            logits: (batch, output_dim)
        """
        batch, seq_len = x.shape
        assert seq_len == self.len_sequence, \
            f"Expected seq_len={self.len_sequence}, got {seq_len}"

        # Token embeddings + one-hot positional encodings
        if self.no_learned_embedding:
            tok_emb = self.tok_enc[x]                                    # (batch, seq_len, vocab_size)
        else:
            tok_emb = self.token_emb(x)                                  # (batch, seq_len, d_model)
        pos = self.pos_enc.unsqueeze(0).expand(batch, -1, -1)           # (batch, seq_len, seq_len)
        h = torch.cat([tok_emb, pos], dim=-1)                           # (batch, seq_len, d_input)

        # Single attention head with sigmoid attention
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        attn = einsum(q, k, 'b i d, b j d -> b i j') * self.scale       # (batch, seq_len, seq_len)
        if self.attention_type == "softmax":
            attn = F.softmax(attn, dim=-1)

        out = einsum(attn, v, 'b i j, b j d -> b i d')                  # (batch, seq_len, d_input)

        # Readout at position 0
        return self.readout(out[:, 0, :]).squeeze(-1)                    # (batch,) or (batch, output_dim)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SharedCurriculaNetwork(nn.Module):
    """Two-task network with separate and shared hidden representations.

    Each task has its own input projection (xA or xB) plus a shared projection
    (xS).  Task outputs are:

        YA = AA(HA) + SA(HS)
        YB = BB(HB) + SB(HS)

    where HA = xA(x), HB = xB(x), HS = xS(x).

    forward() returns [YA; YB] concatenated along dim 0, matching the original
    base_model() convention — output shape is (2 * batch,) for output_dim == 1.

    Args:
        input_dim:  Number of input features.
        hidden_dim: Width of the hidden layer (shared across all projections).
        output_dim: Output size per task (default 1).
        bias:       Whether to add bias to all linear layers (default False).
    """

    ACTIVATIONS = {
        "relu":    nn.functional.relu,
        "tanh":    torch.tanh,
        "sigmoid": torch.sigmoid,
        "gelu":    nn.functional.gelu,
        None:      None,
    }

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        bias: bool = False,
        activation: str | None = None,
    ):
        super().__init__()
        if activation not in self.ACTIVATIONS:
            raise ValueError(f"activation must be one of {list(self.ACTIVATIONS)}, got {activation!r}")
        self.xA = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.xB = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.xS = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.AA = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.SA = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.SB = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.BB = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.output_dim = output_dim
        self.act = self.ACTIVATIONS[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., input_dim) float tensor
        Returns:
            Y: (2 * batch,) for output_dim == 1, or (2 * batch, output_dim) otherwise —
               task A and task B outputs concatenated along dim 0.
        """
        HA = self.xA(x)
        HB = self.xB(x)
        HS = self.xS(x)

        if self.act is not None:
            HA = self.act(HA)
            HB = self.act(HB)
            HS = self.act(HS)

        YA = self.AA(HA) + self.SA(HS)
        YB = self.BB(HB) + self.SB(HS)

        return torch.cat((YA, YB), dim=0).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)