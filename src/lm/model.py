import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from lm.utils import count_params

"""
Dimension symbols:
    B - batch size
    S - sequence length
    D - hidden dimension (n_embd)
    H - number of attention heads (n_head)
    HD - hidden dimension of a single attention head (d // n_head)
    V - size of the vocabulary
"""


class MultiHeadAttention(nn.Module):
    """The multi-head attention module in a decoder block."""

    def __init__(self, n_embd: int, n_head: int, p_dropout: float = 0.1):
        super().__init__()
        """Initialize the modules used by multi-head attention."""

        self.n_head = n_head
        attn_hidden_dim = n_embd // n_head

        self.q_attn = nn.Linear(n_embd, n_embd)
        self.k_attn = nn.Linear(n_embd, n_embd)
        self.v_attn = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(p_dropout)

        scale_factor = 1 / torch.sqrt(torch.tensor(attn_hidden_dim))
        self.register_buffer("scale_factor", scale_factor)

    def q_kT_v(
        self, x: torch.FloatTensor
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Project the hidden states to q, kT, v prior to computing attention.

        Args:
            x: embeddings or hidden states (B x S x D) from the previous decoder block

        Returns:
            q: The query vector used by multi-head attention (B x H x S x HD)
            kT: The transpose of the key vector used by multi-head attention (B x H x HD x S)
            v: The value vector used by multi-head attention (B x H x S x HD)
        """
        q = rearrange(self.q_attn(x), "b s (h hd) -> b h s hd", h=self.n_head)
        k = rearrange(self.k_attn(x), "b s (h hd) -> b h s hd", h=self.n_head)
        kT = rearrange(k, "b h s hd -> b h hd s")
        v = rearrange(self.v_attn(x), "b s (h hd) -> b h s hd", h=self.n_head)

        return q, kT, v

    def self_attention(
        self,
        q: torch.FloatTensor,
        kT: torch.FloatTensor,
        v: torch.FloatTensor,
        attention_mask: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        """Compute multi-head attention over the inputs.

        Args:
            q: The query vector used by multi-head attention (B x H x S x HD)
            kT: The transpose of the key vector used by multi-head attention (B x H x HD x S)
            v: The value vector used by multi-head attention (B x H x S x HD)
            attention_mask (optional): Mask indicating tokens that shouldn't
              be included in self-attention (B x S). 1 stands for a token that is
              included, and 0 stands for a token that isn't.

        Returns:
            attn: Outputs of applying multi-head attention to the inputs (B x S x D)
        """

        # compute the attention weights using q and kT
        qkT = q @ kT
        unmasked_attn_logits = qkT * self.scale_factor

        """
        In decoder models, attention logits are masked such that computation at
        each position does not involve embeddings / hidden states of future
        positions.

        This boolean mask should have shape (S x S) and has value True iff
        position i is allowed to attend to position j (i.e., j <= i).

        Example (S = 5):
        causal_mask = tensor([
         [ True, False, False, False, False],
         [ True,  True, False, False, False],
         [ True,  True,  True, False, False],
         [ True,  True,  True,  True, False],
         [ True,  True,  True,  True,  True]
        ])
        
        Note that `causal mask` needs to be on the same device as the input
        tensors (q, kT, v). You can move a tensor to the right device by calling
        `tensor.to(q.device)`.

        Hint: torch.triu or torch.tril
        """
        s = q.shape[-2]
        causal_mask = torch.tril(torch.ones(s, s, dtype=torch.bool)).to(q.device)

        """
        Sometimes, we want to pad the input sequences so that they have the same
        length and can fit into the same batch. These padding tokens should not
        have any effect on the output of self-attention. To achieve this, we
        need to mask out the logits that correspond to those tokens.

        Example (B = 2, S = 5):
        causal_mask = tensor([
         [ True, False, False, False, False],
         [ True,  True, False, False, False],
         [ True,  True,  True, False, False],
         [ True,  True,  True,  True, False],
         [ True,  True,  True,  True,  True]
        ])

        attention_mask = tensor([
         [0., 0., 1., 1., 1.],
         [1., 1., 1., 1., 1.]
        ])

        mask = tensor([
        [[[False, False, False, False, False],
          [False, False, False, False, False],
          [False, False,  True, False, False],
          [False, False,  True,  True, False],
          [False, False,  True,  True,  True]]],

        [[[ True, False, False, False, False],
          [ True,  True, False, False, False],
          [ True,  True,  True, False, False],
          [ True,  True,  True,  True, False],
          [ True,  True,  True,  True,  True]]]
        ])

        Note that `mask` needs to be on the same device as the input tensors
        q, kT and v.
        """

        if attention_mask is None:
            mask = causal_mask
        else:
            mask = causal_mask[None, None, :, :] & attention_mask[:, None, None, :].bool().to(
                q.device
            )

        """
        Fill unmasked_attn_logits with float_min wherever causal mask has value False.

        Hint: torch.masked_fill
        """
        float_min = torch.finfo(q.dtype).min
        attn_logits = unmasked_attn_logits.masked_fill(~mask, float_min)
        attn_weights = F.softmax(attn_logits, dim=-1)
        # rows with no valid position (a fully-padded query) still get a finite
        # uniform softmax above; zero them out to match SDPA's convention of
        # outputting 0 (rather than an arbitrary average) for such positions
        attn_weights = attn_weights * mask.any(dim=-1, keepdim=True)
        attn_weights = self.dropout(attn_weights)

        # scale value by the attention weights.
        attn = attn_weights @ v
        attn = rearrange(attn, "b h s hd -> b s (h hd)")

        return attn

    def projection(self, attn: torch.FloatTensor) -> torch.FloatTensor:
        """Apply a dropout and a linear projection to outputs of attention"""
        return self.dropout(self.proj(attn))

    def forward(
        self, x: torch.FloatTensor, attention_mask: torch.FloatTensor | None = None
    ) -> torch.FloatTensor:
        """A full forward pass of the multi-head attention module.

        Args:
            x: embeddings or hidden states (B x S x D) from the previous decoder block

        Returns:
            y: outputs (B x S x D) of the multi-head attention module
        """

        q, kT, v = self.q_kT_v(x)
        attn = self.self_attention(q, kT, v, attention_mask)
        y = self.projection(attn)
        return y


class FeedForward(nn.Module):
    """The feedforward attention module in a decoder block."""

    def __init__(self, n_embd: int, p_dropout: float = 0.1):
        """Initialize the modules used by feedforward."""
        super().__init__()

        middle_dim = 4 * n_embd  # stick to what GPT-2 does
        self.linear_in = nn.Linear(n_embd, middle_dim)
        self.linear_out = nn.Linear(middle_dim, n_embd)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """A full forward pass of the feedforward module.

        Args:
            x: outputs (B x S x D) of the first Add & Norm operation

        Returns:
            z: outputs (B x S x D) of the feedforward module

        Different from what you saw in class which uses ReLU as the activation,
        we are going to follow GPT-2 which uses GeLU. You should also apply
        self.dropout to the output.
        """

        y = F.gelu(self.linear_in(x))
        z = self.dropout(self.linear_out(y))
        return z
