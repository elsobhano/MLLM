import torch
from transformers import MBartModel
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"

        # Linear layers for values, keys, and queries
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, padding_mask, causal_mask):
        """
        Forward pass for multi-head self-attention with padding and causal masks.
        
        Args:
            values: Values tensor. Shape: (N, seq_length, embed_size)
            keys: Keys tensor. Shape: (N, seq_length, embed_size)
            query: Query tensor. Shape: (N, seq_length, embed_size)
            padding_mask: Padding mask. Shape: (N, seq_length)
            causal_mask: Causal mask. Shape: (seq_length, seq_length)
        """
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # Linear transformations for each head
        values = self.values(values)  # (N, seq_length, heads, head_dim)
        keys = self.keys(keys)        # (N, seq_length, heads, head_dim)
        queries = self.queries(queries)  # (N, seq_length, heads, head_dim)

        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # (N, heads, query_len, key_len)

        # Apply padding mask
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)  # (N, 1, 1, seq_length)

        # Apply causal mask
        if causal_mask is not None:
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_length, seq_length)
            combined_mask = padding_mask * causal_mask
            diag_mask = torch.eye(query_len, dtype=torch.bool, device=energy.device).unsqueeze(0).unsqueeze(0)
            combined_mask = combined_mask | diag_mask            
        else:
            combined_mask = padding_mask

        combined_mask = combined_mask.to(energy.dtype)

        energy = energy.masked_fill(combined_mask == 0, float("-inf"))

        # Softmax over the last dimension (key_len)
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        # Apply attention to values
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])  # (N, query_len, heads, head_dim)
        out = out.reshape(N, query_len, self.heads * self.head_dim)  # Concatenate heads

        # Final linear layer
        out = self.fc_out(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_size)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class CausalDecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, ff_dim, dropout):
        super(CausalDecoderBlock, self).__init__()
        self.self_attention = MultiHeadSelfAttention(embed_size, heads)
        self.cross_attention = MultiHeadSelfAttention(embed_size, heads)
        self.feed_forward = FeedForward(embed_size, ff_dim)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask, causal_mask):
        # Self-Attention with both masks
        self_attention = self.self_attention(x, x, x, tgt_mask, causal_mask)
        x = self.norm1(x + self.dropout(self_attention))

        # Cross-Attention (attends to encoder output)
        cross_attention = self.cross_attention(encoder_output, encoder_output, x, src_mask, None)  # No causal mask for cross-attention
        x = self.norm2(x + self.dropout(cross_attention))

        # Feed-Forward
        ff = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff))

        return x

class DecoderWithMBartEmbeddings(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, ff_dim, dropout, max_length, device, mbart_model_name="facebook/mbart-large-cc25"):
        super(DecoderWithMBartEmbeddings, self).__init__()
        self.device = device

        # Load only the MBart embedding layer
        self.mbart_embeddings = MBartModel.from_pretrained(mbart_model_name).get_input_embeddings()

        # Freeze the MBart embeddings if you don't want to fine-tune them
        for param in self.mbart_embeddings.parameters():
            param.requires_grad = False

        # Ensure embed_size matches MBart's embedding size
        self.embed_size = self.mbart_embeddings.embedding_dim
        assert embed_size == self.embed_size, "Embed size must match MBart's embedding size"

        # Positional Embedding (since MBart doesn't provide positional embeddings separately)
        self.position_embedding = nn.Embedding(max_length, self.embed_size)

        # Decoder layers
        self.layers = nn.ModuleList([
            CausalDecoderBlock(self.embed_size, heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(self.embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, tgt_mask, src_mask):
        """
        Forward pass for the decoder.
        
        Args:
            x: Token IDs (output of tokenizer, handled in data loader). Shape: (N, seq_length)
            encoder_output: Output from the encoder. Shape: (N, seq_length, embed_size)
            src_mask: Mask for encoder output. Shape: (N, seq_length)
            trg_mask: Mask for decoder input (causal mask). Shape: (N, seq_length, seq_length)
        """
        N, seq_length = x.shape

        # Convert input tokens to MBart embeddings
        with torch.no_grad():  # No gradient computation for MBart embeddings
            mbart_embeddings = self.mbart_embeddings(x)  # (N, seq_length, embed_size)

        # Add positional embeddings
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(mbart_embeddings + self.position_embedding(positions))

        causal_mask = (torch.triu(torch.ones(seq_length, seq_length), 1) == 0.0)
        causal_mask = causal_mask.to(self.device)
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask, causal_mask)

        # Final output projection
        out = self.fc_out(x)
        return out