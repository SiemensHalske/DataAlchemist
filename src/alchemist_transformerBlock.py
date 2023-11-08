import torch.nn as nn
import torch


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, num_embeddings):
        super(TransformerBlock, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings, hidden_size)

        # MultiheadAttention layer
        self.attention = nn.MultiheadAttention(
            hidden_size, num_attention_heads)

        # Feed-forward layer
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        # Normalization layers
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        # Dropout
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input):
        """Forward pass of the transformer block.

        Args:
            input: A tensor of shape [seq_len, batch_size] containing the input sequence.

        Returns:
            A tensor of shape [seq_len, batch_size, hidden_size] containing the output sequence.

        """

        embedded_input = self.embedding(
            input) + self.positional_encodings(input)
        attention_output, _ = self.attention(
            embedded_input, embedded_input, embedded_input)

        x = self.norm1(embedded_input + self.dropout(attention_output))
        feed_forward_output = self.feed_forward(x)
        output = self.norm2(x + self.dropout(feed_forward_output))

        return output

    def positional_encodings(self, input):
        """Calculates sinusoidal positional encoding for the input sequence.

        Args:
            input: A tensor of shape [seq_len, batch_size] containing the input sequence.

        Returns:
            A tensor of shape [seq_len, batch_size, embedding_dim] containing the positional
            encoding for the input sequence.
        """

        seq_len, batch_size = input.size()
        hidden_size = self.embedding.weight.size(1)

        position_encodings = torch.zeros(seq_len, batch_size, hidden_size)

        for position in range(seq_len):
            for i in range(hidden_size // 2):
                position_encodings[position][:, 2 * i] = torch.sin(
                    position / (10000 ** (2 * i / hidden_size)))
                position_encodings[position][:, 2 * i +
                                             1] = torch.cos(position / (10000 ** (2 * i / hidden_size)))

        return position_encodings
