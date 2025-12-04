# ML_Assignments_Hw5
Name: Babithji Repalle
700780633


Part B — Coding
Q1. Compute Scaled Dot-Product Attention (Python)
Write a Python function to compute the scaled dot-product attention given query Q, key K, and value Vmatrices.
	Use NumPy for matrix operations.
	Normalize scores using softmax.
	Return both attention weights and the resulting context vector.
Hint:
"Attention"(Q,K,V)="softmax"((QK^T)/√(d_k ))V
Here’s a Python function to compute the scaled dot-product attention using NumPy. This function takes matrices for queries Q, keys K, and values V as inputs, calculates the attention weights, and returns both the attention weights and the resulting context vector.
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """
    Compute the scaled dot-product attention.
    
    Parameters:
    Q : numpy.ndarray
        Query matrix of shape (num_queries, d_k)
    K : numpy.ndarray
        Key matrix of shape (num_keys, d_k)
    V : numpy.ndarray
        Value matrix of shape (num_keys, d_v)
    
    Returns:
    attention_weights : numpy.ndarray
        The attention weights of shape (num_queries, num_keys)
    context_vector : numpy.ndarray
        The resulting context vector of shape (num_queries, d_v)
    """
    # Calculate the dimension of keys
    d_k = K.shape[1]
    
    # Compute the dot product of Q and K^T
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    
    # Apply softmax to get attention weights
    attention_weights = np.exp(scores - np.max(scores, axis=1, keepdims=True))  # for numerical stability
    attention_weights /= np.sum(attention_weights, axis=1, keepdims=True)
    
    # Compute the context vector as the weighted sum of values
    context_vector = np.dot(attention_weights, V)
    
    return attention_weights, context_vector

# Example usage:
Q = np.array([[1, 0, 1], [0, 1, 0]])  # Example query
K = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]])  # Example keys
V = np.array([[1, 2], [3, 4], [5, 6]])  # Example values

attention_weights, context_vector = scaled_dot_product_attention(Q, K, V)

print("Attention Weights:\n", attention_weights)
print("Context Vector:\n", context_vector)

Explanation:
	Input Matrices: The function expects three matrices: QQ (queries), KK (keys), and VV (values).
	Scaling: The dot product of QQ and KTKT is scaled by the square root of the dimension of the keys dkdk.
	Softmax: The softmax function is applied to the scores to obtain the attention weights, ensuring they sum to 1.
	Context Vector: The final context vector is computed by multiplying the attention weights with the value matrix VV.
Example Usage:
You can modify the example input matrices QQ, KK, and VV as needed to test different scenarios.




Q2. Implement Simple Transformer Encoder Block (PyTorch)
Implement a simplified transformer encoder block in PyTorch with the following components:
	Multi-head self-attention layer
	Feed-forward network (2 linear layers with ReLU)
	Add & Norm layers
Sub-tasks:
a) Initialize dimensions d_model=128,h=8.
b) Add residual connections and layer normalization.
c) Verify the output shape for a batch of 32 sentences, each with 10 tokens.


Here’s an implementation of a simplified transformer encoder block in PyTorch, incorporating multi-head self-attention, a feed-forward network, and residual connections with layer normalization.
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, h):
        super(MultiHeadSelfAttention, self).__init__()
        self.h = h
        self.d_k = d_model // h
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        
        # Linear transformations
        Q = self.linear_q(x)  # (batch_size, seq_len, d_model)
        K = self.linear_k(x)  # (batch_size, seq_len, d_model)
        V = self.linear_v(x)  # (batch_size, seq_len, d_model)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # (batch_size, h, seq_len, d_k)
        K = K.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # (batch_size, h, seq_len, d_k)
        V = V.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # (batch_size, h, seq_len, d_k)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # (batch_size, h, seq_len, seq_len)
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, h, seq_len, seq_len)
        output = torch.matmul(attention_weights, V)  # (batch_size, h, seq_len, d_k)

        # Concatenate heads and pass through final linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)  # (batch_size, seq_len, d_model)
        return self.linear_out(output)

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, h, d_ff):
        super(TransformerEncoderBlock, self).__init__()
        self.self_attention = MultiHeadSelfAttention(d_model, h)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Multi-head self-attention with residual connection
        attn_output = self.self_attention(x)
        x = self.layer_norm1(x + attn_output)  # Residual connection and layer norm

        # Feed-forward network with residual connection
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)  # Residual connection and layer norm
        
        return x

# Example usage
if __name__ == "__main__":
    d_model = 128  # Dimension of model
    h = 8         # Number of heads
    d_ff = 512    # Dimension of feed-forward network
    batch_size = 32
    seq_len = 10

    # Create a random input tensor with shape (batch_size, seq_len, d_model)
    input_tensor = torch.rand(batch_size, seq_len, d_model)

    # Instantiate the transformer encoder block
    transformer_encoder = TransformerEncoderBlock(d_model, h, d_ff)

    # Forward pass
    output = transformer_encoder(input_tensor)

    # Verify output shape
    print("Output shape:", output.shape)  # Should be (batch_size, seq_len, d_model)
Explanation:
	Multi-Head Self-Attention: This class implements the multi-head self-attention mechanism, which includes linear transformations for queries, keys, and values, followed by scaled dot-product attention.
	Feed-Forward Network: This class represents a simple feed-forward network consisting of two linear layers with a ReLU activation in between.
	Transformer Encoder Block: This class combines the multi-head self-attention and feed-forward network, adding residual connections and layer normalization after each component.
	Output Verification: The output shape is verified to ensure it matches the expected dimensions of (batch_size, seq_len, d_model). In this case, it should output (32, 10, 128).
Example Usage:
You can run the provided code to instantiate the transformer encoder block and verify the output shape with random input data. Adjust the dimensions and parameters as needed for different configurations.




