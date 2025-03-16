import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def scaled_dot_product(q, k, v, mask=None):
    batch_size, num_heads, query_length, head_dim = q.size()
    _, _, key_length, _ = k.size()

    # Check if the query and key lengths are compatible for dot product
    if query_length != key_length:
        raise ValueError("Query length and Key length must be the same for dot product.")

    # Compute the dot product of q and k
    dot_product = torch.matmul(q, k.transpose(-1, -2))

    # Scale by square root of head dimension
    scaled_dot_product = dot_product / math.sqrt(head_dim)

    # Apply mask if provided
    if mask is not None:
        # Expand the mask tensor to match the shape of the scaled dot product tensor
        mask = mask.unsqueeze(1).unsqueeze(1)
        scaled_dot_product = scaled_dot_product.masked_fill(mask == 0, float('-inf'))

    # Apply softmax to obtain attention weights
    attention_weights = F.softmax(scaled_dot_product, dim=-1)

    # Compute the weighted sum of v based on attention weights
    attended_values = torch.matmul(attention_weights, v)

    return attended_values, attention_weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length=32):
        super().__init__()
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        self.device = get_device()  # Use the get_device function to determine the device

    def forward(self, batch_size, device=None):
        if device is None:
            device = self.device
        
        positions = self._get_positions(batch_size, self.max_sequence_length, device)
        print("Positions tensor:", positions)  # Print positions tensor
        print("Positions tensor shape:", positions.shape)
        position_encodings = self._compute_positional_encodings(positions)
        print("Position encodings tensor:", position_encodings)  # Print position_encodings tensor
        return position_encodings
    
    def _get_positions(self, batch_size, max_sequence_length, device):
        positions = torch.arange(max_sequence_length).unsqueeze(0).expand(batch_size, -1)
        return positions

    def _compute_positional_encodings(self, positions):
        position_encodings = torch.zeros_like(positions, dtype=torch.float32)
        angle_rads = self._get_angles(positions)
        print("Angle radians tensor:", angle_rads)  # Print angle_rads tensor
        position_encodings[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        position_encodings[:, 1::2] = torch.cos(angle_rads[:, 1::2])
        return position_encodings

    def _get_angles(self, positions):
        max_length = positions.size(1)
        power_values = torch.pow(1000, 2 * torch.arange(0, self.d_model, 2).float() / self.d_model)
        angle_rads = positions.float() / power_values[:max_length]
        return angle_rads

import torch
import torch.nn as nn

class CharacterEmbedding(nn.Module):
    def __init__(self, d_model, START_TOKEN, END_TOKEN, PADDING_TOKEN, max_sequence_length=32):
        super().__init__()
        self.vocab_size = 98  # Assuming the vocab size is 98
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
    
    def batch_tokenize(self, positions, start_token, end_token):
        def tokenize(positions, start_token, end_token):
            tokenized_positions = []

            for pos_tensor in positions:
                sentence_char_indices = []
                for pos in pos_tensor:
                    # Convert each position to its corresponding character index
                    char_index = pos.item()  # Convert tensor element to Python int
                    sentence_char_indices.append(char_index)

                # Add start and end tokens
                if start_token:
                    sentence_char_indices.insert(0, start_token)
                if end_token:
                    sentence_char_indices.append(end_token)

                # Pad the sequence to ensure it has max_sequence_length
                sentence_char_indices += [self.PADDING_TOKEN] * (self.max_sequence_length - len(sentence_char_indices))

                tokenized_positions.append(sentence_char_indices)

            return torch.tensor(tokenized_positions)

        tokenized = tokenize(positions, start_token, end_token)
        print("All tokenized :", tokenized[1])
        return tokenized.to(self.embedding.weight.device)
   

    def forward(self, x, start_token, end_token):
        print("Shape of input tensor x:", x.shape)  # Input is positional tensor
        x = self.batch_tokenize(x, start_token, end_token)
        print("Shape of input tensor x:", x.shape)
    
        # Embedding
        x_emb = self.embedding(x.to(self.embedding.weight.device))  # Move input tensor to the same device as embedding weight
        print("Shape of tensor x_emb:", x_emb.shape)  # Print shape of tensor x_emb

        # Apply positional encoding
        pos = self.position_encoder(x.size(0), x.size(1)).to(x_emb.device)  # Pass the batch size and sequence length
        print("Shape of tensor pos:", pos.shape)  # Print shape of tensor pos
    
        # Ensure pos has the same sequence length as x_emb
        if pos.size(1) != x_emb.size(1):
            pos = pos[:, :x_emb.size(1)]

        # Apply dropout
        x = self.dropout(x_emb + pos)
    
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Define linear layers for query, key, and value projections
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # Final linear layer after concatenating attention outputs
        self.out_linear = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask):
        batch_size, max_sequence_length, d_model = x.size()
        
        # Project input into query, key, and value tensors
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        
        # Split into multiple heads
        q = q.view(batch_size, max_sequence_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, max_sequence_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, max_sequence_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Calculate scaled dot-product attention
        values, attention = scaled_dot_product(q, k, v, mask)
        
        # Concatenate heads and apply final linear layer
        values = values.permute(0, 2, 1, 3).contiguous().view(batch_size, max_sequence_length, -1)
        out = self.out_linear(values)
        
        return out

class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        return out

  
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, self_attention_mask):
        residual_x = x.clone()
        # Apply attention
        x, _ = self.attention(x, mask=self_attention_mask)
        x = self.dropout1(x)
        # Apply layer normalization and residual connection
        x = self.norm1(x + residual_x)
        residual_x = x.clone()
        # Apply position-wise feed forward network
        x = self.ffn(x)
        x = self.dropout2(x)
        # Apply layer normalization and residual connection
        x = self.norm2(x + residual_x)
        return x
    
class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs):
        x, self_attention_mask = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)
        return x


class Encoder(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers,
                 max_sequence_length,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = CharacterEmbedding(d_model, START_TOKEN, END_TOKEN, PADDING_TOKEN, max_sequence_length=32)
        self.layers = SequentialEncoder(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                      for _ in range(num_layers)])

    def forward(self, x, self_attention_mask, start_token, end_token):
        x = self.sentence_embedding(x, start_token, end_token)
        for layer in self.layers:
            x = layer(x, self_attention_mask)
        return x

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_layer = nn.Linear(d_model, 2 * d_model)
        self.q_layer = nn.Linear(d_model, d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, y, mask=None):
        batch_size, sequence_length, d_model = x.size()
        kv = self.kv_layer(x)
        q = self.q_layer(y)
        kv = kv.view(batch_size, sequence_length, self.num_heads, 2 * self.head_dim)  # Use view instead of reshape
        q = q.view(batch_size, sequence_length, self.num_heads, self.head_dim)  # Use view instead of reshape
        kv = kv.permute(0, 2, 1, 3)
        q = q.permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)  # We don't need the mask for cross attention
        values = values.permute(0, 2, 1, 3).contiguous().view(batch_size, sequence_length, d_model)  # Use contiguous and view
        out = self.linear_layer(values)
        return out

class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.encoder_decoder_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.layer_norm3 = LayerNormalization(parameters_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        _y = y.clone()
        # Apply self-attention mechanism
        y, _ = self.self_attention(y, y, y, mask=self_attention_mask)
        y = self.dropout1(y)
        # Apply layer normalization and residual connection
        y = self.layer_norm1(y + _y)

        _y = y.clone()
        # Apply encoder-decoder attention mechanism
        y = self.encoder_decoder_attention(x, y, mask=cross_attention_mask)
        y = self.dropout2(y)
        # Apply layer normalization and residual connection
        y = self.layer_norm2(y + _y)

        _y = y.clone()
        # Apply position-wise feed-forward network
        y = self.ffn(y)
        y = self.dropout3(y)
        # Apply layer normalization and residual connection
        y = self.layer_norm3(y + _y)
        return y


class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, self_attention_mask, cross_attention_mask = inputs
        for module in self._modules.values():
            y = module(x, y, self_attention_mask, cross_attention_mask)
        return y


class Decoder(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers,
                 max_sequence_length,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = CharacterEmbedding(d_model, START_TOKEN, END_TOKEN, PADDING_TOKEN, max_sequence_length=32)
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self, x, y, self_attention_mask, cross_attention_mask, start_token, end_token):
        y = self.sentence_embedding(y, start_token, end_token)
        for layer in self.layers:
            y = layer(x, y, self_attention_mask, cross_attention_mask)
        return y

class Transformer(nn.Module):
    def __init__(self, 
                d_model, 
                ffn_hidden,
                num_heads, 
                drop_prob, 
                num_layers,
                max_sequence_length, 
                kn_vocab_size,
                english_to_index,
                hindi_to_index,
                START_TOKEN, 
                END_TOKEN, 
                PADDING_TOKEN
                ):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, english_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN).to(self.device)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, hindi_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN).to(self.device)
        self.linear = nn.Linear(d_model, kn_vocab_size).to(self.device)

    def forward(self, 
                x, 
                y, 
                encoder_self_attention_mask=None, 
                decoder_self_attention_mask=None, 
                decoder_cross_attention_mask=None,
                enc_start_token=True,
                enc_end_token=True,
                dec_start_token=True,
                dec_end_token=True):
        # Encode the input English sentences
        x = self.encoder(x, encoder_self_attention_mask, start_token=enc_start_token, end_token=enc_end_token)
        # Decode the encoded input to Kannada transliteration
        out = self.decoder(x, y, decoder_self_attention_mask, decoder_cross_attention_mask, start_token=dec_start_token, end_token=dec_end_token)
        # Apply linear transformation to get the output in Kannada vocabulary space
        out = self.linear(out)
        return out