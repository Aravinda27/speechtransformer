import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeechTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers, hidden_dim, dropout_prob):
        super(SpeechTransformer, self).__init__()
        self.encoder = TransformerEncoder(input_dim, num_heads, num_layers, hidden_dim, dropout_prob)
        self.decoder = TransformerDecoder(output_dim, num_heads, num_layers, hidden_dim, dropout_prob)
        self.output_layer = nn.Linear(hidden_dim, output_dim)  # Adjust output layer to output_dim
    
    def forward(self, encoder_input):
        decoder_input = encoder_input
        encoder_output = self.encoder(encoder_input)
        decoder_output = self.decoder(decoder_input, encoder_output)
        # print(decoder_output.shape)
        desired_size = 512
        current_size = decoder_output.size(2)
        if current_size < desired_size:
            num_repeats = (desired_size + current_size - 1) // current_size 
            padded_decoder_output = torch.cat([decoder_output] * num_repeats, dim=2)
            padded_decoder_output = padded_decoder_output[:, :, :desired_size]
        else:
            padded_decoder_output = decoder_output
        # print(padded_decoder_output.shape)
        output = self.output_layer(padded_decoder_output)
        return output

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, dropout_prob):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        
        # Positional encoding layer
        self.positional_encoding = PositionalEncoding(input_dim)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(input_dim, num_heads, hidden_dim, dropout_prob)
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        # Add positional encoding to input
        x = self.positional_encoding(x)
        
        # Transformer encoder
        for layer in self.encoder_layers:
            x = layer(x)
        
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, output_dim, num_heads, num_layers, hidden_dim, dropout_prob):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        
        # Positional encoding layer
        self.positional_encoding = PositionalEncoding(output_dim)
        
        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(output_dim, num_heads, hidden_dim, dropout_prob)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, encoder_output):
        # Add positional encoding to input
        x = self.positional_encoding(x)
        
        # Transformer decoder
        for layer in self.decoder_layers:
            x = layer(x, encoder_output)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=200):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(64, max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[0, :, 0::2] = torch.sin(position * div_term)
        self.encoding[0, :, 1::2] = torch.cos(position * div_term)
    
    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (self.encoding.shape[1], self.encoding.shape[2]))
        print(x.shape)
        print(self.encoding.shape)
        return x + self.encoding


class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, dropout_prob):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout_prob)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
    
    def forward(self, x):
        # Self-attention layer
        x_normalized = self.layer_norm1(x)
        print(x.shape)
        exit()
        attn_output, _ = self.self_attention(x_normalized, x_normalized, x_normalized)
        x = x + self.dropout(attn_output)
        
        # Feedforward layer
        x_normalized = self.layer_norm2(x)
        x_intermediate = F.relu(self.linear1(x_normalized))
        x_intermediate = self.dropout(x_intermediate)
        x = x + self.linear2(x_intermediate)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, output_dim, num_heads, hidden_dim, dropout_prob):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(output_dim, num_heads, dropout=dropout_prob)
        self.encoder_attention = nn.MultiheadAttention(output_dim, num_heads, dropout=dropout_prob)
        self.linear1 = nn.Linear(output_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm1 = nn.LayerNorm(output_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        self.layer_norm3 = nn.LayerNorm(output_dim)
    
    def forward(self, x, encoder_output):
        # Self-attention layer
        x_normalized = self.layer_norm1(x)
        print(x.shape)
        exit()
        attn_output, _ = self.self_attention(x_normalized, x_normalized, x_normalized)
        x = x + self.dropout(attn_output)
        
        # Encoder-decoder attention layer
        x_normalized = self.layer_norm2(x)
        # print(x_normalized.shape)
        # print(encoder_output.shape)
        encoder_output = F.adaptive_avg_pool2d(encoder_output, (x_normalized.shape[1], x_normalized.shape[2]))
        # print(x_normalized.shape)
        # print(encoder_output.shape)
        attn_output, _ = self.encoder_attention(x_normalized, encoder_output, encoder_output)
        x = x + self.dropout(attn_output)
        
        # Feedforward
        x_normalized = self.layer_norm3(x)
        x_intermediate = F.relu(self.linear1(x_normalized))
        x_intermediate = self.dropout(x_intermediate)
        x = x + self.linear2(x_intermediate)
        return x

def main():
    BATCH_SIZE = 64
    SEQ_LENGTH = 200
    INPUT_DIM = 160
    OUTPUT_DIM = 40
    NUM_HEADS = 8
    NUM_LAYERS = 6
    HIDDEN_DIM = 512
    DROPOUT_PROB = 0.1

    # Create random input tensor
    inputs = torch.rand(BATCH_SIZE, SEQ_LENGTH, INPUT_DIM)
    print(inputs.shape)

    # Initialize the model
    model = SpeechTransformer(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, 
                               num_heads=NUM_HEADS, num_layers=NUM_LAYERS, 
                               hidden_dim=HIDDEN_DIM, dropout_prob=DROPOUT_PROB)

    # Forward pass
    output = model(inputs)

    # Print output shape
    print("Output Shape:", output.shape)

if __name__ == "__main__":
    main()
