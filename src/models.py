import tensorflow as tf
from tensorflow import keras

from keras.layers import Dense, LayerNormalization
from keras.layers import Dense, Embedding
from keras import backend as K

from keras.layers import MultiHeadAttention, Dropout

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)

        self.ffn = keras.Sequential([
            Dense(dff, activation="relu"),
            Dense(d_model),
        ])

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, training=False, mask=None):
        

        # 1) Self-attention block
        attn_input = self.layernorm1(x)
        attn_output = self.mha(attn_input, attn_input, attn_input,
                               attention_mask=mask,
                               training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x = x + attn_output   # residual

        # 2) Feed-forward block
        ffn_input = self.layernorm2(x)
        ffn_output = self.ffn(ffn_input, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        x = x + ffn_output    # residual

        return x



class Transformer_Insertion_Deletion(tf.keras.Model):
    """
    Transformer-based model with the SAME I/O contract as BI_LSTM_Insertion_Deletion2:
      input  : (batch, symmm, 2*gamma+1)
      output : (batch, symmm, output_size)  (logits)
    """

    def __init__(
        self,
        d_model=128,
        d_ffn=[128],
        num_layers=3,
        output_size=4,
        num_heads=4,
        dropout_rate=0.1,
        max_len=4096,
    ):
        super().__init__()

        # Store hyperparameters
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.max_len = max_len

        # Project input window (size 2*gamma+1) to model dimension
        self.input_proj = Dense(self.d_model)

        # Positional embeddings over the time axis (symmm)
        self.pos_embedding = Embedding(input_dim=self.max_len, output_dim=self.d_model)

        # Transformer encoder layers
        dff_internal = 4 * self.d_model
        self.encoder_layers = [
            TransformerEncoderLayer(self.d_model, self.num_heads, dff_internal, dropout_rate=self.dropout_rate)
            for _ in range(self.num_layers)
        ]

        # MLP over time steps
        self.mlp_layers = [Dense(k, activation="relu") for k in self.d_ffn]

        # Final output layer: logits over symbols or bits
        self.output_layer = Dense(output_size)

    def call(self, x, training=False):
        """
        x: (batch, symmm, 2*gamma+1)
        returns: (batch, symmm, output_size)
        """

        # Project input to d_model
        # shape: (batch, symmm, d_model)
        x = self.input_proj(x)

        # Add positional embeddings on the sequence axis
        seq_len = tf.shape(x)[1]  # symmm
        positions = tf.range(start=0, limit=seq_len, delta=1)
        pos_embeddings = self.pos_embedding(positions)          # (symmm, d_model)
        pos_embeddings = tf.expand_dims(pos_embeddings, axis=0)  # (1, symmm, d_model)
        x = x + pos_embeddings

        # Pass through Transformer encoder stack
        for enc_layer in self.encoder_layers:
            x = enc_layer(x, training=training)

        # Time-distributed MLP (Dense works on last dim, keeps (batch, symmm))
        for layer in self.mlp_layers:
            x = layer(x)

        # Final logits
        x = self.output_layer(x)

        return x
