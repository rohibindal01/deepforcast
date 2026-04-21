"""
models/forecaster.py
Advanced Time Series Forecasting Model:
- Temporal Convolutional Network (TCN) encoder
- Bidirectional LSTM layers
- Multi-Head Self-Attention (Transformer block)
- Residual connections + Layer Normalization
- Multi-step output (forecast_horizon days ahead)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from typing import Tuple, Optional
import os


# ─────────────────────────────────────────────────────────────
# Custom Layers
# ─────────────────────────────────────────────────────────────

class TransformerBlock(layers.Layer):
    """Multi-Head Self-Attention + Feed-Forward block with residual connections."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        return config


class TCNBlock(layers.Layer):
    """Temporal Convolutional Block with dilated causal convolutions."""

    def __init__(self, filters: int, kernel_size: int = 3, dilation_rate: int = 1,
                 dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv1D(
            filters, kernel_size, padding="causal",
            dilation_rate=dilation_rate, activation="gelu",
            kernel_regularizer=regularizers.l2(1e-4)
        )
        self.conv2 = layers.Conv1D(
            filters, kernel_size, padding="causal",
            dilation_rate=dilation_rate, activation="gelu",
            kernel_regularizer=regularizers.l2(1e-4)
        )
        self.dropout = layers.Dropout(dropout)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.residual_conv = layers.Conv1D(filters, 1, padding="same")

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.dropout(x, training=training)
        x = self.conv2(x)
        x = self.dropout(x, training=training)
        residual = self.residual_conv(inputs)
        return self.norm(x + residual)

    def get_config(self):
        return super().get_config()


# ─────────────────────────────────────────────────────────────
# Model Builder
# ─────────────────────────────────────────────────────────────

def build_model(
    seq_length: int,
    n_features: int,
    forecast_horizon: int,
    tcn_filters: int = 64,
    lstm_units: int = 128,
    transformer_heads: int = 4,
    transformer_ff_dim: int = 256,
    embed_dim: int = 128,
    dropout: float = 0.2,
) -> Model:
    """
    Build the LSTM-Transformer hybrid forecasting model.

    Architecture:
        Input → TCN Stack (dilated causal convs)
              → Projection → Transformer Block
              → Bidirectional LSTM
              → Attention Pooling
              → Dense Head → Output (forecast_horizon)
    """
    inputs = keras.Input(shape=(seq_length, n_features), name="input_sequences")

    # TCN Encoder (captures local temporal patterns)
    x = TCNBlock(tcn_filters, kernel_size=3, dilation_rate=1, dropout=dropout, name="tcn_1")(inputs)
    x = TCNBlock(tcn_filters, kernel_size=3, dilation_rate=2, dropout=dropout, name="tcn_2")(x)
    x = TCNBlock(tcn_filters * 2, kernel_size=3, dilation_rate=4, dropout=dropout, name="tcn_3")(x)

    # Project to embed_dim for Transformer
    x = layers.Dense(embed_dim, name="projection")(x)
    x = layers.LayerNormalization(epsilon=1e-6, name="pre_transformer_norm")(x)

    # Transformer Block (captures long-range dependencies)
    x = TransformerBlock(embed_dim, transformer_heads, transformer_ff_dim, dropout, name="transformer_block_1")(x)
    x = TransformerBlock(embed_dim, transformer_heads, transformer_ff_dim, dropout, name="transformer_block_2")(x)

    # Bidirectional LSTM (captures sequential dynamics)
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, dropout=dropout, recurrent_dropout=0.1),
        name="bilstm_1"
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(lstm_units // 2, return_sequences=True, dropout=dropout),
        name="bilstm_2"
    )(x)

    # Attention pooling over time steps
    attn_weights = layers.Dense(1, activation="softmax", name="temporal_attention")(x)
    x = layers.Multiply(name="weighted_context")([x, attn_weights])
    x = layers.GlobalAveragePooling1D(name="context_pooling")(x)

    # Prediction head
    x = layers.Dense(256, activation="gelu", kernel_regularizer=regularizers.l2(1e-4), name="dense_1")(x)
    x = layers.Dropout(dropout, name="head_dropout_1")(x)
    x = layers.Dense(128, activation="gelu", name="dense_2")(x)
    x = layers.Dropout(dropout / 2, name="head_dropout_2")(x)

    outputs = layers.Dense(forecast_horizon, activation="linear", name="forecast_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="LSTM_Transformer_Forecaster")
    return model


def compile_model(model: Model, learning_rate: float = 1e-3) -> Model:
    """Compile model with Huber loss (robust to outliers) and multiple metrics."""
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.Huber(delta=1.0),
        metrics=["mae", "mse"],
    )
    return model


def get_callbacks(model_dir: str, patience: int = 15) -> list:
    """Get training callbacks."""
    os.makedirs(model_dir, exist_ok=True)
    return [
        EarlyStopping(
            monitor="val_loss", patience=patience,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=patience // 2, min_lr=1e-6, verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(model_dir, "best_model.keras"),
            monitor="val_loss", save_best_only=True, verbose=0
        ),
    ]


def train_model(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_dir: str = "models/saved",
    epochs: int = 100,
    batch_size: int = 32,
) -> keras.callbacks.History:
    """Train model with callbacks."""
    callbacks = get_callbacks(model_dir)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
        shuffle=False,  # Time series: preserve order
    )
    return history


def save_model(model: Model, path: str):
    """Save model in Keras native format."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"Model saved to {path}")


def load_model(path: str) -> Model:
    """Load saved model with custom objects."""
    custom_objects = {
        "TransformerBlock": TransformerBlock,
        "TCNBlock": TCNBlock,
    }
    return keras.models.load_model(path, custom_objects=custom_objects)
