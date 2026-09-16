"""VAE: Residual+SE блоки, круговой вход, обучение с early stopping."""
from __future__ import annotations
import os
from typing import Callable, Tuple

import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import layers, models, callbacks, regularizers


# ---------- Слои и блоки ----------

def residual_se_block(
    x,
    filters: int,
    k: int = 3,
    reduction: int = 16,
    dropout: float = 0.2,
):
    """Residual + Squeeze-and-Excitation + SpatialDropout."""
    shortcut = x
    in_ch = tf.keras.backend.int_shape(x)[-1]

    x = layers.Conv2D(filters, k, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, k, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)

    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(max(1, filters // reduction), activation="relu")(se)
    se = layers.Dense(filters, activation="sigmoid")(se)
    se = layers.Reshape((1, 1, filters))(se)
    x = layers.multiply([x, se])

    if in_ch != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same",
                                 kernel_regularizer=regularizers.l2(1e-4))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    x = layers.SpatialDropout2D(dropout)(x)
    return x


@keras.saving.register_keras_serializable()
class Sampling(layers.Layer):
    def call(self, inputs):
        mu, lv = inputs
        b = tf.shape(mu)[0]; h = tf.shape(mu)[1]
        w = tf.shape(mu)[2]; c = tf.shape(mu)[3]
        eps = tf.random.normal((b, h, w, c))
        return mu + tf.exp(0.5 * lv) * eps


@keras.saving.register_keras_serializable()
class VAE(models.Model):
    """VAE с кастомным train_step: loss = MSE + KL."""
    def __init__(self, encoder, decoder, **kw):
        super().__init__(**kw)
        self.encoder = encoder
        self.decoder = decoder
        self.t = tf.keras.metrics.Mean(name="total_loss")
        self.r = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.k = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.t, self.r, self.k]

    def _step(self, data, training: bool):
        x, _ = data
        with tf.GradientTape() as tape:
            mu, lv, z = self.encoder(x, training=training)
            r = self.decoder(z, training=training)
            rec = tf.reduce_mean(tf.square(x - r))
            kl = -0.5 * tf.reduce_mean(1 + lv - tf.square(mu) - tf.exp(lv))
            total = rec + kl
        if training:
            g = tape.gradient(total, self.trainable_weights)
            self.optimizer.apply_gradients(zip(g, self.trainable_weights))
        self.t.update_state(total)
        self.r.update_state(rec)
        self.k.update_state(kl)
        return {"loss": self.t.result(),
                "reconstruction_loss": self.r.result(),
                "kl_loss": self.k.result()}

    def train_step(self, data):
        return self._step(data, True)

    def test_step(self, data):
        return self._step(data, False)


def build_vae(
    img_h: int = 256,
    img_w: int = 256,
    bottleneck: int = 64,
    downs: int = 3,
) -> Tuple[models.Model, models.Model, VAE]:
    """Собирает encoder, decoder, vae."""
    lat_h, lat_w = img_h // 2 ** downs, img_w // 2 ** downs

    inp = layers.Input(shape=(img_h, img_w, 3), name="input_image")
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = residual_se_block(x, 32, dropout=0.1);  x = layers.MaxPooling2D(2, padding="same")(x)
    x = residual_se_block(x, 64, dropout=0.15); x = layers.MaxPooling2D(2, padding="same")(x)
    x = residual_se_block(x, 128, dropout=0.2); x = layers.MaxPooling2D(2, padding="same")(x)
    x = residual_se_block(x, 256, dropout=0.25)
    x = residual_se_block(x, bottleneck, dropout=0.4)

    mu = layers.Conv2D(bottleneck, 1, padding="same", name="latent_mu")(x)
    log_var = layers.Conv2D(bottleneck, 1, padding="same", name="latent_log_var")(x)
    z = Sampling()([mu, log_var])
    encoder = models.Model(inp, [mu, log_var, z], name="encoder")

    lat_in = layers.Input(shape=(lat_h, lat_w, bottleneck), name="latent_input")
    d = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(lat_in)
    d = layers.Conv2D(128, 3, padding="same", activation="relu")(d)
    d = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(d)
    d = layers.Conv2D(64, 3, padding="same", activation="relu")(d)
    d = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(d)
    d = layers.Conv2D(32, 3, padding="same", activation="relu")(d)
    recon = layers.Conv2D(3, 3, padding="same", activation="sigmoid", name="reconstruction")(d)
    decoder = models.Model(lat_in, recon, name="decoder")

    return encoder, decoder, VAE(encoder, decoder, name="vae")


class SaveBest(callbacks.Callback):
    """Сохраняет encoder/decoder при лучшем val_loss."""
    def __init__(self, save_dir: str = "best_models"):
        super().__init__()
        self.save_dir = save_dir
        self.best = float("inf")
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        vl = logs.get("val_loss")
        if vl is not None and vl < self.best:
            self.best = vl
            self.model.encoder.save(os.path.join(self.save_dir, "best_encoder.keras"))
            self.model.decoder.save(os.path.join(self.save_dir, "best_decoder.keras"))
            print(f"💾 best saved (val_loss={vl:.5f})")