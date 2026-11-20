"""The network variants compared in this step (Keras).

All take the physical inputs x = [xi_n, etr_n, M_n, eps, T] and return the
normalized next state [xi, etr]; `denormalize` turns the output into physical
units. Only the shape between input and output differs.

    A      dense 64-64-32, swish                                    6,690 weights
    B2     the same trunk, xi through a sigmoid (xi in [0, 1])      6,690
    C32    physics features, 3 residual blocks, 2 heads, width 32   9,218
    C64    the same topology, width 64                             35,330
    C128   the same topology, width 128                           138,242

B1 is not a network of its own: it is A with xi clipped to [0, 1] after prediction.
"""
import numpy as np
import tensorflow as tf

HEAD_UNITS = {32: (16, 8), 64: (32, 16), 128: (64, 32)}


class Standardize(tf.keras.layers.Layer):
    """(x - mean) / std with fixed constants, no weights."""

    def __init__(self, values, **kwargs):
        super().__init__(**kwargs)
        values = np.asarray(values, np.float32)
        self.mean = tf.constant(values.mean(0, keepdims=True))
        self.std = tf.constant(values.std(0, keepdims=True) + 1e-12)

    def call(self, x):
        return (x - self.mean) / self.std


def physics_features(x):
    """Four products that appear in the energy; a dense layer can only form weighted sums."""
    xi, etr, m, eps = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
    return tf.concat([eps - xi * etr,                 # elastic strain
                      m / tf.maximum(xi, 0.02),       # memory per unit of martensite
                      (1.0 - xi) * eps,               # strain carried by the austenite
                      xi * etr], axis=1)              # macroscopic transformation strain


def build_dense(x_train, units=(64, 64, 32)):
    """Variant A."""
    inputs = tf.keras.Input(shape=(5,))
    h = Standardize(x_train)(inputs)
    for u in units:
        h = tf.keras.layers.Dense(u, activation="swish")(h)
    return tf.keras.Model(inputs, tf.keras.layers.Dense(2)(h), name="A")


def build_bounded(x_train, y_train, units=(64, 64, 32)):
    """Variant B2: A with xi = sigmoid(.), mapped back to normalized units for the loss."""
    y_mean, y_std = np.asarray(y_train)[:, 0].mean(), np.asarray(y_train)[:, 0].std()
    inputs = tf.keras.Input(shape=(5,))
    h = Standardize(x_train)(inputs)
    for u in units:
        h = tf.keras.layers.Dense(u, activation="swish")(h)
    raw = tf.keras.layers.Dense(2)(h)
    xi = (tf.keras.ops.sigmoid(raw[:, 0:1]) - y_mean) / y_std
    return tf.keras.Model(inputs, tf.keras.layers.Concatenate()([xi, raw[:, 1:2]]), name="B2")


def build_residual(x_train, width=128, blocks=3):
    """Variant C: physics features -> dense -> residual blocks -> shared layer -> one head per output."""
    x_train = np.asarray(x_train, np.float32)
    inputs = tf.keras.Input(shape=(5,))
    x_norm = Standardize(x_train)(inputs)
    feats = tf.keras.layers.Lambda(physics_features, output_shape=(4,))(inputs)
    feats = Standardize(physics_features(tf.constant(x_train)).numpy())(feats)

    h = tf.keras.layers.Dense(width, activation="swish")(tf.keras.layers.Concatenate()([x_norm, feats]))
    for _ in range(blocks):
        r = tf.keras.layers.Dense(width, activation="swish")(h)
        r = tf.keras.layers.Dense(width)(r)
        h = tf.keras.layers.Activation("swish")(tf.keras.layers.Add()([h, r]))
    shared = tf.keras.layers.Dense(width, activation="swish")(h)

    outputs = []
    for _ in ("xi", "etr"):
        head = tf.keras.layers.Concatenate()([shared, x_norm])       # the inputs again, next to the trunk
        for u in HEAD_UNITS[width]:
            head = tf.keras.layers.Dense(u, activation="swish")(head)
        outputs.append(tf.keras.layers.Dense(1)(head))
    return tf.keras.Model(inputs, tf.keras.layers.Concatenate()(outputs), name=f"C{width}")


def build(name, x_train, y_train):
    if name == "A":
        return build_dense(x_train)
    if name == "B2":
        return build_bounded(x_train, y_train)
    return build_residual(x_train, width=int(name[1:]))


def train(model, x, y, x_val, y_val, epochs=1000, patience=40):
    """The one recipe used for every variant: Adam 1e-3, batch 512, MSE on normalized outputs, early stopping."""
    y_mean, y_std = np.asarray(y).mean(0), np.asarray(y).std(0)
    norm = lambda a: ((np.asarray(a) - y_mean) / y_std).astype(np.float32)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    model.fit(np.float32(x), norm(y), validation_data=(np.float32(x_val), norm(y_val)), batch_size=512,
              epochs=epochs, callbacks=[stop], verbose=2)
    return lambda a: model(np.float32(a)).numpy() * y_std + y_mean     # predictor in physical units


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0.1, 0.9, (256, 5)), rng.uniform(0.1, 0.9, (256, 2))
    for name in ("A", "B2", "C32", "C64", "C128"):
        print(f"{name:5s} {build(name, x, y).count_params():7,d} weights")
