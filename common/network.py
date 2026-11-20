"""The surrogate network (TensorFlow / Keras).

Input  [xi_n, etr_n, M_n, eps, T]
Output [d_xi, d_etr], the increments of the two internal variables.

Predicting increments makes "nothing changes", the most frequent case, the
zero output. The new state is formed outside the network:

    xi  = clip(xi_n + d_xi, 0, 1)
    etr = etr_n + d_etr
"""
import tensorflow as tf

WIDTH = 128
layers = tf.keras.layers


class PhysicsFeatures(layers.Layer):
    """Four products of the inputs that a dense layer could only approximate.

        eps - xi_n * etr_n     elastic strain of the old state
        M_n / max(xi_n, 0.02)  memory per unit of martensite
        (1 - xi_n) * eps       strain carried by the austenite
        xi_n * etr_n           macroscopic transformation strain
    """

    def call(self, x):
        xi, etr, M, eps = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
        return tf.concat([eps - xi * etr, M / tf.maximum(xi, 0.02), (1 - xi) * eps, xi * etr], axis=1)


def _residual_block(x, width):
    r = layers.Dense(width, activation="swish")(x)
    r = layers.Dense(width)(r)
    return layers.Activation("swish")(layers.Add()([x, r]))


def _head(shared, normalized_input, name):
    h = layers.Concatenate()([shared, normalized_input])
    h = layers.Dense(64, activation="swish")(h)
    h = layers.Dense(32, activation="swish")(h)
    return layers.Dense(1, name=name)(h)


def build_network(training_inputs, width=WIDTH):
    """Normalization + features -> dense -> 3 residual blocks -> shared dense -> 2 heads.

    `training_inputs` (N, 5) sets the mean and variance of the normalization layer.
    138,242 trainable parameters at width 128.
    """
    normalization = layers.Normalization(name="norm")
    inputs = tf.keras.Input((5,), name="input")
    normalized = normalization(inputs)
    x = layers.Concatenate()([normalized, PhysicsFeatures()(inputs)])
    x = layers.Dense(width, activation="swish")(x)
    for _ in range(3):
        x = _residual_block(x, width)
    shared = layers.Dense(width, activation="swish")(x)
    d_xi = _head(shared, normalized, "dksi")
    d_etr = _head(shared, normalized, "deps")
    model = tf.keras.Model(inputs, layers.Concatenate()([d_xi, d_etr]), name="delta")
    normalization.adapt(training_inputs)
    return model


def export_onnx(model, path):
    """Wrap the increments into the new state and save as ONNX (needs `tf2onnx`)."""
    import tf2onnx
    inputs = tf.keras.Input((5,), name="input")
    delta = model(inputs)
    xi = layers.Lambda(lambda a: tf.clip_by_value(a[0][:, 0:1] + a[1][:, 0:1], 0.0, 1.0))([inputs, delta])
    etr = layers.Lambda(lambda a: a[0][:, 1:2] + a[1][:, 1:2])([inputs, delta])
    wrapped = tf.keras.Model(inputs, layers.Concatenate(name="output")([xi, etr]))
    tf2onnx.convert.from_keras(wrapped, opset=13, output_path=str(path))
