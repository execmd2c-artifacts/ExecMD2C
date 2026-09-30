"""Ablated template for the subculture-colorization benchmark.

This file consolidates only the two cGAN model families used by the project:
text-to-palette generation and palette-conditioned image colorization. Training
orchestration, demos, saved artifacts, dataset code, application wrappers, and
remote feature-extraction models are intentionally excluded.
"""

# --- Third-party imports ---
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *


# --- [Original file: models/text2palette.py] ---

class T2PGenerator(keras.Model):
    def __init__(self):
        super(T2PGenerator, self).__init__()

        activation = tf.nn.relu

        self.fc_1 = Dense(512, activation=activation)
        self.fc_2 = Dense(256, activation=activation)
        self.fc_3 = Dense(128, activation=activation)
        self.fc_4 = Dense(3, activation=tf.nn.sigmoid)

    def call(self, z, context):
        """[TODO] Generate one RGB palette color from noise and multimodal context.

        Input:
            z: (batch, 128) - latent noise vector for palette diversity.
            context: (batch, 143) - fused context feature containing subculture
                attributes and the rolling context palette.

        Output:
            (batch, 3) - one normalized RGB color, with every channel in [0, 1].

"""
        pass


class T2PDiscriminator(keras.Model):
    def __init__(self):
        super(T2PDiscriminator, self).__init__()
        activation = tf.nn.leaky_relu

        self.fc_1 = Dense(512, activation=activation)
        self.fc_2 = Dense(256, activation=activation)
        self.fc_3 = Dense(128, activation=activation)
        self.fc_4 = Dense(1, activation=tf.nn.sigmoid)

    def call(self, result, context):
        """[TODO] Score whether a generated RGB palette color matches its context.

        Input:
            result: (batch, 3) - candidate normalized RGB color.
            context: (batch, 143) - fused context feature used to condition the
                discriminator decision.

        Output:
            (batch, 1) - probability-like real/fake score for each sample.

"""
        pass


# --- [Original file: models/colorization.py] ---

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def ColorizationGenerator():
    """[TODO] Build the palette-conditioned U-Net generator for image colorization.

    Input:
        grayscale image input: (batch, 256, 256, 1) - normalized single-channel
            image to be colorized.
        context input: (batch, 143) - palette/context condition vector.

    Output:
        Keras model mapping the two inputs to (batch, 256, 256, 3), a normalized
        RGB image.

"""
    pass


def ColorizationDiscriminator():
    """[TODO] Build the context-conditioned image discriminator.

    Input:
        RGB image input: (batch, 256, 256, 3) - real or generated color image.
        context input: (batch, 143) - palette/context condition vector.

    Output:
        Keras model mapping the two inputs to (batch, 1), a bounded adversarial
        score for each image/context pair.

"""
    pass


if __name__ == "__main__":
    tf.random.set_seed(42)

    passed = 0
    failed = 0

    def check(test_name, condition, detail=""):
        global passed, failed
        if bool(condition):
            passed += 1
            print(f"  [{test_name}] PASS")
        else:
            failed += 1
            suffix = f" - {detail}" if detail else ""
            print(f"  [{test_name}] FAIL{suffix}")

    def skip_checks(count, reason):
        global failed
        failed += count
        print(f"  [skipped {count} check(s)] FAIL - {reason}")

    print("Subculture colorization benchmark checks")

    print("[Test 1/4] T2PGenerator.call")
    try:
        generator = T2PGenerator()
        z = tf.ones((2, 128), dtype=tf.float32)
        context = tf.ones((2, 143), dtype=tf.float32) * 0.25
        palette = generator(z, context)
        check("T2PGenerator output not None", palette is not None)
        if palette is not None:
            check("T2PGenerator output shape", tuple(palette.shape) == (2, 3), str(tuple(palette.shape)))
            check("T2PGenerator output finite", bool(tf.reduce_all(tf.math.is_finite(palette)).numpy()))
            check("T2PGenerator sigmoid range",
                  bool(tf.reduce_all((palette >= 0.0) & (palette <= 1.0)).numpy()))
        else:
            skip_checks(3, "T2PGenerator returned None")
    except Exception as exc:
        skip_checks(4, f"T2PGenerator.call raised {type(exc).__name__}: {exc}")

    print("[Test 2/4] T2PDiscriminator.call")
    try:
        discriminator = T2PDiscriminator()
        result = tf.ones((2, 3), dtype=tf.float32) * 0.5
        context = tf.ones((2, 143), dtype=tf.float32) * 0.25
        score = discriminator(result, context)
        check("T2PDiscriminator output not None", score is not None)
        if score is not None:
            check("T2PDiscriminator output shape", tuple(score.shape) == (2, 1), str(tuple(score.shape)))
            check("T2PDiscriminator output finite", bool(tf.reduce_all(tf.math.is_finite(score)).numpy()))
            check("T2PDiscriminator sigmoid range",
                  bool(tf.reduce_all((score >= 0.0) & (score <= 1.0)).numpy()))
        else:
            skip_checks(3, "T2PDiscriminator returned None")
    except Exception as exc:
        skip_checks(4, f"T2PDiscriminator.call raised {type(exc).__name__}: {exc}")

    print("[Test 3/4] ColorizationGenerator")
    try:
        color_generator = ColorizationGenerator()
        check("ColorizationGenerator returns model", isinstance(color_generator, tf.keras.Model),
              type(color_generator).__name__)
        if color_generator is not None:
            input_shapes = [tuple(tensor.shape.as_list()[1:]) for tensor in color_generator.inputs]
            sequential_layers = [layer for layer in color_generator.layers if isinstance(layer, tf.keras.Sequential)]
            concatenate_layers = [layer for layer in color_generator.layers if isinstance(layer, Concatenate)]
            check("ColorizationGenerator input signatures",
                  input_shapes == [(256, 256, 1), (143,)], str(input_shapes))
            check("ColorizationGenerator output image shape",
                  tuple(color_generator.output_shape) == (None, 256, 256, 3), str(color_generator.output_shape))
            check("ColorizationGenerator U-Net block count", len(sequential_layers) == 15,
                  str(len(sequential_layers)))
            check("ColorizationGenerator context and skip concatenations", len(concatenate_layers) == 8,
                  str(len(concatenate_layers)))
        else:
            skip_checks(4, "ColorizationGenerator returned None")
    except Exception as exc:
        skip_checks(5, f"ColorizationGenerator raised {type(exc).__name__}: {exc}")

    print("[Test 4/4] ColorizationDiscriminator")
    try:
        color_discriminator = ColorizationDiscriminator()
        check("ColorizationDiscriminator returns model", isinstance(color_discriminator, tf.keras.Model),
              type(color_discriminator).__name__)
        if color_discriminator is not None:
            input_shapes = [tuple(tensor.shape.as_list()[1:]) for tensor in color_discriminator.inputs]
            sequential_layers = [layer for layer in color_discriminator.layers if isinstance(layer, tf.keras.Sequential)]
            concatenate_layers = [layer for layer in color_discriminator.layers if isinstance(layer, Concatenate)]
            check("ColorizationDiscriminator input signatures",
                  input_shapes == [(256, 256, 3), (143,)], str(input_shapes))
            check("ColorizationDiscriminator output score shape",
                  tuple(color_discriminator.output_shape) == (None, 1), str(color_discriminator.output_shape))
            check("ColorizationDiscriminator downsampling block count", len(sequential_layers) == 4,
                  str(len(sequential_layers)))
            check("ColorizationDiscriminator context fusion", len(concatenate_layers) == 1,
                  str(len(concatenate_layers)))
        else:
            skip_checks(4, "ColorizationDiscriminator returned None")
    except Exception as exc:
        skip_checks(5, f"ColorizationDiscriminator raised {type(exc).__name__}: {exc}")

    print(f"RESULT: passed={passed} failed={failed}")
    if failed != 0:
        raise SystemExit(1)
