"""Ground-truth core models for the subculture-colorization benchmark.

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
        inputs = tf.concat((z, context), 1)

        x = self.fc_1(inputs)
        x = self.fc_2(x)
        x = self.fc_3(x)
        x = self.fc_4(x)

        return x        # (batch, 3)


class T2PDiscriminator(keras.Model):
    def __init__(self):
        super(T2PDiscriminator, self).__init__()
        activation = tf.nn.leaky_relu

        self.fc_1 = Dense(512, activation=activation)
        self.fc_2 = Dense(256, activation=activation)
        self.fc_3 = Dense(128, activation=activation)
        self.fc_4 = Dense(1, activation=tf.nn.sigmoid)

    def call(self, result, context):
        inputs = tf.concat([result, context], 1)

        x = self.fc_1(inputs)
        x = self.fc_2(x)
        x = self.fc_3(x)
        x = self.fc_4(x)

        return x        # (batch ,1)


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
    inputs = Input(shape=[256, 256, 1])
    context = Input(shape=[143])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
        downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(512, 4),  # (bs, 16, 16, 512)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4),  # (bs, 16, 16, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(3, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           use_bias=False,
                                           activation='tanh')  # (bs, 256, 256, 3)
    
    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    context_reshape = Reshape([1, 1, 143])(context)

    x = tf.keras.layers.Concatenate(3)([x, context_reshape])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=[inputs, context], outputs=x)


def ColorizationDiscriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inputs = Input(shape=[256, 256, 3])
    context = Input(shape=[143])

    x = inputs

    down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)
    down4 = downsample(512, 4)(down3)  # (bs, 16, 16, 512)
    down5 = Conv2D(1, 1, strides=(1, 1), padding='valid')(down4)
    projection = Reshape([256])(down5)

    concat = Concatenate(1)([projection, context])

    last = Dense(1, activation='sigmoid')(concat)

    return tf.keras.Model(inputs=[inputs, context], outputs=last)


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
