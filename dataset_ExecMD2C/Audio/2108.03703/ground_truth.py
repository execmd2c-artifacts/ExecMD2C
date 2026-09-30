# -*- coding: utf-8 -*-
"""Core model components extracted from audio-spectral-enhancement.

This file contains only the Keras model architecture from model.py. Corpus
creation, command-line entry points, file serialization, audio conversion, and
parameter-update code are intentionally excluded.
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, SeparableConv2D, PReLU, Multiply


class SingleBlock(tf.keras.layers.Layer):
    def __init__(self, dim=256, out_shape=512, *args, **kwargs):
        super(SingleBlock, self).__init__()
        self.dim = dim
        self.out_shape = out_shape
        self.l1 = SeparableConv2D(
            self.dim,
            5,
            1,
            "same",
            kernel_initializer="he_uniform",
            depthwise_initializer="he_uniform",
        )
        self.l2 = PReLU()
        self.l3 = SeparableConv2D(
            self.out_shape,
            5,
            1,
            "same",
            kernel_initializer="he_uniform",
            depthwise_initializer="he_uniform",
        )

    def call(self, inputs, **kwargs):
        x = self.l1(inputs)
        x = self.l2(x)
        x = self.l3(x)
        return x

    def get_config(self):
        config = super(SingleBlock, self).get_config()
        config.update({"dim": self.dim, "out_shape": self.out_shape})
        return config


def create_model(input_shape=(2, 400, 512), num_blocks=1):
    inp = Input(input_shape)
    block_list = []
    if num_blocks == 1:
        b1 = SingleBlock()(inp)
        return tf.keras.models.Model(inp, [b1, b1])

    elif num_blocks % 2 == 0:
        for i in range(num_blocks // 2):
            if i == 0:
                b = SingleBlock()(inp)
            else:
                b = SingleBlock()(b)
            block_list.append(b)
        for i in reversed(block_list):
            b = SingleBlock()(b)
            b = Multiply()([b, i])
        b = SeparableConv2D(
            input_shape[-1],
            5,
            1,
            "same",
            kernel_initializer="he_uniform",
            depthwise_initializer="he_uniform",
        )(b)
        return tf.keras.models.Model(inp, [b, b])

    else:
        for i in range(num_blocks // 2):
            if i == 0:
                b = SingleBlock()(inp)
            else:
                b = SingleBlock()(b)
            block_list.append(b)
        b = SingleBlock()(b)
        for i in reversed(block_list):
            b = SingleBlock()(b)
            b = Multiply()([b, i])
        b = SeparableConv2D(
            input_shape[-1],
            5,
            1,
            "same",
            kernel_initializer="he_uniform",
            depthwise_initializer="he_uniform",
        )(b)
        return tf.keras.models.Model(inp, [b, b])


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

    print("=" * 70)
    print("audio-spectral-enhancement: SingleBlock spectral autoencoder")
    print("=" * 70)

    print("-" * 70)
    print("[Test 1/4] SingleBlock.call")
    try:
        block = SingleBlock(dim=4, out_shape=6)
        x = tf.random.normal((2, 2, 8, 3))
        y = block(x, training=False)
        check("SingleBlock output not None", y is not None)
        if y is not None:
            check("SingleBlock output shape", tuple(y.shape) == (2, 2, 8, 6), str(tuple(y.shape)))
            check("SingleBlock output finite", bool(tf.reduce_all(tf.math.is_finite(y)).numpy()))
            check("SingleBlock first layer is separable convolution", isinstance(block.l1, SeparableConv2D), type(block.l1).__name__)
            check("SingleBlock second layer is PReLU", isinstance(block.l2, PReLU), type(block.l2).__name__)
            check("SingleBlock third layer reaches out_shape", block.l3.filters == 6, str(block.l3.filters))
            with tf.GradientTape() as tape:
                y_grad = block(x, training=True)
                total = tf.reduce_sum(y_grad)
            grads = tape.gradient(total, block.trainable_variables)
            check("SingleBlock has trainable gradients", any(g is not None for g in grads))
        else:
            skip_checks(6, "SingleBlock returned None")
    except Exception as exc:
        skip_checks(7, f"SingleBlock raised {type(exc).__name__}: {exc}")

    print("-" * 70)
    print("[Test 2/4] create_model with one block")
    try:
        model = create_model(input_shape=(2, 8, 3), num_blocks=1)
        x = tf.random.normal((2, 2, 8, 3))
        outputs = model(x, training=False)
        check("one-block model output not None", outputs is not None)
        if outputs is not None:
            check("one-block model returns two outputs", isinstance(outputs, list) and len(outputs) == 2, str(type(outputs)))
            check("one-block first output shape", tuple(outputs[0].shape) == (2, 2, 8, 512), str(tuple(outputs[0].shape)))
            check("one-block output tensors share shape", tuple(outputs[0].shape) == tuple(outputs[1].shape))
            check("one-block output finite", bool(tf.reduce_all(tf.math.is_finite(outputs[0])).numpy()))
            check("one-block model contains one SingleBlock", sum(isinstance(layer, SingleBlock) for layer in model.layers) == 1, str([type(layer).__name__ for layer in model.layers]))
        else:
            skip_checks(5, "one-block create_model returned None")
    except Exception as exc:
        skip_checks(6, f"one-block create_model raised {type(exc).__name__}: {exc}")

    print("-" * 70)
    print("[Test 3/4] create_model with even number of blocks")
    try:
        model = create_model(input_shape=(2, 8, 3), num_blocks=2)
        x = tf.random.normal((2, 2, 8, 3))
        outputs = model(x, training=False)
        check("even-block model output not None", outputs is not None)
        if outputs is not None:
            check("even-block model returns two outputs", isinstance(outputs, list) and len(outputs) == 2, str(type(outputs)))
            check("even-block restores input channel count", tuple(outputs[0].shape) == (2, 2, 8, 3), str(tuple(outputs[0].shape)))
            check("even-block output tensors share shape", tuple(outputs[0].shape) == tuple(outputs[1].shape))
            check("even-block output finite", bool(tf.reduce_all(tf.math.is_finite(outputs[0])).numpy()))
            check("even-block model uses multiplicative skip", any(isinstance(layer, Multiply) for layer in model.layers), str([type(layer).__name__ for layer in model.layers]))
        else:
            skip_checks(5, "even-block create_model returned None")
    except Exception as exc:
        skip_checks(6, f"even-block create_model raised {type(exc).__name__}: {exc}")

    print("-" * 70)
    print("[Test 4/4] create_model with odd number of blocks")
    try:
        model = create_model(input_shape=(2, 8, 3), num_blocks=3)
        x = tf.random.normal((2, 2, 8, 3))
        outputs = model(x, training=False)
        check("odd-block model output not None", outputs is not None)
        if outputs is not None:
            check("odd-block model returns two outputs", isinstance(outputs, list) and len(outputs) == 2, str(type(outputs)))
            check("odd-block restores input channel count", tuple(outputs[0].shape) == (2, 2, 8, 3), str(tuple(outputs[0].shape)))
            check("odd-block output tensors share shape", tuple(outputs[0].shape) == tuple(outputs[1].shape))
            check("odd-block output finite", bool(tf.reduce_all(tf.math.is_finite(outputs[0])).numpy()))
            check("odd-block includes encoder, center, and decoder SingleBlocks", sum(isinstance(layer, SingleBlock) for layer in model.layers) == 3, str([type(layer).__name__ for layer in model.layers]))
        else:
            skip_checks(5, "odd-block create_model returned None")
    except Exception as exc:
        skip_checks(6, f"odd-block create_model raised {type(exc).__name__}: {exc}")

    print("=" * 70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
