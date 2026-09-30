#!/usr/bin/env python3
"""ICIAR2018 core CNN encoder model components."""

# ============================================================
# ground_truth.py - ICIAR2018 Core Model Components
# Source: Medical/ICIAR2018-master/models.py
#
# Contains ONLY the CNN encoder architecture wrappers used for
# feature extraction. No training, dataset, blending, or submission code.
# ============================================================

from keras.applications.resnet50 import ResNet50, preprocess_input as preprocess_resnet
from keras.applications.inception_v3 import InceptionV3, preprocess_input as preprocess_inception
from keras.applications.vgg16 import VGG16, preprocess_input as preprocess_vgg
from keras import backend as K
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Concatenate
import numpy as np


# --- [Original file: models.py] ---
class ResNet:
    __name__ = "ResNet"

    def __init__(self, batch_size=32):
        self.model = ResNet50(include_top=False, weights='imagenet', pooling="avg")
        self.batch_size = batch_size
        self.data_format = K.image_data_format()

    def predict(self, x):
        if self.data_format == "channels_first":
            x = x.transpose(0, 3, 1, 2)
        x = preprocess_resnet(x.astype(K.floatx()))
        return self.model.predict(x, batch_size=self.batch_size)


class Inception:
    __name__ = "Inception"

    def __init__(self, batch_size=32):
        self.model = InceptionV3(include_top=False, weights="imagenet", pooling="avg")
        self.batch_size = batch_size
        self.data_format = K.image_data_format()

    def predict(self, x):
        if self.data_format == "channels_first":
            x = x.transpose(0, 3, 1, 2)
        x = preprocess_inception(x.astype(K.floatx()))
        return self.model.predict(x, batch_size=self.batch_size)


class VGG:
    __name__ = "VGG"

    def __init__(self, batch_size=32):
        model = VGG16(include_top=False, weights="imagenet", pooling="avg")
        x2 = GlobalAveragePooling2D()(model.get_layer("block2_conv2").output)  # 128
        x3 = GlobalAveragePooling2D()(model.get_layer("block3_conv3").output)  # 256
        x4 = GlobalAveragePooling2D()(model.get_layer("block4_conv3").output)  # 512
        x5 = GlobalAveragePooling2D()(model.get_layer("block5_conv3").output)  # 512
        x = Concatenate()([x2, x3, x4, x5])
        self.model = Model(inputs=model.input, outputs=x)
        self.batch_size = batch_size
        self.data_format = K.image_data_format()

    def predict(self, x):
        if self.data_format == "channels_first":
            x = x.transpose(0, 3, 1, 2)
        x = preprocess_vgg(x.astype(K.floatx()))
        return self.model.predict(x, batch_size=self.batch_size)


# ============================================================
# __main__: Automated test suite for 3 ablated functions
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

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

    class _FakeKerasModel:
        def __init__(self, output_dim):
            self.output_dim = output_dim
            self.seen_input = None
            self.seen_batch_size = None

        def predict(self, x, batch_size=None):
            self.seen_input = x
            self.seen_batch_size = batch_size
            return np.full((x.shape[0], self.output_dim), 0.5, dtype=np.float32)

    def _make_preprocess_recorder(name, calls):
        def _preprocess(x):
            calls.append((name, x.copy()))
            return x + np.array(0.25, dtype=x.dtype)
        return _preprocess

    def _build_encoder(encoder_cls, output_dim, data_format, batch_size):
        encoder = object.__new__(encoder_cls)
        encoder.model = _FakeKerasModel(output_dim)
        encoder.batch_size = batch_size
        encoder.data_format = data_format
        return encoder

    def _run_predict_test(encoder_cls, preprocess_global, output_dim, data_format, expected_shape):
        global preprocess_resnet, preprocess_inception, preprocess_vgg

        original_preprocess = globals()[preprocess_global]
        calls = []
        globals()[preprocess_global] = _make_preprocess_recorder(preprocess_global, calls)
        try:
            batch_size = 3
            encoder = _build_encoder(encoder_cls, output_dim, data_format, batch_size)
            x = np.arange(2 * 8 * 6 * 3, dtype=np.uint8).reshape(2, 8, 6, 3)
            y = encoder.predict(x)

            name = encoder_cls.__name__
            check(f"{name}.predict output not None", y is not None)
            if y is not None:
                check(f"{name}.predict output shape", y.shape == (2, output_dim),
                      f"expected {(2, output_dim)}, got {getattr(y, 'shape', None)}")
                check(f"{name}.predict output finite", np.isfinite(y).all())
                check(f"{name}.predict batch size forwarded", encoder.model.seen_batch_size == batch_size,
                      f"expected {batch_size}, got {encoder.model.seen_batch_size}")
                check(f"{name}.predict preprocess called once", len(calls) == 1,
                      f"expected 1 call, got {len(calls)}")
                if encoder.model.seen_input is not None and len(calls) == 1:
                    check(f"{name}.predict input shape", encoder.model.seen_input.shape == expected_shape,
                          f"expected {expected_shape}, got {encoder.model.seen_input.shape}")
                    check(f"{name}.predict input dtype", encoder.model.seen_input.dtype == np.dtype(K.floatx()),
                          f"expected {np.dtype(K.floatx())}, got {encoder.model.seen_input.dtype}")
                    expected_preprocessed = calls[0][1] + np.array(0.25, dtype=calls[0][1].dtype)
                    check(f"{name}.predict preprocessed data forwarded",
                          np.array_equal(encoder.model.seen_input, expected_preprocessed))
                else:
                    skip_checks(3, f"{name}.predict did not provide preprocessed input")
            else:
                skip_checks(7, f"{name}.predict returned None")
        except Exception as exc:
            skip_checks(8, f"{encoder_cls.__name__}.predict raised {type(exc).__name__}: {exc}")
        finally:
            globals()[preprocess_global] = original_preprocess

    print("=" * 70)
    print("ICIAR2018: CNN Encoder Wrapper Benchmark")
    print("Automated Test Suite - 3 ablated functions")
    print("=" * 70)
    print()

    print("-" * 60)
    print("[Test 1/3] ResNet.predict - channel-first preprocessing and batched descriptor extraction")
    _run_predict_test(ResNet, "preprocess_resnet", 2048, "channels_first", (2, 3, 8, 6))
    print()

    print("-" * 60)
    print("[Test 2/3] Inception.predict - channel-last preprocessing and batched descriptor extraction")
    _run_predict_test(Inception, "preprocess_inception", 2048, "channels_last", (2, 8, 6, 3))
    print()

    print("-" * 60)
    print("[Test 3/3] VGG.predict - descriptor extraction for concatenated multi-layer VGG encoder")
    _run_predict_test(VGG, "preprocess_vgg", 1408, "channels_last", (2, 8, 6, 3))
    print()

    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed == 0:
        print("All tests PASSED! The model code is complete and correct.")
    else:
        print(f"{failed} check(s) FAILED - some [TODO] functions may not be implemented correctly.")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
