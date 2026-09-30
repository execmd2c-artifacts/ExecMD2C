"""Ablated template for the StrengthNet LLM reproduction benchmark.

This file intentionally contains only the architecture code needed to build
the CNN-BLSTM StrengthNet model. Training loops, data loading, feature
extraction, saved training artifacts, and evaluation utilities are excluded
from the benchmark surface.
"""

import tensorflow
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import Bidirectional, Conv2D, Dense, Dropout, LSTM, TimeDistributed


class CNN_BLSTM(object):
    def __init__(self):
        print('CNN_BLSTM init')

    def build(self):
        """[TODO] Reproduce the StrengthNet CNN-BLSTM architecture.

        Implement the complete model-building path from the original code:
        accept variable-length frame sequences with 80 input features, project
        them to 257 bins, reshape to a 2-D convolutional representation, apply
        four three-layer Conv2D blocks whose final convolution downsamples the
        frequency axis with stride 3 and channel sizes 16, 32, 64, and 128,
        reshape the convolutional tensor to per-frame 512-dimensional features,
        build one bidirectional LSTM branch for frame/average emotion-strength
        regression, build a second bidirectional LSTM branch for four-class
        utterance emotion classification, and return a Keras Model whose
        outputs are [average_score, frame_score, out_class] with layer names
        'avg', 'frame', and 'class' preserved.
        """
        pass


if __name__ == "__main__":
    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        global passed, failed
        if condition:
            print(f"PASS: {name}")
            passed += 1
        else:
            suffix = f" - {detail}" if detail else ""
            print(f"FAIL: {name}{suffix}")
            failed += 1

    def skip_checks(count, reason):
        global failed
        for index in range(count):
            print(f"FAIL: skipped dependent check {index + 1}/{count} - {reason}")
        failed += count

    print("StrengthNet benchmark checks: CNN_BLSTM.build architecture")

    construction_error = None
    try:
        builder = CNN_BLSTM()
        built_model = builder.build()
    except Exception as exc:
        built_model = None
        construction_error = f"CNN_BLSTM.build raised {type(exc).__name__}: {exc}"

    if construction_error is not None:
        skip_checks(10, construction_error)
    elif built_model is not None:
        check("build returns a keras Model", isinstance(built_model, Model), type(built_model).__name__)
        check("model has exactly three outputs", len(built_model.outputs) == 3, str(len(built_model.outputs)))
        check("input feature dimension is 80", built_model.input_shape[-1] == 80, str(built_model.input_shape))
        check("input keeps variable time dimension", built_model.input_shape[1] is None, str(built_model.input_shape))

        layer_names = [layer.name for layer in built_model.layers]
        conv_layers = [layer for layer in built_model.layers if isinstance(layer, Conv2D)]
        recurrent_wrappers = [layer for layer in built_model.layers if isinstance(layer, Bidirectional)]
        recurrent_units = [
            layer.forward_layer.units
            for layer in recurrent_wrappers
            if hasattr(layer, "forward_layer") and isinstance(layer.forward_layer, LSTM)
        ]

        check("contains twelve Conv2D layers", len(conv_layers) == 12, str(len(conv_layers)))
        check("uses four CNN channel sizes", [layer.filters for layer in conv_layers[::3]] == [16, 32, 64, 128],
              str([layer.filters for layer in conv_layers[::3]]))
        check("contains two bidirectional LSTM branches", len(recurrent_wrappers) == 2, str(len(recurrent_wrappers)))
        check("both LSTM branches use 128 units", recurrent_units == [128, 128], str(recurrent_units))
        check("frame and average score heads are named", "frame" in layer_names and "avg" in layer_names, str(layer_names))
        check("classification head predicts four emotion classes",
              "class" in layer_names and built_model.output_shape[2][-1] == 4, str(built_model.output_shape))
    else:
        check("build returns a keras Model", False, "returned None")
        skip_checks(9, "model is missing")

    print(f"RESULT: passed={passed} failed={failed}")
    if failed != 0:
        raise SystemExit(1)
