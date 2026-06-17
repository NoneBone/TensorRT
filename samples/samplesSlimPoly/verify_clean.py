#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify ONNX model numerical consistency before TensorRT deployment
"""

import os
import glob
import numpy as np
from PIL import Image
import onnxruntime as ort


MODEL_ORIG = "mnist.onnx"
MODEL_CLEAN = "mnist.clean.onnx"
IMG_DIR = "."
IMG_PATTERN = "*.pgm"

INPUT_NAME = "Input3"
OUTPUT_NAME = "Plus214_Output_0"

TOLERANCE_MSE = 1e-6
TOLERANCE_COS = 1e-5


def preprocess(pgm_path):
    img = Image.open(pgm_path)
    img = img.resize((28, 28))
    arr = np.array(img, dtype=np.float32)

    # MNIST normalization (CNTK style)
    arr = arr / 255.0
    arr = arr.reshape(1, 1, 28, 28)
    return arr


def infer(session, input_data):
    return session.run(
        [OUTPUT_NAME],
        {INPUT_NAME: input_data}
    )[0]


def cosine_similarity(a, b):
    a = a.flatten()
    b = b.flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    assert os.path.exists(MODEL_ORIG), f"{MODEL_ORIG} not found"
    assert os.path.exists(MODEL_CLEAN), f"{MODEL_CLEAN} not found"

    sess_orig = ort.InferenceSession(
        MODEL_ORIG,
        providers=["CPUExecutionProvider"]
    )
    sess_clean = ort.InferenceSession(
        MODEL_CLEAN,
        providers=["CPUExecutionProvider"]
    )

    pgm_files = sorted(glob.glob(os.path.join(IMG_DIR, IMG_PATTERN)))
    if len(pgm_files) == 0:
        raise RuntimeError("No PGM files found")

    print(f"Found {len(pgm_files)} test images\n")

    all_pass = True

    for pgm in pgm_files:
        x = preprocess(pgm)

        y_orig = infer(sess_orig, x)
        y_clean = infer(sess_clean, x)

        mse = np.mean((y_orig - y_clean) ** 2)
        cos = cosine_similarity(y_orig, y_clean)

        pred_orig = y_orig.argmax()
        pred_clean = y_clean.argmax()

        status = (
            "✅ PASS"
            if (
                mse < TOLERANCE_MSE and
                abs(1.0 - cos) < TOLERANCE_COS and
                pred_orig == pred_clean
            )
            else "❌ FAIL"
        )

        if status == "❌ FAIL":
            all_pass = False

        print(f"{os.path.basename(pgm):<6} | "
              f"Orig={pred_orig}, Clean={pred_clean} | "
              f"MSE={mse:.2e}, Cos={cos:.8f} | {status}")

    print("\n" + "=" * 60)
    if all_pass:
        print("✅ All tests passed. Model is numerically consistent.")
    else:
        print("❌ Some tests failed. Check model optimization steps.")


if __name__ == "__main__":
    main()