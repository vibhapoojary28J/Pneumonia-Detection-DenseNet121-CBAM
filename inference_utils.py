"""Reusable code for the Pneumonia Detection project (DenseNet121 + CBAM).

This module is the single source of truth for everything that must behave
identically in training, evaluation, Grad-CAM and the future Streamlit app:

* constants (image size, class names, layer names)
* serializable custom Keras layers (DenseNetPreprocess, ChannelAttention,
  SpatialAttention, CBAM)
* the model builder
* image preprocessing (bytes / path -> uint8 224x224x3 tensor)
* Grad-CAM
* single-image prediction (predict_xray)

Model contract
--------------
Input  : float32 tensor (batch, 224, 224, 3), RGB, raw pixel values 0..255.
         DenseNet preprocessing is part of the model, so callers must NOT
         normalise images themselves.
Output : float32 tensor (batch, 1) = sigmoid probability of PNEUMONIA.
         Label mapping: 0 = NORMAL, 1 = PNEUMONIA.

Research prototype only - not a medical device.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
IMG_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
CLASS_TO_INDEX = {"NORMAL": 0, "PNEUMONIA": 1}
INDEX_TO_CLASS = {0: "NORMAL", 1: "PNEUMONIA"}
DECISION_THRESHOLD = 0.5

PACKAGE = "pneumonia"
BACKBONE_NAME = "densenet121"
PREPROCESS_LAYER_NAME = "densenet_preprocess"
HEAD_LAYER_NAMES = ("cbam", "gap", "head_dropout", "classifier")

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "saved_models" / "best_pneumonia_model.keras"

# DenseNet121 in keras.applications uses "torch" preprocessing:
# scale to [0, 1], then subtract the ImageNet mean and divide by the ImageNet std.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Custom, serializable layers
# --------------------------------------------------------------------------- #
@tf.keras.utils.register_keras_serializable(package=PACKAGE)
class DenseNetPreprocess(keras.layers.Layer):
    """ImageNet 'torch-mode' preprocessing used by DenseNet121 (inside the model)."""

    def call(self, inputs):
        x = tf.cast(inputs, tf.float32) / 255.0
        mean = tf.constant(_IMAGENET_MEAN, dtype=tf.float32)
        std = tf.constant(_IMAGENET_STD, dtype=tf.float32)
        return (x - mean) / std


@tf.keras.utils.register_keras_serializable(package=PACKAGE)
class ChannelAttention(keras.layers.Layer):
    """CBAM channel attention: "what" to focus on.

    A shared two-layer MLP is applied to the global-average-pooled and the
    global-max-pooled descriptors; the sum goes through a sigmoid and rescales
    every channel of the input.
    """

    def __init__(self, reduction_ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = int(reduction_ratio)

    def build(self, input_shape):
        channels = int(input_shape[-1])
        hidden = max(channels // self.reduction_ratio, 1)
        self.mlp_hidden = keras.layers.Dense(hidden, activation="relu", name="mlp_hidden")
        self.mlp_out = keras.layers.Dense(channels, name="mlp_out")
        # Build sub-layers explicitly so their weights exist before a saved
        # model's weights are restored (required for reliable .keras loading).
        self.mlp_hidden.build((None, channels))
        self.mlp_out.build((None, hidden))
        super().build(input_shape)

    def call(self, inputs):
        avg_desc = tf.reduce_mean(inputs, axis=[1, 2])
        max_desc = tf.reduce_max(inputs, axis=[1, 2])
        avg_out = self.mlp_out(self.mlp_hidden(avg_desc))
        max_out = self.mlp_out(self.mlp_hidden(max_desc))
        scale = tf.nn.sigmoid(avg_out + max_out)
        return inputs * scale[:, tf.newaxis, tf.newaxis, :]

    def get_config(self):
        config = super().get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config


@tf.keras.utils.register_keras_serializable(package=PACKAGE)
class SpatialAttention(keras.layers.Layer):
    """CBAM spatial attention: "where" to focus.

    Channel-wise average and max maps are concatenated and passed through a
    7x7 convolution with sigmoid activation to give a (H, W, 1) attention map.
    """

    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = int(kernel_size)

    def build(self, input_shape):
        self.conv = keras.layers.Conv2D(
            1, self.kernel_size, padding="same", activation="sigmoid", name="spatial_conv"
        )
        self.conv.build(tuple(input_shape[:-1]) + (2,))
        super().build(input_shape)

    def call(self, inputs):
        avg_map = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_map = tf.reduce_max(inputs, axis=-1, keepdims=True)
        attention = self.conv(tf.concat([avg_map, max_map], axis=-1))
        return inputs * attention

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


@tf.keras.utils.register_keras_serializable(package=PACKAGE)
class CBAM(tf.keras.layers.Layer):
    """Convolutional Block Attention Module: channel attention, then spatial attention."""

    def __init__(self, reduction_ratio=16, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = int(reduction_ratio)
        self.kernel_size = int(kernel_size)

    def build(self, input_shape):
        self.channel_attention = ChannelAttention(self.reduction_ratio, name="channel_attention")
        self.spatial_attention = SpatialAttention(self.kernel_size, name="spatial_attention")
        self.channel_attention.build(input_shape)
        self.spatial_attention.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        x = self.channel_attention(inputs)
        return self.spatial_attention(x)

    def get_config(self):
        config = super().get_config()
        config.update({"reduction_ratio": self.reduction_ratio, "kernel_size": self.kernel_size})
        return config


CUSTOM_OBJECTS = {
    f"{PACKAGE}>DenseNetPreprocess": DenseNetPreprocess,
    f"{PACKAGE}>ChannelAttention": ChannelAttention,
    f"{PACKAGE}>SpatialAttention": SpatialAttention,
    f"{PACKAGE}>CBAM": CBAM,
    "DenseNetPreprocess": DenseNetPreprocess,
    "ChannelAttention": ChannelAttention,
    "SpatialAttention": SpatialAttention,
    "CBAM": CBAM,
}


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(weights="imagenet", dropout_rate=0.4, cbam_reduction=16, cbam_kernel=7):
    """DenseNet121 (ImageNet) -> CBAM -> GAP -> Dropout -> Dense(1, sigmoid)."""
    inputs = keras.Input(shape=INPUT_SHAPE, name="xray_input")
    x = DenseNetPreprocess(name=PREPROCESS_LAYER_NAME)(inputs)

    backbone = keras.applications.DenseNet121(
        include_top=False,
        weights=weights,
        input_shape=INPUT_SHAPE,
        name=BACKBONE_NAME,
    )
    x = backbone(x)
    x = CBAM(reduction_ratio=cbam_reduction, kernel_size=cbam_kernel, name="cbam")(x)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="classifier")(x)
    return keras.Model(inputs, outputs, name="DenseNet121_CBAM")


def load_trained_model(model_path=DEFAULT_MODEL_PATH):
    """Load the saved .keras model (custom layers are registered by importing this module)."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return keras.models.load_model(str(model_path), custom_objects=CUSTOM_OBJECTS, compile=False)


# --------------------------------------------------------------------------- #
# Image preprocessing (shared by training pipeline, evaluation, Grad-CAM, app)
# --------------------------------------------------------------------------- #
def preprocess_image_bytes(raw_bytes):
    """Encoded image bytes (JPEG/PNG/...) -> uint8 tensor (224, 224, 3), RGB.

    Grayscale images are expanded to 3 channels, RGBA is reduced to RGB.
    The DenseNet normalisation is NOT applied here; it is part of the model.
    """
    image = tf.io.decode_image(raw_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE, method="bilinear", antialias=True)
    return tf.cast(tf.round(tf.clip_by_value(image, 0.0, 255.0)), tf.uint8)


def load_image_tf(path):
    """File path (str or string tensor) -> uint8 tensor (224, 224, 3)."""
    return preprocess_image_bytes(tf.io.read_file(path))


def load_image_array(image_path):
    """Convenience wrapper for eager code: path -> numpy uint8 (224, 224, 3)."""
    image_path = Path(image_path).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    try:
        return preprocess_image_bytes(image_path.read_bytes()).numpy()
    except tf.errors.InvalidArgumentError as exc:
        raise ValueError(f"Could not decode '{image_path}' as an image.") from exc


# --------------------------------------------------------------------------- #
# Grad-CAM
# --------------------------------------------------------------------------- #
def make_gradcam_heatmap(model, image_batch, class_index=None):
    """Grad-CAM for the DenseNet121 + CBAM model.

    The nested DenseNet121 is NOT reached through the outer functional graph.
    Instead the forward pass is split by hand:

        conv_maps = backbone(preprocess(image))        # last DenseNet feature maps
        output    = head(conv_maps)                    # CBAM -> GAP -> Dropout -> Dense

    Gradients of the class score are taken w.r.t. ``conv_maps`` (the final
    spatial feature maps of the backbone, before CBAM and the classifier).

    Parameters
    ----------
    image_batch : float tensor/array (1, 224, 224, 3), raw 0..255 RGB values.
    class_index : None (= explain the predicted class), 0 (NORMAL) or 1 (PNEUMONIA).

    Returns
    -------
    heatmap (H, W) float32 in [0, 1], sigmoid probability of PNEUMONIA, info dict.
    """
    image_batch = tf.convert_to_tensor(image_batch, dtype=tf.float32)
    if image_batch.shape.rank != 4 or image_batch.shape[0] != 1:
        raise ValueError(f"Expected a batch of exactly one image, got shape {image_batch.shape}")

    preprocess = model.get_layer(PREPROCESS_LAYER_NAME)
    backbone = model.get_layer(BACKBONE_NAME)
    head_layers = [model.get_layer(name) for name in HEAD_LAYER_NAMES]

    conv_maps = backbone(preprocess(image_batch), training=False)

    # Probability and the class to explain (decided before the tape starts).
    probability = float(model(image_batch, training=False).numpy().reshape(-1)[0])
    target = int(probability >= DECISION_THRESHOLD) if class_index is None else int(class_index)

    with tf.GradientTape() as tape:
        tape.watch(conv_maps)
        h = conv_maps
        for layer in head_layers:
            if isinstance(layer, keras.layers.Dropout):
                h = layer(h, training=False)
            else:
                h = layer(h)
        prob_tensor = h[:, 0]
        score = prob_tensor if target == 1 else 1.0 - prob_tensor

    grads = tape.gradient(score, conv_maps)
    if grads is None:
        raise RuntimeError(
            "Grad-CAM gradients are None: the head is not connected to the backbone feature maps."
        )

    weights = tf.reduce_mean(grads, axis=(0, 1, 2))            # (channels,)
    cam = tf.reduce_sum(conv_maps[0] * weights, axis=-1)       # (H, W)
    cam = tf.nn.relu(cam)
    cam_max = float(tf.reduce_max(cam))
    cam = (cam / cam_max).numpy() if cam_max > 0 else np.zeros(cam.shape, dtype=np.float32)

    grads_np = grads.numpy()
    info = {
        "target_class": INDEX_TO_CLASS[target],
        "feature_map_shape": tuple(int(s) for s in conv_maps.shape[1:]),
        "grad_abs_mean": float(np.abs(grads_np).mean()),
        "grad_nonzero_fraction": float((grads_np != 0).mean()),
        "heatmap_is_empty": cam_max <= 0,
    }
    return cam.astype(np.float32), probability, info


def heatmap_to_rgb(cam, size=IMG_SIZE):
    """Upsample a low-resolution CAM to ``size`` and apply the 'jet' colormap."""
    import matplotlib

    cam_resized = tf.image.resize(np.asarray(cam, dtype=np.float32)[..., np.newaxis], size, method="bilinear")
    cam_resized = np.clip(cam_resized.numpy()[..., 0], 0.0, 1.0)
    colored = matplotlib.colormaps["jet"](cam_resized)[..., :3]
    return cam_resized, colored


def overlay_heatmap(image_uint8, cam, alpha=0.4):
    """Blend the colored heatmap over the (224, 224, 3) uint8 image; returns floats in [0, 1]."""
    _, colored = heatmap_to_rgb(cam, size=image_uint8.shape[:2])
    base = image_uint8.astype("float32") / 255.0
    return np.clip((1.0 - alpha) * base + alpha * colored, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Prediction (reusable from notebook and Streamlit)
# --------------------------------------------------------------------------- #
_MODEL_CACHE = {}


def get_model(model_path=DEFAULT_MODEL_PATH):
    """Load the model once and reuse it (handy for Streamlit with st.cache_resource)."""
    key = str(Path(model_path).resolve())
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_trained_model(model_path)
    return _MODEL_CACHE[key]


def predict_image_bytes(model, raw_bytes, threshold=DECISION_THRESHOLD):
    """Predict from encoded image bytes (e.g. an uploaded file in Streamlit)."""
    image = preprocess_image_bytes(raw_bytes)
    batch = tf.cast(image, tf.float32)[tf.newaxis, ...]
    probability = float(model(batch, training=False).numpy().reshape(-1)[0])
    label_index = int(probability >= threshold)
    return {
        "label": CLASS_NAMES[label_index],
        "label_index": label_index,
        "sigmoid_output": probability,
        "confidence": probability if label_index == 1 else 1.0 - probability,
        "threshold": float(threshold),
        "image": image.numpy(),
    }


def predict_xray(image_path, model=None, threshold=DECISION_THRESHOLD, show=True, model_path=DEFAULT_MODEL_PATH):
    """Predict NORMAL / PNEUMONIA for one chest X-ray image file.

    Uses exactly the same decoding / RGB conversion / resizing as the training
    pipeline; DenseNet normalisation happens inside the model.
    """
    image_path = Path(image_path).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if model is None:
        model = get_model(model_path)
    try:
        result = predict_image_bytes(model, image_path.read_bytes(), threshold=threshold)
    except tf.errors.InvalidArgumentError as exc:
        raise ValueError(f"Could not decode '{image_path}' as an image.") from exc

    image = result.pop("image")
    result["image_path"] = str(image_path)

    if show:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4.5, 4.8))
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(
            f"Prediction: {result['label']}\n"
            f"sigmoid (P-PNEUMONIA) = {result['sigmoid_output']:.4f}  |  threshold = {threshold}",
            fontsize=10,
        )
        plt.tight_layout()
        plt.show()
        print("Research prototype only - not a medical diagnosis.")
    return result
