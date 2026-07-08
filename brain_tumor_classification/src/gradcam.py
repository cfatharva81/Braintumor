"""
Grad-CAM: visualizes which regions of an MRI image most influenced the
model's prediction, by looking at how strongly the gradient of the
predicted class flows back into the last convolutional feature map.

Used for the "analyze results" part of the report: it turns "the model
said tumor" into "the model said tumor because of activity in this region,"
which is what makes a classifier's output defensible/interpretable rather
than a black box, and is standard practice to mention for any medical
imaging classifier.
"""
import matplotlib as mpl
import numpy as np
import tensorflow as tf


def find_last_conv_layer(model: tf.keras.Model) -> str:
    """
    Walks the model backwards and returns the name of the last layer with a
    4D output (batch, h, w, channels) -- i.e. the last conv-like feature map
    before the classification head flattens/pools it away. Grad-CAM needs a
    spatial feature map, so this must run before GlobalAveragePooling.
    """
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            return layer.name
        # For a wrapped backbone (Model-as-a-layer, e.g. base(inputs)),
        # recurse into its sublayers.
        if hasattr(layer, "layers"):
            for sublayer in reversed(layer.layers):
                if len(sublayer.output_shape) == 4:
                    return sublayer.name
    raise ValueError("No 4D (conv-like) layer found in this model.")


def make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model, last_conv_layer_name: str = None):
    """
    img_array: single preprocessed image, shape (1, H, W, 3), already run
    through the same normalization used at training time.

    Returns a 2D heatmap in [0, 1], resized to the conv layer's native
    resolution (upsampling to the original image size happens in
    overlay_gradcam).
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    # Build a model mapping input -> (last conv feature map, model output).
    # Search top-level layers first, then inside a wrapped backbone.
    conv_layer = None
    for layer in model.layers:
        if layer.name == last_conv_layer_name:
            conv_layer = layer
            break
        if hasattr(layer, "layers"):
            for sublayer in layer.layers:
                if sublayer.name == last_conv_layer_name:
                    conv_layer = sublayer
                    break

    grad_model = tf.keras.models.Model(
        inputs=model.inputs, outputs=[conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array, training=False)
        # Binary sigmoid output: use the single logit/probability directly
        # as the "class score" whose gradient we backprop.
        class_score = predictions[:, 0]

    grads = tape.gradient(class_score, conv_output)
    # Global-average the gradients over height/width -> one importance
    # weight per channel, then weight each channel of the feature map by it.
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)  # ReLU + normalize
    return heatmap.numpy()


def overlay_gradcam(original_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    original_img: uint8 or float [0,255] array, shape (H, W, 3).
    Resizes the heatmap to match, applies the 'jet' colormap, and alpha-blends.
    """
    heatmap_resized = tf.image.resize(heatmap[..., tf.newaxis], original_img.shape[:2]).numpy()
    heatmap_resized = np.squeeze(heatmap_resized)

    jet = mpl.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[np.uint8(255 * heatmap_resized)]

    original = np.asarray(original_img, dtype=np.float32)
    if original.max() <= 1.0:
        original = original * 255.0

    overlaid = jet_heatmap * 255 * alpha + original * (1 - alpha)
    return np.uint8(np.clip(overlaid, 0, 255))
