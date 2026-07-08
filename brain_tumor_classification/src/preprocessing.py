"""
Image loading, resizing, channel handling, and augmentation pipelines.

Design decisions (for the viva):
- Target size 224x224: this is the input size MobileNetV2/VGG16 were
  pretrained on. Using anything else means the pretrained filters see
  feature-map shapes they never trained on, which weakens transfer learning.
- Grayscale -> 3 channel: pretrained backbones expect 3-channel RGB input
  because ImageNet is RGB. Some MRI scans in this dataset are single-channel
  grayscale, so we stack the channel 3x rather than dropping the backbone's
  first conv layer (dropping/re-initializing it would throw away the
  pretrained low-level edge/texture filters we're trying to reuse).
- Augmentation is applied as Keras layers *inside* the tf.data pipeline
  (not baked into the saved image files) so every epoch sees a fresh random
  transform of the same source image instead of a fixed set of copies.
- Two augmentation strengths exist (standard vs heavy) because we
  deliberately augment the minority ("no") class harder than the majority
  ("yes") class during training -- see dataset.py for how they're combined.
"""
import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)


def load_and_resize(path: str, target_size=IMG_SIZE) -> np.ndarray:
    """
    Decode an image file, force it to 3-channel RGB (stacking grayscale if
    needed), and resize to target_size. Returns a float32 array in [0, 255].

    Using tf.io/tf.image (not PIL) here so this function can be wrapped in a
    tf.data pipeline with tf.py_function/tf.numpy_function or, as done in
    dataset.py, used directly as a tf.io-native decode step.
    """
    raw = tf.io.read_file(path)
    # expand_animations=False avoids a shape-rank surprise on GIF-like files;
    # channels=3 forces grayscale/RGBA images to 3-channel RGB automatically.
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, target_size, method="bilinear")
    img.set_shape([target_size[0], target_size[1], 3])
    return img


def normalize_simple(img):
    """Scale to [0, 1]. Used for the from-scratch CNN baseline."""
    return img / 255.0


def normalize_for_backbone(img, backbone: str):
    """
    Apply the exact preprocessing each pretrained backbone was trained with.
    Using the wrong normalization (e.g. plain /255 for a backbone trained on
    ImageNet-mean-subtracted or [-1,1]-scaled input) silently degrades
    transfer learning performance -- the pretrained weights expect a specific
    input distribution.
    """
    if backbone == "mobilenetv2":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    elif backbone == "vgg16":
        from tensorflow.keras.applications.vgg16 import preprocess_input
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return preprocess_input(img)


def build_standard_augmentation() -> tf.keras.Sequential:
    """
    Light augmentation applied to the majority ("yes") class.
    Kept mild because 259 real examples already give the model reasonable
    coverage of natural variation -- we don't want to distort the majority
    class enough to blur the decision boundary.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),   # ~18 degrees max
            tf.keras.layers.RandomZoom(0.05),
        ],
        name="standard_augmentation",
    )


def build_heavy_augmentation() -> tf.keras.Sequential:
    """
    Stronger, more varied augmentation applied to the minority ("no") class.
    With only ~22 source images, the model would otherwise memorize this
    exact handful and generalize poorly. Wider ranges + more transform types
    (rotation, flip, zoom, contrast, brightness, shear-via-translation) give
    each source image many more distinct-looking training variants, which is
    combined with oversampling in dataset.py to shrink the effective class
    gap before class weights are even applied.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.15),        # ~54 degrees max
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomTranslation(0.1, 0.1),  # approximates shear
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.RandomBrightness(0.2, value_range=(0, 255)),
        ],
        name="heavy_minority_augmentation",
    )
