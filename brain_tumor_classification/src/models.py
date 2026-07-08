"""
Model architectures: a from-scratch CNN baseline and a transfer-learning model.

Both are binary classifiers with a single sigmoid output (P(class = "yes")),
which pairs with tf.keras.losses.BinaryCrossentropy and lets class_weight
be passed straight into model.fit.
"""
import tensorflow as tf
from tensorflow.keras import layers, models

from .preprocessing import IMG_SIZE


def build_simple_cnn(input_shape=(*IMG_SIZE, 3)) -> tf.keras.Model:
    """
    A small CNN trained from random initialization: 3 conv blocks
    (conv -> batchnorm -> relu -> maxpool) followed by global average
    pooling and a dense head.

    Purpose in this assignment: a *baseline*, not a competitor. With ~280
    total images and no pretrained knowledge of edges/textures/shapes, a
    from-scratch CNN has to learn every visual feature from a training set
    that's smaller than a single ImageNet class. We expect it to overfit or
    plateau well below the transfer-learning model -- that gap is the point,
    and the report should show and discuss it rather than treat it as a
    failure.

    Design choices:
    - GlobalAveragePooling2D instead of Flatten + big Dense layer: far fewer
      parameters, which matters when there are only ~280 training images to
      constrain them (a Flatten+Dense head here would have orders of
      magnitude more weights than training examples).
    - BatchNorm after each conv: stabilizes training given the small,
      noisy-gradient batches this dataset forces us into.
    - Dropout before the final layer: an explicit overfitting brake, since
      this model has no other regularization (no pretrained weights acting
      as a prior).
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="simple_cnn_baseline")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def build_transfer_model(
    input_shape=(*IMG_SIZE, 3),
    backbone: str = "mobilenetv2",
    freeze_base: bool = True,
    fine_tune_last_n: int = 0,
    learning_rate: float = 1e-4,
) -> tf.keras.Model:
    """
    Pretrained backbone (ImageNet weights) + a small custom classification
    head, following the standard transfer-learning recipe.

    Why transfer learning is expected to win here: MobileNetV2/VGG16 already
    encode general-purpose low/mid-level visual features (edges, textures,
    gradients, shapes) learned from 1.4M ImageNet images. With only ~280 MRI
    images we can't learn those features from scratch reliably -- transfer
    learning lets us reuse them and train only a small head, which is a much
    easier optimization problem for this amount of data.

    freeze_base=True (default) keeps the backbone's ImageNet weights fixed
    and only trains the new head -- appropriate first step given how little
    data we have relative to the backbone's parameter count. fine_tune_last_n
    optionally unfreezes the last N backbone layers for a low-learning-rate
    fine-tuning pass after the head has already converged (see notebook
    Section 5b) -- unfreezing everything immediately, before the head is
    trained, would let large random-head gradients destroy pretrained
    weights early in training.
    """
    if backbone == "mobilenetv2":
        base = tf.keras.applications.MobileNetV2(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
    elif backbone == "vgg16":
        base = tf.keras.applications.VGG16(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    base.trainable = not freeze_base
    if not freeze_base and fine_tune_last_n > 0:
        for layer in base.layers[:-fine_tune_last_n]:
            layer.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False)  # training=False keeps BatchNorm stats frozen
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name=f"transfer_{backbone}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def get_callbacks(checkpoint_path: str, monitor: str = "val_loss", patience: int = 8):
    """
    Early stopping (on val_loss, restore_best_weights) + checkpointing.
    monitor=val_loss rather than val_accuracy because, on a dataset this
    imbalanced, accuracy can plateau/tie easily while loss keeps
    distinguishing genuinely-better epochs.
    """
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=patience, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor=monitor, save_best_only=True, verbose=1
        ),
    ]
