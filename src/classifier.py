"""
Classification utility functions.
Handles building models, transfer learning setups, and class weights calculation.
"""

import numpy as np
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras import applications, layers, models, optimizers


def calculate_class_weights(dataset: tf.data.Dataset) -> dict[int, float]:
    """
    Calculates class weights from a TensorFlow tf.data.Dataset to handle class imbalance.

    Parameters:
    -----------
    dataset : tf.data.Dataset
        The training image dataset (must yield features and integer labels).

    Returns:
    --------
    class_weights : dict
        A dictionary mapping integer class indices to balanced weight multipliers.
    """
    # Concatenate all label batches to calculate overall distribution
    labels = np.concatenate([y for x, y in dataset], axis=0)
    unique_classes = np.unique(labels)
    weights_array = class_weight.compute_class_weight("balanced", classes=unique_classes, y=labels)
    return dict(enumerate(weights_array))


def build_efficientnet_model(
    img_size: tuple[int, int],
    num_classes: int,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.4,
    use_augmentation: bool = True,
) -> tf.keras.Model:
    """
    Constructs an EfficientNetB0 model using transfer learning with custom top classification layers.

    Parameters:
    -----------
    img_size : tuple of int (height, width)
        Expected shape of the input image features.
    num_classes : int
        Number of output categories for classification.
    learning_rate : float
        Initial learning rate for the Adam optimizer.
    dropout_rate : float
        Regularization dropout rate on the dense projection layer.
    use_augmentation : bool
        If True, embeds an image-augmentation layer prefix in the model inputs.

    Returns:
    --------
    model : tf.keras.Model
        The compiled Keras Model.
    """
    # 1. Base augmentation pipeline
    augmentation_layers = []
    if use_augmentation:
        augmentation_layers = [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.3),
            layers.RandomZoom(0.3),
            layers.RandomContrast(0.3),
        ]
    data_augmentation = tf.keras.Sequential(augmentation_layers, name="data_augmentation")

    # 2. Pre-trained Base Feature Extractor
    base_model = applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(img_size[0], img_size[1], 3)
    )
    base_model.trainable = False

    # 3. Model Architecture
    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    # 4. Compilation
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
