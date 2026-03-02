import os
import numpy as np

# 1. Set backend to JAX
os.environ["KERAS_BACKEND"] = "jax"

# 2. Prevent TensorFlow from stealing JAX's GPU memory!
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import keras
from keras import layers
import jax

def main():
    # 3. Detect the 2 local Quadro RTX 5000 GPUs
    devices = jax.devices("gpu")
    print(f"Found {len(devices)} GPUs: {devices}")

    # Create dummy data
    inputs = np.random.normal(size=(128, 28, 28, 1))
    labels = np.random.normal(size=(128, 10))
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(16)

    print("\n--- Running DATA PARALLELISM ---")
    # DataParallel automatically detects your 2 GPUs and replicates the model
    data_parallel = keras.distribution.DataParallel(devices=devices)
    keras.distribution.set_distribution(data_parallel)

    # Build and train the model (Math happens seamlessly across both GPUs!)
    model_dp = build_model()
    model_dp.compile(loss="mse")
    model_dp.fit(dataset, epochs=2)

    print("\n--- Running MODEL PARALLELISM ---")
    # Define a 1x2 mesh (1 Data dimension, 2 Model dimensions)
    # This splits the model weights across your 2 GPUs.
    mesh_1x2 = keras.distribution.DeviceMesh(
        shape=(1, 2), axis_names=["data", "model"], devices=devices
    )
    
    # Define how specific layers are sharded across the GPUs
    layout_map = keras.distribution.LayoutMap(mesh_1x2)
    layout_map["dense_1/kernel"] = (None, "model") # Shard this kernel
    layout_map["dense_1/bias"] = ("model",)        # Shard this bias
    
    model_parallel = keras.distribution.ModelParallel(
        layout_map, batch_dim_name="data"
    )
    keras.distribution.set_distribution(model_parallel)

    model_mp = build_model()
    model_mp.compile(loss="mse")
    model_mp.fit(dataset, epochs=2)
    print("Demo Complete!")

def build_model():
    """Simple CNN for demonstration"""
    inputs = layers.Input(shape=(28, 28, 1))
    y = layers.Flatten()(inputs)
    y = layers.Dense(units=200, use_bias=False, activation="relu", name="dense_1")(y)
    y = layers.Dropout(0.4)(y)
    y = layers.Dense(units=10, activation="softmax", name="dense_2")(y)
    return keras.Model(inputs=inputs, outputs=y)

if __name__ == "__main__":
    main()