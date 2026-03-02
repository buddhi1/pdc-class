import os
import time
import numpy as np

# Ensure we use TensorFlow as the backend if Keras 3 is installed
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras

def main():
    # Verify the available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Found {len(gpus)} GPUs: {gpus}")

    # 1. Initialize CentralStorageStrategy
    # Note: It is in the 'experimental' module but is stable for this use case
    strategy = tf.distribute.experimental.CentralStorageStrategy()
    print(f"Number of compute replicas in sync: {strategy.num_replicas_in_sync}")

    # 2. Create dummy data
    # We use a larger batch size to keep the GPUs busy
    global_batch_size = 128 
    inputs = np.random.normal(size=(1024, 28, 28, 1)).astype(np.float32)
    labels = np.random.randint(0, 10, size=(1024, 1)).astype(np.float32)
    
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(global_batch_size)

    print("\n--- Building Model on Central Storage (CPU) ---")
    # 3. Build and compile the model INSIDE the strategy scope
    with strategy.scope():
        # Under the hood, TF assigns these variables to the CPU
        inputs_layer = keras.layers.Input(shape=(28, 28, 1))
        x = keras.layers.Conv2D(32, 3, activation='relu')(inputs_layer)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        outputs = keras.layers.Dense(10)(x)
        
        model = keras.Model(inputs=inputs_layer, outputs=outputs)
        
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy']
        )

    print("Model built successfully. Variables are stored centrally!")
    
    # 4. Train the model (Compute happens on GPUs)
    print("\n--- Starting Training ---")
    start_time = time.time()
    
    model.fit(dataset, epochs=5)
    
    print(f"\nTraining completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()