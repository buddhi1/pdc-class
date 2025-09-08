
import tensorflow as tf
import os
import json as json
from tensorflow.python.client import device_lib
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# print GPU info
print(device_lib.list_local_devices())
print("\nTensorflow Version: " + tf.__version__)

# synchronous distributed training on multiple GPUs on one machine
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

print(f'batch size: {BATCH_SIZE}')

# Open a strategy scope
with strategy.scope():
    model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(10000, activation='relu'),
  tf.keras.layers.Dense(9000, activation='relu'),
  tf.keras.layers.Dense(10000, activation='relu'),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
checkpoint_path = "training_1/cp.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


model.fit(x_train, y_train, epochs=2, batch_size=BATCH_SIZE, callbacks=[cp_callback])

model.evaluate(x_test,  y_test, verbose=2)
