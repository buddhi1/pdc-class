# TensorFlow Introduction (HPML)

Two Google Colab notebooks that introduce TensorFlow 2 and Keras before the distributed-training material.

| Notebook | Contents |
|----------|----------|
| `TensorFlow_Introduction_HPML_Spring_2025.ipynb` | **Tensors:** `tf.Variable` vs `tf.constant`, rank (`tf.rank`), shape, `tf.reshape`, slicing. **Neural-network concepts:** layers, weights, biases, activation functions, loss, backpropagation, optimizers. **First model:** a Fashion-MNIST classifier (Flatten → Dense 128 → Dense 10) built, compiled, trained, evaluated and used for predictions. Also includes the computer-vision section below, a `tf.GradientTape` custom-training example, and checks for an available GPU. |
| `Computer_Vision.ipynb` | **CNNs:** filters, padding, stride, pooling, feature maps. A Conv/MaxPool CNN on CIFAR-10 (about 70% accuracy). **Small datasets:** data augmentation with `ImageDataGenerator`, and **transfer learning** with a frozen MobileNetV2 base, `GlobalAveragePooling2D` and a single-unit head on `cats_vs_dogs`. |

## Running

Open the notebooks in Google Colab, or in Jupyter with `tensorflow`, `tensorflow_datasets` and `matplotlib` installed.

Some cells use older APIs:

- `%tensorflow_version 2.x` works only on Colab. Skip it elsewhere.
- `RMSprop(lr=...)` is now `learning_rate=...`.
- `keras.preprocessing.image.ImageDataGenerator` is not available in Keras 3. Use `tf_keras` or `keras.layers.Random*` augmentation layers instead.
