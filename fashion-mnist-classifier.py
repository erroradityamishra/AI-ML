
import tensorflow as tf
print(tf.__version__)
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
import matplotlib.pyplot as plt
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])
training_images  = training_images / 255.0
test_images = test_images / 255.0
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
  tf.keras.layers.Dense(128, activation=tf.nn.relu), 
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)
classifications = model.predict(test_images)
print(classifications[0])


# Justification of the Code's Process and Uses:
# Library Imports:

# The code imports TensorFlow and Matplotlib. TensorFlow is used for building and training the neural network, while Matplotlib is used for visualizing the images.
# Loading the Dataset:

# The Fashion MNIST dataset is loaded, which contains grayscale images of clothing items categorized into 10 classes (e.g., T-shirts, trousers, etc.).
# The dataset is split into training and test sets.
# Data Normalization:

# The pixel values of the images are normalized by dividing by 255.0. This scales the values to a range of 0 to 1, which helps the model learn more effectively.
# Model Building:

# A Sequential model is created with:
# A Flatten layer to convert the 2D images into a 1D array.
# A Dense layer with 128 neurons and ReLU activation function to learn complex patterns.
# A final Dense layer with 10 neurons (one for each class) and softmax activation to output probabilities for each class.
# Model Compilation:

# The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss function (suitable for integer labels), and accuracy as a metric.
# Model Training:

# The model is trained on the training images and labels for 5 epochs, during which it learns to classify the images.
# Model Evaluation:

# After training, the model is evaluated on the test dataset to determine its accuracy.
# Predictions:

# The model makes predictions on the test images, providing the predicted class probabilities for the first test image.
# Use Cases:
# Educational Purposes: This code serves as an excellent introduction to machine learning and neural networks for beginners.
# Fashion Industry Applications: It can be adapted for real-world applications in fashion retail for inventory management, recommendation systems, or automated tagging of clothing items.
# Research and Development: It provides a foundation for experimenting with different architectures, optimizers, and hyperparameters in image classification tasks.
# Conclusion:
# This code effectively demonstrates the basic workflow of a machine learning project using TensorFlow, from data loading and preprocessing to model training and evaluation. It's a practical example for anyone looking to understand or implement image classification using deep learning techniques.
