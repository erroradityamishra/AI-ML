import tensorflow as tf
import numpy as np
from tensorflow import keras
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)
model.fit(xs, ys, epochs=500)
print(model.predict([10.0]))



# Code Explanation:
# Imports: The code imports TensorFlow and NumPy.
# Model Creation: A sequential model is created with a single dense layer.
# Compilation: The model is compiled using stochastic gradient descent (SGD) as the optimizer and mean squared error as the loss function.
# Data Preparation: Input (xs) and output (ys) data are defined.
# Model Training: The model is trained for 500 epochs on the provided data.
# Prediction: The model predicts the output for the input value 10.0.
