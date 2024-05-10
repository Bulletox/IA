import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
# MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape, train_labels.shape)
# Normalization of data
normalized_train_images = train_images / 255
normalized_test_images = test_images / 255

# Convert labels to one-hot encoding
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

print('train_images.shape', train_images.shape)
print(train_images.shape[0], 'Train Sample')
print(test_images.shape[0], 'Test Sample')
# Build the model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model training
model.fit(normalized_train_images, one_hot_train_labels, epochs=5, batch_size=32)

# Model evaluation
model.evaluate(normalized_test_images, one_hot_test_labels)

# Model saving
model.save('mnist.h5')
print('Model saved')

# prediction
new_model = tf.keras.models.load_model('mnist.h5')
predictions = new_model.predict(normalized_test_images)
print(predictions)
print(np.argmax(predictions[5]))
plt.imshow((tf.squeeze(test_images[5])))
plt.show()
#predict on the first five images
pred=model.predict(normalized_test_images[:5])
print(np.argmax(pred, axis=1))
print(test_labels[:5])
for i in range (0,5):
  first_img=test_images[i]
  first_img=np.array(first_img,dtype='float')
  pixels=first_img.reshape((28,28))
#   print(predictions[i])
  plt.imshow(pixels, cmap='gray')
  plt.show()
model.delete('mnist.h5')
