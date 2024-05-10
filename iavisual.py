import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Actualmente, la memoria no se asigna previamente; la memoria crece según sea necesario.
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPU")
#     except RuntimeError as e:
#         # La inicialización de GPU debe ser después de que se hayan establecido las GPUs
#         print(e)
# else:
#     print("No GPU devices available.")
# Carga del dataset MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalización de los datos
normalized_train_images = train_images / 255.0
normalized_test_images = test_images / 255.0

# Convertir las etiquetas a codificación one-hot
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

# Construcción del modelo
model = Sequential([Flatten(input_shape=(28, 28)),Dense(128, activation='relu'),Dense(128, activation='relu'),Dense(10, activation='softmax')])

# Compilación del modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento del modelo
model.fit(normalized_train_images, one_hot_train_labels, epochs=10, batch_size=250)

# Evaluación del modelo
test_loss, test_acc = model.evaluate(normalized_test_images, one_hot_test_labels)
print('\nTest accuracy:', test_acc)

# Guardar el modelo
model.save('mnist.h5')

# Cargar el modelo
new_model = tf.keras.models.load_model('mnist.h5')

# Realizar predicciones
predictions = new_model.predict(normalized_test_images)

# Función para visualizar la imagen, la predicción y la etiqueta real
def plot_image_prediction(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img, cmap=plt.cm.binary)
    
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                        100*np.max(predictions_array),
                                        true_label),
                                        color=color)

# Visualizar las primeras 5 imágenes y sus predicciones
num_rows = 5
num_cols = 2
num_images = num_rows * num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(5):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image_prediction(i, predictions[i], test_labels, test_images)
plt.tight_layout()
plt.show()
