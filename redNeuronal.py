import tensorflow as ts
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard

#desacargar catalogo
datos, metadatos = tfds.load('cats_vs_dogs', as_supervised=True, with_info=True)

plt.figure(figsize=(20,20))

TAMANO_IMG=120

datos_entrenamiento =[]

for i, (imagen, etiqueta) in enumerate(datos['train']): #todos los datos
  imagen= cv2.resize(imagen.numpy(),(TAMANO_IMG,TAMANO_IMG))
  imagen =cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
  imagen = imagen.reshape(TAMANO_IMG,TAMANO_IMG,1)
  datos_entrenamiento.append([imagen,etiqueta])


x=[]
y=[]


for imagen, etiqueta in datos_entrenamiento:
  x.append(imagen)
  y.append(etiqueta)


x = np.array(x).astype(float)/255
y=np.array(y)

#


#importar el generador con aumentos de datos
datagen= ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=15,
    zoom_range=[0.5, 1.5],
    horizontal_flip=True,
    vertical_flip=True
)

datagen.fit(x)

plt.figure(figsize=(20,8))

for imagen, etiqueta in datagen.flow(x,y, batch_size=10, shuffle=False):
  for i in range(10):
    plt.subplot(2,5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(imagen[i].reshape(120,120),cmap="gray")
  break

modeloCNN_AD =ts.keras.models.Sequential([
   ts.keras.layers.Conv2D(32,(3,3), activation='relu', input_shape=(120,120,1)),
   ts.keras.layers.MaxPooling2D(2,2),
   ts.keras.layers.Conv2D(64,(3,3), activation='relu'),
   ts.keras.layers.MaxPooling2D(2,2), 
   ts.keras.layers.Conv2D(128,(3,3), activation='relu'),
   ts.keras.layers.MaxPooling2D(2,2),

   ts.keras.layers.Flatten(),
   ts.keras.layers.Dense(100, activation='relu'),
   ts.keras.layers.Dense(1, activation='sigmoid')                                     
])

modeloCNN_AD.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

x_entrenamiento=x[:19700]
x_validacion=x[19700:]

y_entrenamiento=y[:19700]
y_validacion=y[19700:]

data_gen_entrenamiento = datagen.flow(x_entrenamiento, y_entrenamiento, batch_size=32)

tensorboardCNN_AD = TensorBoard(log_dir='logs/cnn_AD')

modeloCNN_AD.fit(
    data_gen_entrenamiento,
    epochs=100, batch_size=32,
    validation_data=(x_validacion, y_validacion),
    steps_per_epoch=int(np.ceil(len(x_entrenamiento)/float(32))),
    validation_steps=int(np.ceil(len(x_validacion)/float(32))),
    callbacks=[tensorboardCNN_AD]
)


