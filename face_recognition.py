from tensorflow.keras.preprocessing.image import ImageDataGenerator #loading the image and data preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import os

import tensorflow as tf
print("GPU available:", tf.config.list_physical_devices('GPU'))

train_data_dir = r'D:\machine learning\emotions\train'
validation_data_dir = r'D:\machine learning\emotions\test'

print("Checking paths...")
print(f"Train path exists: {os.path.exists(train_data_dir)}")
print(f"Validation path exists: {os.path.exists(validation_data_dir)}")

# Kalau ada, liat isinya
if os.path.exists(train_data_dir):
    print(f"\nFolders in train: {os.listdir(train_data_dir)}")

train_datagen = ImageDataGenerator(
                  rescale = 1./255,
                  rotation_range = 30,
                  shear_range = 0.3,
                  zoom_range = 0.3,
                  horizontal_flip = True,
                  fill_mode = 'nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                  train_data_dir,
                  color_mode = 'grayscale',
                  target_size = (48,48),
                  batch_size = 32,
                  class_mode = 'categorical',
                  shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
                       validation_data_dir,
                       color_mode = 'grayscale',
                       target_size = (48,48),
                       batch_size = 32,
                       class_mode = 'categorical',
                       shuffle = True
)

class_labels=['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', "Surprise"]

img, label = train_generator.__next__()

model = Sequential()

# 48 cross - 48 cross - 1 grayscale
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Dropout(0.1))

# 64 neuron layers
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

# 128 neuron layers
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

# 256 neuron layers
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

#512 neuron layers
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(7, activation = 'softmax'))

model.compile(
  optimizer='adam', 
  loss='categorical_crossentropy', 
  metrics=['accuracy'])
print(model.summary())


train_path = train_data_dir
test_path = validation_data_dir

num_train_imgs = 0
for root, dirs, files in os.walk(train_path):
  num_train_imgs += len(files)

num_test_imgs = 0
for root, dirs, files in os.walk(test_path):
  num_test_imgs += len(files)

print(num_train_imgs)
print(num_test_imgs)
epochs = 28    

history = model.fit(
        train_generator,
        steps_per_epoch=num_train_imgs//32,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=num_test_imgs//32,
)

model.save('model_file_epoch30.h5 ')

"""# New Section"""