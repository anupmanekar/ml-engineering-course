import os
import glob
import shutil
import keras
import tensorflow as tf
import numpy as np
import logging
import matplotlib.pyplot as plot
from keras.preprocessing.image import ImageDataGenerator

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# First load and unzip the data
_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
zip_dir = keras.utils.get_file('flower_photos.tgz', origin=_URL, extract=True)

# Print folder details
zip_dir_base = os.path.dirname(zip_dir)

for root, dirs, files in os.walk(zip_dir_base):
    for dir in dirs:
        print(os.path.join(root, dir))

# Get the count in each dir
base_dir = os.path.join(os.path.dirname(zip_dir), 'flower_photos')
roses_dir = os.path.join(base_dir, 'roses')
sunflowers_dir = os.path.join(base_dir, 'sunflowers')
daisy_dir = os.path.join(base_dir, 'daisy')
dandelion_dir = os.path.join(base_dir, 'dandelion')
tulips_dir = os.path.join(base_dir, 'tulips')

roses_count = len(os.listdir(roses_dir))
sunflowers_count = len(os.listdir(sunflowers_dir))
daisy_count = len(os.listdir(daisy_dir))
dandelion_count = len(os.listdir(dandelion_dir))
tulips_count = len(os.listdir(tulips_dir))

print(f"Rose Images :{roses_count}")
print(f"Sunflower Images :{sunflowers_count}")
print(f"Daisy Images :{daisy_count}")
print(f"Dandelion Images :{dandelion_count}")
print(f"Tulips Images :{tulips_count}")

classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']

""" for cl in classes:
  img_path = os.path.join(base_dir, cl)
  images = glob.glob(img_path + '/*.jpg')
  print("{}: {} Images".format(cl, len(images)))
  train, val = images[:round(len(images)*0.8)], images[round(len(images)*0.8):]

  for t in train:
    if not os.path.exists(os.path.join(base_dir, 'train', cl)):
      os.makedirs(os.path.join(base_dir, 'train', cl))
    shutil.move(t, os.path.join(base_dir, 'train', cl))

  for v in val:
    if not os.path.exists(os.path.join(base_dir, 'val', cl)):
      os.makedirs(os.path.join(base_dir, 'val', cl))
    shutil.move(v, os.path.join(base_dir, 'val', cl))
 """
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
print("After Training and Validation separation---")

for root, dirs, files in os.walk(zip_dir_base):
    for dir in dirs:
        print(os.path.join(root, dir))


# Setup data augmentation
train_image_gen = ImageDataGenerator(
                            rescale=1./255,
                            rotation_range=45,
                            width_shift_range=0.15,
                            height_shift_range=0.15,
                            shear_range=0.2,
                            zoom_range=0.5,
                            horizontal_flip=True,
                            fill_mode='nearest')
val_image_gen = ImageDataGenerator(rescale=1./255)

BATCH_SIZE = 100
IMG_SHAPE = 150
train_data = train_image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                                    directory=train_dir,
                                                    shuffle=True,
                                                    target_size=(IMG_SHAPE,IMG_SHAPE), #(150,150)
                                                    class_mode='binary')
val_data = val_image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                                    directory=val_dir,
                                                    shuffle=True,
                                                    target_size=(IMG_SHAPE,IMG_SHAPE), #(150,150)
                                                    class_mode='binary')

# Define a plot method to visualize the dataset
def plotImages(images_arr):
    fig, axes = plot.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plot.tight_layout()
    plot.show()

visual_images, _ = next(train_data)
# plotImages(visual_images)

# Define Model

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5)
])

# Compile Model

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print(model.summary())

EPOCHS = 10
history = model.fit_generator(
    train_data,
    steps_per_epoch=int(np.ceil(train_data.n / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=val_data,
    validation_steps=int(np.ceil(val_data.n / float(BATCH_SIZE)))
)

# Visualize the results

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plot.figure(figsize=(8, 8))
plot.subplot(1, 2, 1)
plot.plot(epochs_range, acc, label='Training Accuracy')
plot.plot(epochs_range, val_acc, label='Validation Accuracy')
plot.legend(loc='lower right')
plot.title('Training and Validation Accuracy')

plot.subplot(1, 2, 2)
plot.plot(epochs_range, loss, label='Training Loss')
plot.plot(epochs_range, val_loss, label='Validation Loss')
plot.legend(loc='upper right')
plot.title('Training and Validation Loss')
plot.show()
