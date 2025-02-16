import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import os
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array, img
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None, None
def predict_dressing_score(img_path):
    img_array, img = preprocess_image(img_path)
    if img_array is not None:
        prediction = model.predict(img_array)
        dressing_score = int(prediction[0][0] * 100)
        return dressing_score, img
    else:
        return None, None
train_image_paths = ['woman1.jpg', 'man.jpg'] 
train_labels = [0, 1]  

val_image_paths = ['woman2.jpg', 'man1.jpg']  
val_labels = [1, 0] 

train_images = []
train_labels_cleaned = []
for img_path, label in zip(train_image_paths, train_labels):
    img_array, _ = preprocess_image(img_path)
    if img_array is not None:
        train_images.append(img_array)
        train_labels_cleaned.append(label)

train_images = np.vstack(train_images)
train_labels = np.array(train_labels_cleaned)
val_images = []
val_labels_cleaned = []
for img_path, label in zip(val_image_paths, val_labels):
    img_array, _ = preprocess_image(img_path)
    if img_array is not None:
        val_images.append(img_array)
        val_labels_cleaned.append(label)

val_images = np.vstack(val_images)
val_labels = np.array(val_labels_cleaned)

if len(train_images) > 0 and len(val_images) > 0:
    model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(val_images, val_labels))
else:
    print("No valid images found for training or validation.")
img_path = 'man2.jpg' 
score, img = predict_dressing_score(img_path)

if img is not None:
    plt.imshow(img)
    plt.title(f"Dressing Score: {score}")
    plt.axis('off')
    plt.show()
else:
    print(f"Error: Unable to load or process the test image {img_path}.")
