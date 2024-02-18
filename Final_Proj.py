import os
import cv2
import numpy as np
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split

# Set the path to your dataset
DATASET_PATH = 'C:/Users/prajs_28/OneDrive/Desktop/ADV Malware/Project/dataset'

# Set the image size
IMG_SIZE = (224, 224)

# Load the images and labels
data = []
labels = []
for class_name in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, class_name)
    for filename in os.listdir(class_path): #preprocess the images loaded
        img = cv2.imread(os.path.join(class_path, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        data.append(img)
        labels.append(class_name)

random.shuffle(data)
#print(data)
#exit()

# Convert the data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)


# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.3, random_state=42)

print('Number of images in the training set:', len(train_data))
print('Number of images in the testing set:', len(test_data))

# Create the deep learning model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.5), # Adding a dropout layer to reduce overfitting
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5), # Adding another dropout layer to reduce overfitting
    tf.keras.layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Convert the labels to encoded vectors
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# Train the model with early stopping and model checkpoint
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3) # Adding early stopping to prevent overfitting
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True) # Adding model checkpoint to save the best model
model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stop, model_checkpoint])

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_data, test_labels)
print('Test accuracy:', test_acc)

#Classify the random input provided to the model

input_path = 'C:/Users/prajs_28/OneDrive/Desktop/ADV Malware/Project/input4.jpg' #provide path to the input image **Recheck the file extensions**

# Set the image size
IMG_SIZE = (224, 224)

# Load and preprocess the input image
img = cv2.imread(input_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, IMG_SIZE)
img = np.expand_dims(img, axis=0)

# Use the trained model to make a prediction
prediction = model.predict(img)
print("Prediction Score of not a Ransom Note :",prediction[0][0])
print("Prediection Score of a Ransom Note:",prediction[0][1])

# Print the prediction
if prediction[0][1] < 0.5: #if the probabiltiy score of class 2(ransom note) is less than 0.5
    print('The input is NOT a ransom note.')
else:
    print('The input is a ransom note.')











