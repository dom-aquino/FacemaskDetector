import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array

testing_img = []
testing_labels = []

for img in os.listdir("./dataset/for_testing"):
    if img.endswith(".jpeg") or img.endswith(".jpg") or img.endswith(".png"):
        try:
            image_in_pil = load_img("./dataset/for_testing/{}".format(img))
            testing_img.append(img_to_array(image_in_pil))

            if "with_mask" in img:
                testing_labels.append(1)
            elif "without_mask" in img:
                testing_labels.append(0)
        except:
            print("error in: {}".format(img))

x_test = np.array(testing_img)
y_test = np.array(testing_labels)

training_img = []
training_labels = []

# Build training images and labels
for img in os.listdir("./dataset/for_training"):
    if img.endswith(".jpeg") or img.endswith(".jpg") or img.endswith(".png"):
        try:
            image_in_pil = load_img("./dataset/for_training/{}".format(img))
            training_img.append(img_to_array(image_in_pil))

            if "with_mask" in img:
                training_labels.append(1)
            elif "without_mask" in img:
                training_labels.append(0)
        except:
            print("error in: {}".format(img))

# Convert training images and labels to Numpy array
x_train = np.array(training_img)
y_train = np.array(training_labels)

# Build the neural network model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Rescaling(1.0 / 255))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model -- back propagation to determine the weights and biases
model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))
model.summary()

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])

# Save the model for usage
model.save('mask_recog_v1.h5')

