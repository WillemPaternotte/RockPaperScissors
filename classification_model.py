import time
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from enum import Enum
import glob
import os
from data_augmentation import augment_list_of_images
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class Hand(Enum):
    ROCK = 0
    PAPER = 1
    SCISSORS = 2



def plot_history(history):
    # Plot Training and Validation Loss
    plt.figure(figsize=(5, 5))
    plt.plot(history.history['loss'], label='Train Loss', color='orange')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

if __name__ == "__main__":
    X1 = np.load('./data/OnlyHands/photos_augmented.npy')
    y1 = np.load('./data/OnlyHands/labels_augmented.npy')
    X2 = np.load('./data/OnlyHands/model_photos_augmented_no_kernel.npy')
    y2 = np.load('./data/OnlyHands/model_labels_augmented_no_kernel.npy')

    X = X1.tolist() + X2.tolist()
    y = y1.tolist() + y2.tolist()
    print(y1.shape, y2.shape)

    X = np.array(X)
    print(X.shape)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

    print(X_train.shape)
    print(y_train.shape)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    ##NORMALIZE DATA
    X_train = X_train/255
    X_test = X_test/255
    print(X_train[0])

    in_shape = (X_train[1].shape[0], X_train[1].shape[1], X_train[1].shape[2])
    print(in_shape)

    ##CREATE MODEL
    model = tf.keras.models.Sequential([
                #convolution layers
                tf.keras.layers.Conv2D(40, (7, 7), activation='relu', input_shape=in_shape),
                tf.keras.layers.MaxPooling2D((5, 5)),
                tf.keras.layers.Conv2D(30, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),

                # flatten before the fully connected layers
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(12, activation='relu'),

                tf.keras.layers.Dense(3, activation='softmax')
            ])

    model.compile(optimizer="RMSprop", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    # hyperparameters
    epochs = 100
    batch_size = 150
    validation_split = 0.2

    y_train = np.argmax(y_train, axis=1)

    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=10,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks = [callback])
    model.save('classification_model_saves/my_model_nokernel.keras')
    plot_history(history)
    predictions = model.predict(X_test)

    # test_loss, test_accuracy = model.evaluate(X_test, y_test)
    # print("loss of evaluation:", test_loss)
    # print("test_accuray:", test_accuracy)
    y_test_numerical = []
    for y in y_test:
        y_test_numerical.append(np.argmax(y))

    best_predictions = []

    for i in range(len(X_test)):
        # Scale the image back to 0–255 and convert to uint8
        prediction = predictions[i]
        image = (X_test[i] * 255).astype(np.uint8)

        best_prediction = np.argmax(prediction)
        best_predictions.append(best_prediction)


        # cv.imshow(str(Hand(best_prediction).name), cv.resize(image, (320,320)))
        #
        # # Wait for key press (press 'q' to quit showing images)
        # if cv.waitKey(0) & 0xFF == ord('q'):
        #     pass

    print("where is the confusion matrix?")
    cm = confusion_matrix(y_test_numerical, best_predictions)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Rock', 'Paper', 'Scissors'])
    disp.plot()

    plt.show()
    print("where is the confusion matrix?")
    # cv.destroyAllWindows()


