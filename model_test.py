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



##HELPER FUNCTIONS
def tlwh_to_xyxy(box):
    x, y, w, h = tf.split(box, 4, axis=-1)
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return tf.concat([x1, y1, x2, y2], axis=-1)

def iou_metric(y_true, y_pred):
    y_true = tlwh_to_xyxy(y_true)
    y_pred = tlwh_to_xyxy(y_pred)

    xA = tf.maximum(y_true[..., 0], y_pred[..., 0])
    yA = tf.maximum(y_true[..., 1], y_pred[..., 1])
    xB = tf.minimum(y_true[..., 2], y_pred[..., 2])
    yB = tf.minimum(y_true[..., 3], y_pred[..., 3])

    interArea = tf.maximum(0.0, xB - xA) * tf.maximum(0.0, yB - yA)
    boxAArea = (y_true[..., 2] - y_true[..., 0]) * (y_true[..., 3] - y_true[..., 1])
    boxBArea = (y_pred[..., 2] - y_pred[..., 0]) * (y_pred[..., 3] - y_pred[..., 1])
    unionArea = boxAArea + boxBArea - interArea

    iou = interArea / (unionArea + 1e-7)
    return tf.reduce_mean(iou)

def iou_loss(y_true, y_pred):
    return 1 - iou_metric(y_true, y_pred)
def mirror_data(images,labels,axis, width= 320, height = 180):
    augmented_images = []
    augmented_labels = []
    for image in images:
        augmented_images.append(cv.flip(image, axis))
    for label in labels:
        if axis:
            new_x = width - (label[0]+label[2])
            augmented_labels.append([new_x, label[1], label[2], label[3]])
        else:
            new_y = height - (label[1]+label[3])
            augmented_labels.append([label[0], new_y, label[2], label[3]])
    return augmented_images, augmented_labels
def load_data(folderpath):
    list_of_files = glob.glob(folderpath)
    X=[]
    y=[]
    for file in list_of_files:

        labelfile = folderpath[:-7]+"LABEL/"+(file.split("/")[-1])[6:-4]+"_box.npy"

        X.append(np.load(file))
        y.append(np.load(labelfile, allow_pickle=True))

    return X, y

def load_empty_data():
    list_of_files = glob.glob('./data/EMPTY/Photo/*')
    images= []
    labels = []
    for file in list_of_files:
        images.append(np.load(file))
        images.append(np.flip(np.load(file), 1))
        labels.append([0,0,320,180])
        labels.append([0, 0, 320, 180])

    return images, labels


def normalize_labels(labels, width = 320, height = 180):
    normalized_labels = []
    for i in range(len(labels)):
        # print(labels[i][0]/width)
        normalized_labels.append([float(labels[i][0])/float(width),float(labels[i][1]/height),float(labels[i][2]/width),float(labels[i][3]/height)])
    return normalized_labels

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
    X0, y0 = load_data('./data/ROCK/PHOTO/*')
    X1, y1 = load_data('./data/PAPER/PHOTO/*')
    X2, y2 = load_data('./data/SCISSORS/PHOTO/*')

    X = X0 + X1 + X2
    y = y0 + y1 + y2

    mirX, miry = mirror_data(X, y, 1)
    X = np.array(X + mirX)
    y = np.array(y + miry)
    print(X.shape)
    print(y.shape)



    # X = X[..., tf.newaxis]
    X = np.array(X)
    y = np.array(y)

    X_temp_train, X_test, y_temp_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
    # X_train, y_train = augment_list_of_images(X_temp_train, y_temp_train)
    # np.save("data/AUGMENTED/photos", np.array(X_train))
    # np.save("data/AUGMENTED/labels", np.array(y_train))
    X_aug = np.load('data/AUGMENTED/photos.npy')
    y_aug = np.load('data/AUGMENTED/labels.npy')
    X_empty, y_empty = load_empty_data()
    X_train = np.concat((X_aug, X_empty))
    y_train = np.concat((y_aug, y_empty))
    X_train, no, y_train, nono = train_test_split(X_train, y_train, test_size=0.001, random_state=42)#using train test to reshuffle images
    print(X_train.shape)
    print(y_train.shape)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    ##NORMALIZE DATA
    X_train = X_train/255
    X_test = X_test/255
    print(X[0])
    y_train = normalize_labels(y_train)
    y_test = normalize_labels(y_test)
    print(y_train)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    in_shape = (X_train[1].shape[0], X_train[1].shape[1], X_train[1].shape[2])
    print(in_shape)

    ##CREATE MODEL
    model = tf.keras.models.Sequential([
                # use some bigger filters for first layers, then narrow down
                tf.keras.layers.Conv2D(120, (7, 7), activation='relu', input_shape=in_shape),
                tf.keras.layers.MaxPooling2D((5, 5)),
                tf.keras.layers.Conv2D(60, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(60, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),

                # flatten before the fully connected layers
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(60, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(30, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10, activation='relu'),

                tf.keras.layers.Dense(4, activation='sigmoid')
            ])


    model.compile(optimizer='adam', loss=iou_loss, metrics=['mae', iou_metric])

    model.summary()

    # hyperparameters
    epochs = 100
    batch_size = 32
    validation_split = 0.2

    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=10,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[callback])
    model.save('model_saves/my_model_alldata_new2.keras')
    plot_history(history)

    boxes = model.predict(X_test)
    for i in range(len(X_test)):
        # Scale the image back to 0–255 and convert to uint8
        image = (X_test[i] * 255).astype(np.uint8)
        test_box = [int(y_test[i][0]*320),int(y_test[i][1]*180),int(y_test[i][2]*320),int(y_test[i][3]*180)]

        # Convert coordinates to integers
        x = int(boxes[i][0] * 320)
        y = int(boxes[i][1] * 180)
        w = int(boxes[i][2] * 320)
        h = int(boxes[i][3] * 180)

        # Print bounding box coords
        print(f"Top-left: ({x}, {y}), Bottom-right: ({x + w}, {y + h})")

        # Draw rectangle
        cv.rectangle(image, (test_box[0], test_box[1]), (test_box[0] + test_box[2], test_box[1] + test_box[3]), (0, 255, 0), 2) #green rectangle, actual box
        cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # blue rectangle, predicted box
        # Show image
        cv.imshow("Test Image", image)

        # Wait for key press (press 'q' to quit showing images)
        if cv.waitKey(0) & 0xFF == ord('q'):
            pass

    cv.destroyAllWindows()


