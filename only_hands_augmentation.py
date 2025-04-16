import numpy as np
import cv2 as cv
import random
from data_augmentation import add_noise, add_random_brightness, apply_kernel
from scipy import ndimage
def augment_image(image, label):
    images = []
    labels = []

    input_image = image.copy()
    images.append(input_image)

    noise_image = image.copy()
    noise_image = ndimage.rotate(noise_image, random.randint(-45,45), reshape=False, mode='nearest')
    images.append(add_noise(noise_image))

    brightness_image = image.copy()
    brightness_image = ndimage.rotate(brightness_image, random.randint(-45, 45), reshape=False, mode='nearest')
    images.append(add_random_brightness(brightness_image))

    blur_image = image.copy()
    blur_image = ndimage.rotate(blur_image, random.randint(-45, 45), reshape=False, mode='nearest')
    images.append(cv.blur(blur_image, (4,4)))

    # kernel_image = image.copy()
    # images.append(apply_kernel(kernel_image))

    for i in range(len(images)):
        labels.append(label)

    return images, labels





if __name__ == "__main__":


    loaded_images = np.load('./data/OnlyHands/model_photos.npy')
    loaded_labels = np.load('./data/OnlyHands/model_labels.npy')
    print(loaded_labels.shape)
    mirrored_images = []
    mirrored_labels = []
    # for i in range(len(loaded_images)):
    #     mirrored_images.append(loaded_images[i])
    #     mirrored_labels.append(loaded_labels[i])
    #     mirrored_images.append(cv.flip(loaded_images[i],1))
    #     mirrored_labels.append(loaded_labels[i])

    augmented_images = []
    augmented_labels = []
    for i in range(len(loaded_images)):
        temp_augmented_images, temp_augmented_labels = augment_image(loaded_images[i], loaded_labels[i])
        # print(np.array(temp_augmented_images).shape)
        for image in temp_augmented_images:
            # print(image[0])
            # cv.imshow("check", cv.resize(image, (320, 320)))
            # if cv.waitKey(0):
            #     pass
            augmented_images.append(image)
        for label in temp_augmented_labels:
            augmented_labels.append(label)

    print(np.array(augmented_images).shape)
    print(np.array(augmented_labels).shape)
    np.save("data/OnlyHands/model_photos_augmented_no_kernel", augmented_images)
    np.save("data/OnlyHands/model_labels_augmented_no_kernel", augmented_labels)
    print(np.array(augmented_images).shape)
    print("pictures saved")
    ####UNCOMMENT TO LOOP TRHOUGH ALL IMAGES
    # for i in range(len(augmented_images)):
    #     cv.imshow(str(augmented_labels[i]), cv.resize(augmented_images[i], (320,320)))
    #     if cv.waitKey(0):
    #         pass
