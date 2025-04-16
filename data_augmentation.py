import cv2 as cv
import numpy as np
import random
import glob
from sklearn.model_selection import train_test_split
#


def load_data(folderpath):
    list_of_files = glob.glob(folderpath)
    X=[]
    y=[]
    for file in list_of_files:

        labelfile = folderpath[:-7]+"LABEL/"+(file.split("/")[-1])[6:-4]+"_box.npy"

        X.append(np.load(file))
        y.append(np.load(labelfile, allow_pickle=True))

    return X, y
def augment_list_of_images(images, labels):
    new_images = []
    new_labels = []
    for i in range(len(images)):
        image = images[i].copy()
        new_images.append(image)
        new_labels.append(labels[i])

        crop_bright = image.copy()
        crop_bright, crop_bright_box = random_crop(crop_bright, labels[i])
        if type(crop_bright) != int:
            crop_bright = add_random_brightness(crop_bright)
            new_images.append(crop_bright)
            new_labels.append(crop_bright_box)

        crop_noise = image.copy()
        crop_noise, crop_noise_box = random_crop(crop_noise, labels[i])
        if type(crop_noise) != int:
            crop_noise = add_noise(crop_noise)
            new_images.append(crop_noise)
            new_labels.append(crop_noise_box)



    return new_images, new_labels

def add_random_brightness(image):
    factor = random.uniform(0.5, 1.5)
    new_image = np.clip(image.astype(np.float32) * factor, 0, 255)
    return new_image.astype(np.uint8)

def add_noise(image):
    (B, G, R) = cv.split(image)
    B = salt_pepper(B)
    G = salt_pepper(G)
    R = salt_pepper(R)
    return cv.merge((B, G, R))

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

def salt_pepper(image):
    row, col = image.shape
    pixel_count = row*col
    number_of_pixels = random.randint(int(pixel_count/2), int(pixel_count))
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)
        # print("X_coord", x_coord)
        # Color that pixel to white
        value = image[y_coord][x_coord]
        # print("Value_og: ", value)
        value = value*1.1

        # print("Value: ", value)
        # print("X_coord", x_coord)
        if value > 255:
            image[y_coord][x_coord] = 255
        else:
            image[y_coord][x_coord] = value

    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(int(pixel_count/4), int(pixel_count/2))
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to black
        image[y_coord][x_coord] = (image[y_coord][x_coord])/1.1

    return image

def random_crop(image, box, width=320,height=180):
    x,y,w,h = box
    aspect_ratio=height/width
    # print(image.shape)
    bad_crop = True
    tries = 0
    while bad_crop:

        tries +=1
        if tries > 500:
            print(tries)
            return -1, -1

        topleft_x = random.randint(0, x)
        topleft_y = random.randint(0, y)
        bottomright_x = random.randint(x+w, 320)
        bottomright_y = int(topleft_y + (bottomright_x - topleft_x)*aspect_ratio)

        if bottomright_y<180 and bottomright_y> y+h:
            new_image = image.copy()[topleft_y:bottomright_y, topleft_x:bottomright_x]
            new_box = [x-topleft_x, y-topleft_y, w,h]
            bad_crop = False

    print(tries)
    scale = height/new_image.shape[0]
    new_image = cv.resize(new_image, (320,180))
    new_new_box = [int(new_box[0]*scale),int(new_box[1]*scale), int(new_box[2]*scale), int(new_box[3]*scale)]
    return new_image, new_new_box

def apply_kernel(input_image):
    kernel = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ])
    return cv.filter2D(input_image, -1, kernel)

# new_frame, new_box = random_crop(new_frame, box)
# new_frame = add_random_brightness(new_frame)
# crop_frame, crop_box = random_crop(frame, box)
# crop_frame = add_noise(crop_frame)
# x, y, w, h = crop_box
# cv.rectangle(crop_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
# kernel_frame = apply_kernel(frame.copy())
#
# print(new_frame.shape)
#
#
# cv.imshow("last frame", frame)
#
# cv.imshow("brightness frame", new_frame)
# cv.imshow("Crop frame", crop_frame)
# cv.imshow("kernel frame", kernel_frame)
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

    X_temp_train, X_test, y_temp_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
    X_train, y_train = augment_list_of_images(X_temp_train, y_temp_train)
    np.save("data/AUGMENTED/photos_nokernel", np.array(X_train))
    np.save("data/AUGMENTED/labels_nokernel", np.array(y_train))
    new_frames, new_boxes = X_train, y_train
    # new_frames, new_boxes = augment_list_of_images(frames, boxes)
    for i in range(len(new_frames)):
        frame = new_frames[i]
        box = new_boxes[i]
        x, y, w, h = box
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv.imshow("last frame", frame)
        if cv.waitKey(0):
            pass





    cv.destroyAllWindows()