import cv2
import numpy as np
import json
import os
import glob



# Globals for drawing
drawing = False
ix, iy = -1, -1
box = None  # To store just one bounding box
temp_box = None
current_image = None
image_name = ""

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, temp_box, box

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        temp_box = None

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_box = (ix, iy, x - ix, y - iy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        w, h = x - ix, y - iy
        box = [ix, iy, x - ix, y - iy]
        temp_box = None

def save_boxes_to_npy(image_filename, box_data):
    filename = './data/SCISSORS/LABEL/'+(image_filename[21:])[:-4] + "_box"
    np.save(filename, box_data)
    print(f"Saved bounding box to {filename}")

def main():
    global current_image, boxes, image_name

    # image_path = input("").strip()
    # image_name = os.path.basename(image_path)

    # Load .npy image

    list_of_files = glob.glob('./data/SCISSORS/PHOTO/*')
    for file in list_of_files:
        image_name = file
        current_image = np.load(file)

        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", draw_rectangle)
        print(file)
        print("Draw boxes with mouse. Press 's' to save, 'q' to quit.")

        while True:
            display_image = current_image.copy()

            # Show the current temp box if dragging
            if temp_box:
                x, y, w, h = temp_box
                cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Show final box if set
            elif box:
                x, y, w, h = box
                cv2.rectangle(display_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow("Image", display_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                if box:
                    save_boxes_to_npy(image_name, box)
                else:
                    print("No box to save.")
            elif key == ord('q'):
                break


if __name__ == "__main__":
    main()
