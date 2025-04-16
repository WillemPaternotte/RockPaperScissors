import cv2 as cv
import numpy as np
import glob

def convert_box(input_file):
    raw_box = np.load(input_file, allow_pickle=True)

    # Make sure it's a dict
    if isinstance(raw_box, np.ndarray):
        print(raw_box)
        raw_box = raw_box.tolist()
    print(raw_box)
    x, y = raw_box["top_left"]
    w = raw_box["width"]
    h = raw_box["height"]

    formatted_box = [x, y, w, h]
    print(formatted_box)
    print("Converted box:", formatted_box)

    # Save if output path provided

    np.save(input_file, formatted_box)
    print(f"Saved to: {input_file}")

list_of_files = glob.glob('./data/ROCK/Photo/*')

for file in list_of_files:
    # convert_box(file)
    frames = np.load(file, allow_pickle=True)


    print(file[22:])
    # print(frames)
    cv.imshow( "photo", frames)
    # np.save("./data/SCISSORS/Photo/photo_"+file[22:], frames)
    if cv.waitKey(0) & 0xFF == ord('q'):
        pass

# cv.imshow("last frame", frame)

cv.waitKey(1000)

cv.destroyAllWindows()