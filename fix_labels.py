import glob
import numpy as np
list_of_files = glob.glob('./data/PAPER/LABEL/*')
for file in list_of_files:
    x,y,w,h = np.load(file)
    if x < 0:
        x=0
        print("changed x")
        print(file)
    if  y < 0:
        y=0
        print("changed y")
        print(file)
    if x+w > 320:
        w = 320 -x
        print("changed width")
        print(file)
    if y+h > 180:
        h = 180 -y
        print("changed height")
        print(file)
    np.save(file, [x,y,w,h])