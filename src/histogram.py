import cv2
import os
from shutil import copyfile
import matplotlib.pyplot as plt

with open("results/lp_discrete_regressor/aligned/labels.txt") as f:
    contents = [x.split() for x in f.readlines()]
    data_dict = {x[0] : float(x[1]) for x in contents}

plt.hist(data_dict.values(), bins="auto")
plt.show()

x = 0
for (file, label) in data_dict.items():
    if label > 0.75:
        copyfile(file, os.path.join("../trainA", os.path.basename(file)))
        x += 1
        if x >= 10000:
            break
