import LCNN9 as LCNN9
import numpy as np
import cv2
import scipy.io as sio
import random

res = []
root_folder = '/home/wtx/RGBD_dataset/eaststation/'
# labs = []
f = open('lfw_list_part.txt', 'r')
labs = np.empty([2080, 1], dtype=object)
count = 0
for line in f.readlines():
    line = line.strip()
    labs[count] = line[-1]
    imgs = []
    img = cv2.imread(root_folder+line[:-2], 0)
    # img = cv2.resize(img, (122, 144))
    # M2 = np.float32([[1, 0, 11], [0, 1, 0]])
    # img = cv2.warpAffine(img, M2, (144, 144))

    # for i in range(1):
    #     w = 8
    #     h = 8
    #     img2 = img[w:w + 128, h:h + 128] / 255.
    #     img2 = np.float32(img2)
    imgs.append(np.float(img/255))

    imgs = np.array(imgs)
    feas = LCNN9.eval(imgs)
    res.append(feas)
    count += 1
    # if count == 10:
    # 	break
    if count % 10 == 0:
        print(count)
res = np.array(res)
res = np.reshape(res, [2080, 512])
print(res.shape)
print(labs.shape)
sio.savemat('eaststation_feas.mat', {'data': res, 'label': labs})
f.close()
