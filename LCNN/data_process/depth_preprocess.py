import numpy as np
import cv2
from os import path as osp
from glob import glob
from PIL import Image


class Loader():
    def __init__(self, folder, backend='.jpg'):
        self.folder = folder
        self.backend = backend
        self.filenames = self.load_filenames()

    def load_filenames(self):
        files = glob(osp.join(self.folder, '*{}'.format(self.backend)))
        filenames = [n.split('/')[-1][:-len(self.backend)] for n in files]
        return filenames

    def load_image(self, name):
        img = cv2.imread(osp.join(self.folder, name + self.backend), cv2.IMREAD_UNCHANGED)
        return img

    def show_img(self, img):
        im = Image.fromarray(img).convert('L')
        im.show()

    def save_img(self, img, filepath):
        cv2.imwrite(filepath + self.backend, img)

    def get_filenames(self):
        return self.filenames


def fill_hole(depth_im):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    depth_im_de = cv2.morphologyEx(depth_im, cv2.MORPH_OPEN, kernel)
    mask = np.less(depth_im, 1)
    depth_im_de = np.where(mask, depth_im, depth_im_de)
    return depth_im_de


if __name__ == '__main__':
    print("preprocess depth map now")
    folder_name = "/Users/liuhan/Documents/2019毕业设计/RGB2Depth_exp"
    depth = Loader(folder_name, '.jpg')
    for depth_name in depth.get_filenames():
        depth_map = depth.load_image(depth_name)
        filled_depth = fill_hole(depth_map)
        depth.save_img(filled_depth, osp.join(folder_name, "filled_depth", depth_name))
    # img = cv2.imread("/Volumes/Untitled/depth/NU/010/010_Kinect_NU_1DEPTH_25.png")
    # depth_map = fill_hole(img)
    # im = Image.fromarray(depth_map).convert('L')
    # im.show()
