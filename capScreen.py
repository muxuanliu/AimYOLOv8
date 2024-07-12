import torch
from mss import mss
import numpy as np
import cv2
from utils.datasets import letterbox
sct_img = mss()
bound_box = {'left': 0, 'top': 0, 'width': 1920, 'height': 1080}
def capScreen(mss_object, bound_box):
    '''
    :param mss_object:  mss object
    :param bound_box:   dict,截图区域,eg.{'left':0,'top':0,'width':1920,'height':1080}
    :return: numpy_array
    '''
    sct_img = mss_object.grab(bound_box)
    # mss 抓取的图片通常为4通道：BGRA A是透明度
    sct_img = np.array(sct_img)

    # cv是基于数组的操作，需要numpy将图片转为数组
    sct_img = cv2.cvtColor(sct_img, cv2.COLOR_BGRA2RGB)
    return sct_img





if __name__ == '__main__':
    capScreen(sct_img, bound_box)
