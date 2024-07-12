import cv2
import numpy as np

# 读取图片
image = cv2.imread('test.jpg')

# 坐标点列表，例如：[[x1, y1], [x2, y2], ...]
points = np.array([
         [ 586.5417,  567.9073],[ 518.6461,  547.0000]], np.int32)

# 在图像上绘制坐标点
for point in points:
    cv2.circle(image, (point[0], point[1]), radius=5, color=(0, 255, 0), thickness=-1)

# 显示图像
cv2.imshow('Image with Points', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
        #  [ 518.6461,  547.0000],
        #  [ 606.2086,  613.8230],
        #  [ 442.3024,  567.2548],
        #  [ 587.3171,  809.3562],
        #  [ 330.1561,  765.6517],
        #  [ 490.1160, 1073.1523],
        #  [ 256.8439, 1046.7496],
        #  [   0.0000,    0.0000],
        #  [ 409.1062, 1105.8918]