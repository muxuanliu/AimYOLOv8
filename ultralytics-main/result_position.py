import cv2
from ultralytics import YOLO
from PIL import Image
import pyautogui



# Load a pretrained YOLOv8n model
model = YOLO('./yolov8n-pose.pt')

# # Define path to video file
source = 'test3.jpg'

# # Run inference on the source
def inference_img(img, model):
    """
    :param img: 预测图片
    :param augment:
    :param model: 使用模型
    :param conf_thres: 置信度
    :param iou_thres:  重叠度
    :param classes: 类别
    :param agnostic:
    :return:    detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    results = model(img, stream=True, save=True, show=True)  # generator of Results objects

    # results 是一个生成器，我们需要从它里面迭代获取 Results 对象
    for result in results:
        print(result,end='\n')
        print(type(result))
        results_xy = result.keypoints.xy

        result_xy = results_xy[0]
        print('results_xy',results_xy)

        # 将结果转换为列表
        result_xy = result_xy.tolist()
    return result_xy


# 读取图片
image = cv2.imread('test2.jpg')


# 部位索引到部位名称的映射字典
body_parts_map = {
    0: "鼻子",
    1: "左眼",
    2: "右眼",
    3: "左耳",
    4: "右耳",
    5: "左肩膀",
    6: "右肩膀",
    7: "左肘部",
    8: "右肘部",
    9: "左手腕",
    10: "右手腕",
    11: "左胯",
    12: "右胯",
    13: "左膝盖",
    14: "右膝盖",
    15: "左脚",
    16: "右脚"
}
#
# # 将points转换为int32类型，因为cv2.polylines要求点的类型为int32
# inference_results = inference_results.astype(np.int32)
result_xy = inference_img(source,model)



def coords_and_des(result_xy):
    if len(result_xy) == 0:
        return -1,-1
    # 遍历推断结果
    for index, coords in enumerate(result_xy):
        # 将坐标转换为整数
        x, y = coords[0],coords[1]

        # 检查坐标是否不为 [0.0, 0.0] 并且索引小于5
        if x != 0 and y != 0 and index < 5:
            # 获取部位名称
            part_name = body_parts_map.get(index, "未知部位")
            # 输出坐标点对应的含义
            print(f"索引 {index} 对应 {part_name}, 坐标: {coords}")
            return x,y


x,y = coords_and_des(result_xy)

if x==-1 and y==-1:
    print('不行')
print(x,y)



'''
0：鼻子
1：眼睛
2：眼睛
3：耳朵
4：耳朵
5：肩膀
6：肩膀
7:肘部
8：肘部
9：手腕
10：手腕
11：胯
12：胯
13：膝盖
14：膝盖
15：脚
16：脚
'''

# # 在图像上绘制坐标点
# for point in inference_results:
#     cv2.circle(image, (point[0], point[1]), radius=5, color=(0, 255, 0), thickness=-1)


        #  [ 518.6461,  547.0000],
        #  [ 606.2086,  613.8230],
        #  [ 442.3024,  567.2548],
        #  [ 587.3171,  809.3562],
        #  [ 330.1561,  765.6517],
        #  [ 490.1160, 1073.1523],
        #  [ 256.8439, 1046.7496],
        #  [   0.0000,    0.0000],
        #  [ 409.1062, 1105.8918]