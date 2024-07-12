from ultralytics import YOLO
from PIL import Image
import pyautogui

# Load a pretrained YOLOv8n model
model = YOLO('./weights/yolov8n-pose.pt')
# model = YOLO('yolov8n-pose.pt')

# Define path to video file
source = './data/test.jpg'

# Run inference on the source
results = model(source, stream=True,save=True,show=True)  # generator of Results objects

# print(results.boxes)
# results 是一个生成器，我们需要从它里面迭代获取 Results 对象
for result in results:
    img_path = result.path  # 现在 result 是一个 Results 对象，我们可以访问它的 path 属性
    # print(img_path,end='\n')  # 打印出图像的路径
    print(result.keypoints)


# # Visualize the results
# for i, r in enumerate(results):
#     # Plot results image
#     im_bgr = r.plot()  # BGR-order numpy array
#     im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
#
#     # Show results to screen (in supported environments)
#     r.show()
#
#     # Save results to disk
#     r.save(filename=f"D:/Pycharm/ultralytics-main/ultralytics-main/results{i}.jpg")


