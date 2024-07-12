from ultralytics import YOLO

# 加载模型
# model = YOLO("yolov8n.yaml")  # 从头开始构建新模型
model = YOLO("../../../Program Files (x86)/Pycharm/AimYolov8/test/yolov8n-pose.pt")  # 加载预训练模型（推荐用于训练）

# Use the model
results = model.train(data="csgo.yaml", epochs=20, batch=4)  # 训练模型
