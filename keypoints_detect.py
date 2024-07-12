import mss
import win32api
import win32con
from pynput import mouse
from utils.utils import *
from capScreen import *
from mouse_sendin import *
import argparse
# from models.experimental import attempt_load
import sys
import signal
from ultralytics import YOLO

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

def pre_process(img0, img_sz, half, device):
    '''
    :param img0: format:bgr,hwc
    :param img_sz:
    :param half:    bool
    :param device:
    :return:    img
    '''
    # resize
    img = letterbox(img0, new_shape=img_sz)[0]
    # convert
    img = img[:, :, ::-1].transpose(2, 0, 1)
    # img[:,:,::-1]:hwc的c反转：RGB->BGR
    # transpose(2,0,1) hwc->chw
    # 参数 (2, 0, 1) 指定了维度的顺序变换规则，表示将原始图像的第三个维度（通道维度）移动到第一个位置，将原始图像的第一个维度（高度维度）移动到第二个位置，将原始图像的第二个维度（宽度维度）移动到第三个位置。
    img = np.ascontiguousarray(img)
    # 确保img数组有一个连续的内存空间
    # preprocess
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0
    # 0.0-255.0 to 0.0-1.0
    if img.ndimension() == 3:
        # 如果维度是3
        img = img.unsqueeze(0)
        # 在第0维添加大小为1的维度
    return img


# # Run inference on the source
def inference_img(img, model, augment, conf_thres, iou_thres, classes, agnostic):
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
    # 假设 img 是预处理后的输入图像张量
    with torch.no_grad():
        results = model(img)  # generator of Results objects

    # results 是一个生成器，我们需要从它里面迭代获取 Results 对象
    for result in results:
        results_xy = result.keypoints.xy

        result_xy = results_xy[0]
        # 将结果转换为列表
        result_xy = result_xy.tolist()
    return result_xy

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


def view_imgs(img0):
    img0 = cv2.resize(img0, (480, 270))
    img0 = cv2.cvtColor(img0,cv2.COLOR_BGR2RGB)
    cv2.imshow('ws demo', img0)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        exit()


def move_mouse(mouse_pynput, aim_persons_center):
    """
    :param mouse_pynput: 鼠标控制器
    :param aim_persons_center: 目标点
    :return: 移动鼠标
    """
    if aim_persons_center:
        # 鼠标位置不为空
        current_x, current_y = mouse_pynput.position
        # 鼠标当前位置
        best_position = None
        # 候选位置
        for aim_person in aim_persons_center:
            # aim_person 是一个列表
            # 计算两点间的距离
            dist = ((aim_person[0] - current_x) ** 2 + (aim_person[1] - current_y) ** 2) ** .5
            if not best_position:
                best_position = (aim_person, dist)
            else:
                _, old_dist = best_position
                # best_position第一个元素是目标人物，第二个是坐标点的距离
                if dist < old_dist:
                    best_position = (aim_person, dist)
        tx = int(best_position[0][0] / win32api.GetSystemMetrics(0) * 65535)
        # 计算目标点的绝对位置，乘以65535是因为Windows中鼠标坐标是以0-65535的范围来表示的
        ty = int(best_position[0][1] / win32api.GetSystemMetrics(1) * 65535)
        # print(tx,ty)
        SendInput(mouse_input(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE, tx, ty))


class AimYolov8:
    def __init__(self, opt):
        # 对象初始化
        self.weights = opt.weights
        self.img_size = opt.img_size
        self.conf_thres = opt.conf_thres
        self.iou_thres = opt.iou_thres
        self.view_img = opt.view_img
        self.classes = opt.classes
        self.agnostic_nms = opt.agnostic_nms
        self.augment = opt.augment
        self.bounding_box = {'left': 0, 'top': 0, 'width': 1920, 'height': 1080}
        # load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 使用YOLO类加载模型，确保指定了正确的设备
        self.model = YOLO(self.weights).to(self.device)
        # 加载模型到对应的设备上
        self.img_size = check_img_size(self.img_size)
        # 检查并返回合适模型最大步长的尺寸
        self.half = self.device.type != 'cpu'
        # 半精度只支持cuda
        if self.device.type == 'cpu' or not self.half:
            # 将模型转换为半精度
            self.model.to(dtype=torch.float16)
        else:
            self.model.to(dtype=torch.float32)
        print(self.half,end='\n')
        print(self.device.type)
        # 如果在gpu则转换model为半精度FP16
        self.names = self.model.modules.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

    @torch.no_grad()
    def run(self):
        img_sz = self.img_size

        img = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)

        # (1,3,self.img_size,self.img_size)
        # 分别指批处理量，通道数，高和宽
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None
        # 如果设备不为cpu则将img转为半精度，在gpu上运行速度更快
        sct = mss()
        print('成功创建MSS实例')
        mouse_control = mouse.Controller()
        print('成功创建鼠标控制器')
        while True:
            aim_person_center_head = []
            img0 = capScreen(sct, self.bounding_box)
            # 此时得到的格式为HWC和BGR

            img = pre_process(img0=img0, img_sz=img_sz, half=self.half, device=self.device)

            t1 = torch_utils.time_synchronized()
            # 记录当前时间
            pred = inference_img(img=img, model=self.model, augment=self.augment, conf_thres=self.conf_thres,
                                 iou_thres=self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
            # pred：(x1, y1, x2, y2, conf, cls)
            t2 = torch_utils.time_synchronized()
            x,y = coords_and_des(pred)
            s = ''
            s += '%gx%g ' % img.shape[2:]
            # img的形状为[batch_size, channels, height, width]，img.shape[2:]表示从第三个元素开始到最后一个
            # print(det)
            if x != -1 and y != -1:
                aim_person_center_head.append([x, y])
                move_mouse(mouse_control, aim_person_center_head)
            print(f'{s} Done. ({t2 - t1:.3f}s)')


            if self.view_img:
                view_imgs(img0=img0)


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/best.pt', help='model.py path(s)')
    # nargs='+'参数可以为一个或多个
    # 形如:python your_script.py --weights path/to/your/model.pt --img-size 800 --conf-thres 0.5 --iou-thres 0.6 --view-img --classes 0 2 3 --agnostic-nms --augment
    parser.add_argument('--img-size', type=int, default=1920, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--view-img', default=True,action='store_true', help='display results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class:class0,or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    # 解析命令行参数保存在opt中
    return opt


def signal_handler(signal, frame):
    print('Exiting...')
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    # 注册信号接收器 Ctrl+C退出程序

    opt = parseArgs()
    print(opt)

    aim_yolo = AimYolov8(opt)
    print('The AimYolov8 Object Created.')

    aim_yolo.run()





