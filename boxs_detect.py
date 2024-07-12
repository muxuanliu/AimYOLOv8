import mss
import win32api
import win32con
from pynput import mouse
from utils.utils import *
from capScreen import *
from mouse_sendin import *
import argparse
from models.experimental import attempt_load
import sys
import signal


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
    # interference
    pred = model(img, augment=augment)[0]
    # 非极大值抑制
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic)
    # pred：(x1, y1, x2, y2, conf, cls)
    # 会返回一个列表
    # Returns:
    #      detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    return pred


def calculate_center(xyxy):
    """
    :param xyxy: 左上角坐标和右下角坐标的列表
    :return: 中心坐标
    """
    c1, c2 = (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3])
    center_x = int((c1[0] + c2[0]) / 2)
    center_y = int((c1[1] + c2[1]) / 2)
    return center_x, center_y


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
        self.model = attempt_load(self.weights, map_location=self.device)  # Load FP32 model
        # 加载模型到对应的设备上
        self.img_size = check_img_size(self.img_size, s=self.model.stride.max())
        # 检查并返回合适模型最大步长的尺寸
        self.half = self.device.type != 'cpu'
        # 半精度只支持cuda
        if self.half:
            self.model.half()
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
            aim_person_center = []
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

            det = pred[0]

            s = ''
            s += '%gx%g ' % img.shape[2:]
            # img的形状为[batch_size, channels, height, width]，img.shape[2:]表示从第三个元素开始到最后一个
            # print(det)
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                # 将推理后得到的坐标转换为在捕获的图片上的坐标
                # 推理得到的坐标是在img上得到的坐标，img的大小与捕获的图片img0大小不同，需要坐标转换

                # print(det)
                for c in det[:, -1].unique():
                    # 遍历到当前类别，会判断所有结果是否符合
                    # det = [
                    #     [10, 20, 50, 80, 1],  # 类别为 1
                    #     [30, 40, 70, 90, 2],  # 类别为 2
                    #     [15, 25, 55, 85, 1],  # 类别为 1
                    #     [35, 45, 75, 95, 3],  # 类别为 3
                    #     [40, 50, 80, 100, 1],  # 类别为 1
                    #     [20, 30, 60, 90, 2],  # 类别为 2
                    #     [25, 35, 65, 95, 1]  # 类别为 1
                    # ]
                    # 若c是1，则会返回[True, False, True, False, True, False, True]
                    n = (det[:, -1] == c).sum()
                    # 每类的数量
                    s += '%g %s, ' % (n, self.names[int(c)])
                    # 每类的名字

                for *xyxy, conf, cls in det:

                    label = '%s %.2f' % (self.names[int(cls)], conf)
                    plot_one_box(xyxy, img0, label=label, color=self.colors[int(cls)], line_thickness=3)
                    center_x, center_y = calculate_center(xyxy)
                    print(center_y,center_x)
                    aim_person_center.append([center_x, center_y])
                    # print(aim_person_center_head)

                    if int(cls) == 2 or int(cls) == 3:
                        aim_person_center_head.append([center_x, center_y])
                        # print(aim_person_center_head)
            print(f'{s} Done. ({t2 - t1:.3f}s)')

            # 计算中心坐标，若有多个目标，只取其中最近的一个
            # print(aim_person_center_head)
            move_mouse(mouse_control, aim_person_center_head)

            if self.view_img:
                view_imgs(img0=img0)


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/best_200.pt', help='model.py path(s)')
    # nargs='+'参数可以为一个或多个
    # 形如:python your_script.py --weights path/to/your/model.pt --img-size 800 --conf-thres 0.5 --iou-thres 0.6 --view-img --classes 0 2 3 --agnostic-nms --augment
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
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





