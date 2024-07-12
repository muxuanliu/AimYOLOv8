import ctypes

LONG = ctypes.c_long
DWORD = ctypes.c_ulong
ULONG_PTR = ctypes.POINTER(DWORD)
WORD = ctypes.c_ushort

INPUT_MOUSE = 0


class MouseInput(ctypes.Structure):
    _fields_ = [
        ('dx', LONG),
        ('dy', LONG),
        ('mouseData', DWORD),
        ('dwFlags', DWORD),
        # DWORD无符号长整型数据
        ('time', DWORD),
        ('dwExtraInfo', ULONG_PTR)
    ]


class InputUnion(ctypes.Union):
    _fields_ = [
        ('mi', MouseInput)
    ]


class Input(ctypes.Structure):
    _fields_ = [
        ('type', DWORD),
        ('iu', InputUnion)
    ]


def mouse_input_set(flags, x, y, data):
    return MouseInput(x, y, data, flags, 0, None)


def input_do(structure):
    if isinstance(structure, MouseInput):
        # 检查structure是否是一个MouseInput的一个实例
        return Input(INPUT_MOUSE, InputUnion(mi=structure))
    # INPUT_MOUSE鼠标输入事件,官方参数
    # 在Windows API中,还有
    # INPUT_KEYBOARD：表示键盘输入事件。
    # INPUT_HARDWARE：表示硬件输入事件。
    # INPUT_TOUCH：表示触摸输入事件。
    # INPUT_PEN：表示笔输入事件。
    raise TypeError('Cannot create Input structure!')


def mouse_input(flags, x=0, y=0, data=0):
    # 未提供的参数默认为0
    return input_do(mouse_input_set(flags, x, y, data))


def SendInput(*inputs):
    n_inputs = len(inputs)
    lp_input = Input * n_inputs
    p_inputs = lp_input(*inputs)
    cb_size = ctypes.c_int(ctypes.sizeof(Input))
    return ctypes.windll.user32.SendInput(n_inputs, p_inputs, cb_size)


if __name__ == '__main__':
    SendInput()
