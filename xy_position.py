from PIL import Image, ImageDraw, ImageFont

# 加载图片
image_path = 'position.png'  # 替换为您的图片路径
image = Image.open(image_path)
draw = ImageDraw.Draw(image)

# 坐标列表，例如：[(x1, y1), (x2, y2), ...]
coordinates = [
    (244.0965118408203, 192.67054748535156),  # 示例坐标1
    (229.957763671875, 195.23287963867188),  # 示例坐标2
    # 添加更多坐标...
]

# 标记大小和颜色
radius = 5
color = (255, 0, 0)  # 红色

# # 加载字体（如果需要在图片上写字）
# font_path = 'path/to/your/font.ttf'  # 替换为您的字体文件路径
# font_size = 15
# font = ImageFont.truetype(font_path, font_size)

# 遍历坐标并绘制标记
for coord in coordinates:
    x, y = coord
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=color)
    # 如果需要在坐标下方添加文本说明，取消下面一行的注释
    # draw.text((x, y + 10), f'坐标: {x},{y}', font=font, fill=color)

rgb_image = image.convert('RGB')
rgb_image.save('marked_image.jpg')  # 替换为保存图片的路径
image.show()