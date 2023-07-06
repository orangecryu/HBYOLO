from xml.dom.minidom import parse
import matplotlib.pyplot as plt
import cv2 as cv
import os

train_annotation_path = 'D:/ycr/proj/yolov7-main/datasets/Detection_voc_format/voc/car/Annotations/'  # 训练集annotation的路径
train_image_path = 'D:/ycr/proj/yolov7-main/datasets/Detection_voc_format/voc/car/JPEGImages/'  # 训练集图片的路径
# 展示图片的数目
show_num = 0


# 打开xml文档

def parase_xml(xml_path):
    """
    输入：xml路径
    返回：image_name, width, height, bboxes
    """
    domTree = parse(xml_path)
    rootNode = domTree.documentElement
    # 得到object,sizem,图片名称属性
    object_node = rootNode.getElementsByTagName("object")
    shape_node = rootNode.getElementsByTagName("size")
    image_node = rootNode.getElementsByTagName("filename")
    image_name = image_node[0].childNodes[0].data
    bboxes = []
    # 解析图片的长宽
    for size in shape_node:
        width = int(size.getElementsByTagName('width')[0].childNodes[0].data)
        height = int(size.getElementsByTagName('height')[0].childNodes[0].data)
    # 解析图片object属性
    for obj in object_node:
        # 解析name属性,并统计类别数
        class_name = obj.getElementsByTagName("name")[0].childNodes[0].data
        # 解析bbox属性，并统计bbox的大小
        bndbox = obj.getElementsByTagName("bndbox")

        for bbox in bndbox:
            x1 = int(bbox.getElementsByTagName('xmin')[0].childNodes[0].data)
            y1 = int(bbox.getElementsByTagName('ymin')[0].childNodes[0].data)
            x2 = int(bbox.getElementsByTagName('xmax')[0].childNodes[0].data)
            y2 = int(bbox.getElementsByTagName('ymax')[0].childNodes[0].data)
            bboxes.append([class_name, x1, y1, x2, y2])
    return image_name, width, height, bboxes


def read_voc(train_annotation_path, train_image_path, show_num):
    """
    train_annotation_path:训练集annotation的路径
    train_image_path：训练集图片的路径
    show_num：展示图片的大小
    """
    # 用于统计图片的长宽
    total_width, total_height = 0, 0
    # 用于统计图片bbox长宽
    bbox_total_width, bbox_total_height, bbox_num = 0, 0, 0
    min_bbox_size = 40000
    max_bbox_size = 0
    # 用于统计聚类所用的图片长宽，bbox长宽
    img_wh = []
    bbox_wh = []
    # 用于统计标签
    total_size = []
    class_static = {'person': 0, 'crazing': 0, 'inclusion': 0, 'patches': 0, 'pitted_surface': 0, 'rolled-in_scale': 0,
                    'scratches': 0}
    num_index = 0

    for root, dirs, files in os.walk(train_annotation_path):
        for file in files:
            num_index += 1
            xml_path = os.path.join(root, file)
            image_name, width, height, bboxes = parase_xml(xml_path)
            image_path = os.path.join(train_image_path, image_name)
            img_wh.append([width, height])
            total_width += width
            total_height += height

            # 如果需要展示，则读取图片
            if num_index < show_num:
                image_path = os.path.join(train_image_path, image_name)
                image = cv.imread(image_path)
            # 统计有关bbox的信息
            wh = []
            for bbox in bboxes:
                class_name = bbox[0]
                class_static[class_name] += 1
                x1, y1, x2, y2 = bbox[1], bbox[2], bbox[3], bbox[4]
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                bbox_size = bbox_width * bbox_height
                # 统计bbox的最大最小尺寸
                if min_bbox_size > bbox_size:
                    min_bbox_size = bbox_size
                if max_bbox_size < bbox_size:
                    max_bbox_size = bbox_size
                total_size.append(bbox_size)
                # 统计bbox平均尺寸
                bbox_total_width += bbox_width
                bbox_total_height += bbox_height
                # 用于聚类使用
                wh.append([bbox_width / width, bbox_height / height])  # 相对坐标
                bbox_num += 1
                # 如果需要展示，绘制方框
                if num_index < show_num:
                    cv.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
                    cv.putText(image, class_name, (x1, y1 + 10), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.2,
                               color=(0, 255, 0), thickness=1)
            bbox_wh.append(wh)
            # 如果需要展示
            if num_index < show_num:
                plt.figure()
                plt.imshow(image)
                plt.show()

    # 去除2个检查文件
    # num_index -= 2
    print("total train num is: {}".format(num_index))
    print("avg total_width is {}, avg total_height is {}".format((total_width / num_index), (total_height / num_index)))
    print("avg bbox width is {}, avg bbox height is {} ".format((bbox_total_width / bbox_num),
                                                                (bbox_total_height / bbox_num)))
    print("min bbox size is {}, max bbox size is {}".format(min_bbox_size, max_bbox_size))
    print("class_static show below:", class_static)

    return img_wh, bbox_wh


img_wh, bbox_wh = read_voc(train_annotation_path, train_image_path, show_num)