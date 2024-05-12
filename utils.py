import os
import json
import pickle
import random

import matplotlib.pyplot as plt


def read_split_data_mode(root: str, mode: str = 'train', supported=[".jpg", ".JPG", ".png", ".PNG", ".JPEG"]):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    assert mode in ['train', 'val'], "mode should be either 'train' or 'val'."

    # 根据mode确定数据集目录
    dataset_dir = os.path.join(root, mode)
    # 遍历文件夹，一个文件夹对应一个类别
    dataset_class = [cla for cla in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, cla))]
    # 排序，保证平台顺序一致
    dataset_class.sort()

    # 生成类别名称以及对应数字索引
    class_indices = dict((k, v) for v, k in enumerate(dataset_class))
    # 将 index:class 写入指定json文件中
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    images_path = []  # 存储所有图片路径
    images_label = []  # 存储图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数

    # 遍历每个文件夹下的文件
    for cla in dataset_class:
        cla_path = os.path.join(dataset_dir, cla)
        # 遍历获得supported支持的文件路径
        images = [os.path.join(dataset_dir, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        images.sort()
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))

        for img_path in images:
            images_path.append(img_path)
            images_label.append(image_class)

    print("{} images were found in the {} dataset.".format(sum(every_class_num), mode))
    return images_path, images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list