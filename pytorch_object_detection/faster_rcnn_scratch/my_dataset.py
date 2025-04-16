import json
import logging

import torch
from PIL import Image
from numpy import dtype
from sympy.physics.mechanics import PinJoint
from torch.utils.data import Dataset
import os
from lxml import etree

class VOCDataSet(Dataset):
    """
    这是一个自定义VOC数据集类
    Attribute：
        root:VOC数据集根目录
        img_root: 图片根目录
        annotations_root:标注文件的根目录
        xml_list:可用的标注文件的列表
        class_dict:类别字典
        transform: transforms
    """
    def __init__(self, voc_root:str,year="2012", transforms=None,
                 txt_name: str="train.txt"):
        if "VOCdevkit" not in voc_root:
            voc_root = os.path.join(voc_root, "VOCdevkit")
        self.root = os.path.join(voc_root, f"VOC{year}")
        self.img_root = os.path.join(self.root, "JPEGImages")
        self.annotations_root = os.path.join(self.root, "Annotations")

        txt_path = os.path.join(self.annotations_root, "ImageSets", "Main", txt_name)
        with open(txt_path, "r") as f:
            xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                        for line in f.readlines()]
        self.xml_list = []
        for xml_path in xml_list:
            if os.path.exists(xml_path) is False:
                print(f"Warning, not found {xml_path}, skip this annotation file")
                continue

            with open(xml_path, "r") as f:
                xml_str = f.read()
            xml = etree.fromstring(xml_str)
            data = self.parse_xml_to_dict(xml)["annotation"]
            if "object" not in data:
                print(f"INFO: no objects in {xml_path}, skip this annotation file.")
                continue
            self.xml_list.append(xml_path)
        assert len(self.xml_list) > 0,  "in '{}' file does not find any information.".format(txt_path)

        json_file = './pascal_voc_classes.json'
        assert os.path.exists(json_file), "{} file not exist.".format(json_file)
        with open(json_file, "r") as f:
            self.class_dict = json.load(f)

        self.transforms = transforms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):

        xml_path = self.xml_list[idx]
        with open(xml_path, "r") as f:
            xml_str = f.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        img_path = os.path.join(self.img_root, data["filename"])
        img = Image.open(img_path)
        if img.format != "JPEG" :
            raise ValueError("Image '{}' format not JPEG".format(img_path))

        boxes = []
        labels = []
        iscrowd = []
        objs = data["object"]
        for obj in objs:
            xmin = float(obj["bndbox"]["xmin"])
            ymin = float(obj["bndbox"]["ymin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymax = float(obj["bndbox"]["ymax"])

            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])

            if "difficult" in obj:
                iscrowd.append(obj["difficult"])
            else:
                iscrowd.append(0)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.float32)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        img_id = torch.as_tensor(idx)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["iscrowd"] = iscrowd
        target["area"] = area
        target["img_id"] = img_id

        if self.transforms is not None:
            image, target = self.transforms(img, target)

        return image, target

    def parse_xml_to_dict(self, xml):
        """
        将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree

        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}
    
    def get_height_and_width(self, idx):
        xml_path = self.xml_list[idx]
        with open(xml_path, 'r') as f:
            xml_str = f.read
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    @staticmethod
    def collate_fn(batch):
        return (tuple(zip(*batch)))


if __name__ == '__main__':
    from draw_box_utils import draw_objs
    import torchvision.transforms as ts
    import json
    import random
    import transforms
    import matplotlib.pyplot as plt
    import numpy as np

    json_path = './pascal_voc_classes.json'
    try:
        json_file = open(json_path, "r")
        class_dict = json.load(json_file)
        category_index = {str(v):str(k) for k, v in class_dict.items()}
    except Exception as e:
        print(e)
        exit(-1)
    
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    train_data_set = VOCDataSet(os.getcwd(), "2012", data_transform["train"], "train.txt")
    print(len(train_data_set))
    for index in random.sample(range(0, len(train_data_set)), k=5):
        img, target = train_data_set[index]
        img = ts.ToPILImage()(img)
        plot_img = draw_objs(img,
                             target["boxes"].numpy(),
                             target["labels"].numpy(),
                             np.ones(target["labels"].shape[0]),
                             category_index=category_index,
                             box_thresh=0.5,
                             line_thickness=3,
                             font='arial.ttf',
                             font_size=20)
        plt.imshow(plot_img)
        plt.show()