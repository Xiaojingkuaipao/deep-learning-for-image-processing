import os
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

json_path = './instances_val2017.json'
img_path = './val2017'

coco = COCO(json_path)

# 获取所有图片的key
ids = sorted(list(coco.imgs.keys()))
print(f"number of images: {len(ids)}")

coco_classes = {v["id"]:v["name"] for k, v in coco.cats.items()}

# 遍历前三张图片并且把目标框打印出来
for img_id in ids[:3]:
    # 根据图像id获取所有的标注信息id
    ann_idx = coco.getAnnIds(imgIds=img_id)

    # 根据ann_id获取所有标注信息
    targets = coco.loadAnns(ann_idx)

    path = coco.loadImgs(img_id)[0]['file_name']
    img = Image.open(os.path.join(img_path, path)).convert('RGB')
    draw = ImageDraw.Draw(img)
    for target in targets:
        x, y, w, h = target["bbox"]
        x1, y1, x2, y2 = x, y, int(x + w), int(y + h)
        draw.rectangle((x1, y1, x2, y2))
        draw.text((x1, y1), coco_classes[target["category_id"]])
    plt.imshow(img)
    plt.show()
