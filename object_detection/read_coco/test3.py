# 使用pycocotools来计算自己的map
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

json_path = './instances_val2017.json'
coco_true = COCO(json_path)
coco_predict = coco_true.loadRes(resFile='./instances_val2017.json')

coco_evaluator = COCOeval(coco_true, coco_predict, iouType='bbox')
coco_evaluator.evaluate()
coco_evaluator.accumulate()
coco_evaluator.summarize()