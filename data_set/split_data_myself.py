import os
from shutil import copy, rmtree
import random

def mk_file(file_path:str):
    if os.path.exists(file_path):
        rmtree(file_path)
    os.makedirs(file_path)

def main():
    random.seed(0)

    split_rate = 0.1

    cwd = os.getcwd()

    data_root = os.path.join(cwd, 'flower_data')

    origin_flower_path = os.path.join(data_root, 'flower_photos')

    assert os.path.exists(origin_flower_path), f"path {origin_flower_path} does not exist"

    flower_class = [cls for cls in os.listdir(origin_flower_path)
                    if os.path.isdir(os.path.join(origin_flower_path, cls))]

    train_root = os.path.join(data_root, 'train')
    mk_file(train_root)
    for cls in flower_class:
        mk_file(os.path.join(train_root, cls))

    val_root = os.path.join(data_root, 'val')
    mk_file(val_root)
    for cls in flower_class:
        mk_file(os.path.join(val_root, cls))

    for cls in flower_class:
        cls_path = os.path.join(origin_flower_path, cls)
        images = os.listdir(cls_path)
        eval_index = random.sample(images, int(len(images) * split_rate))
        for index, image in enumerate(images):
            if index in eval_index:
                image_path = os.path.join(cls_path, image)
                new_path = os.path.join(val_root, cls)
                copy(image_path, new_path)
            else:
                image_path = os.path.join(cls_path, image)
                new_path = os.path.join(train_root, cls)
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cls, index + 1, len(images)), end="")  # processing bar
        print()
    print("preprocessing done!")

if __name__ == "__main__":
    main()