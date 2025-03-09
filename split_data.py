import json
import os
import glob
import random
import shutil

from tqdm import tqdm





def splitting_images(root_dir:str, save_dir:str, ratio:float=None, val_samples:int=None):
    image_paths = glob.glob(os.path.join(root_dir,"**","*.png"), recursive=True)
    random.shuffle(image_paths)
    if val_samples is not None:
        n = val_samples
    elif ratio is not None:
        n = int(len(image_paths)*ratio)
    else:
        raise ValueError("ratio or num_samples should be specified")
    train_paths = image_paths[:-n]
    test_paths = image_paths[-n:]
    # train_dir = os.path.join(save_dir, "train")
    # test_dir = os.path.join(save_dir, "val")
    # os.makedirs(train_dir, exist_ok=True)
    # os.makedirs(test_dir, exist_ok=True)
    # for path in train_paths:
    #     shutil.move(path, train_dir)
    # for path in test_paths:
    #     shutil.move(path, test_dir)

    with open(os.path.join(save_dir, "train.txt"), "w") as f:
        for path in tqdm(train_paths):
            f.write(path.replace("\\", os.sep).replace("/", os.sep)+"\n")

    with open(os.path.join(save_dir, "val.txt"), "w") as f:
        for path in tqdm(test_paths):
            f.write(path.replace("\\", os.sep).replace("/", os.sep)+"\n")

def splitting_features(root_dir:str, save_dir:str, ratio:float=None, val_samples:int=None):
    image_paths = glob.glob(os.path.join(root_dir,"features","*.npy"), recursive=True)
    random.shuffle(image_paths)
    if val_samples is not None:
        n = val_samples
    elif ratio is not None:
        n = int(len(image_paths)*ratio)
    else:
        raise ValueError("ratio or num_samples should be specified")
    train_paths = image_paths[:-n]
    test_paths = image_paths[-n:]
    # train_dir = os.path.join(save_dir, "train")
    # test_dir = os.path.join(save_dir, "val")
    # os.makedirs(train_dir, exist_ok=True)
    # os.makedirs(test_dir, exist_ok=True)
    # for path in train_paths:
    #     shutil.move(path, train_dir)
    # for path in test_paths:
    #     shutil.move(path, test_dir)

    with open(os.path.join(save_dir, "train_features.txt"), "w") as f:
        for path in tqdm(train_paths):
            f.write(os.path.basename(path)[:-4]+"\n")

    with open(os.path.join(save_dir, "val_features.txt"), "w") as f:
        for path in tqdm(test_paths):
            f.write(os.path.basename(path)[:-4]+"\n")



def renames_images(root_dir:str, save_dir:str):
    image_paths = glob.glob(os.path.join(root_dir,"*.png"), recursive=True)
    os.makedirs(save_dir, exist_ok=True)
    for path in tqdm(image_paths):
        name = os.path.basename(path).split("@")[1]
        save_path = os.path.join(save_dir, name)
        shutil.move(path, save_path)


def query_and_copy(ref_dir:str, query_dir:str, save_dir:str):
    ref_paths = glob.glob(os.path.join(ref_dir,"*.png"), recursive=True)
    query_paths = glob.glob(os.path.join(query_dir,"*.png"), recursive=True)
    os.makedirs(save_dir, exist_ok=True)

    database= {os.path.basename(path).split("@")[1][:-4]:path for path in ref_paths}

    for path in tqdm(query_paths):
        name = os.path.basename(path).split("@")[1][:-4]
        ref_path = database.get(name, None)
        if ref_path is None:
            print(f"No reference image found for {name}")
            continue
        save_path = os.path.join(save_dir, os.path.basename(path))
        shutil.copy(ref_path, save_path)


def replace_label_names(path:str, save_path:str):
    with open(path, "r") as f:
        labels = json.load(f)
    
    new_labels = {}
    for k in labels.keys():
        new_k = k.split("@")[1]
        new_labels[new_k] = labels[k]


    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(new_labels, f, indent=4, ensure_ascii=False)



if __name__ == "__main__":
    # root_dir = r"data/images"
    # save_dir = "data/images"
    # renames_images(root_dir, save_dir)


    # ref_dir = r"D:\Project\dataset_gen\dataset\110k_cleaned\images"
    # query_dir = r"data/aug_images"
    # save_dir = "data/images"
    # query_and_copy(ref_dir, query_dir, save_dir)


    # path = r"D:\Project\dataset_gen\dataset\110k_cleaned\labels.json"
    # save_path = "data/labels_new.json"

    # replace_label_names(path, save_path)

    root_dir = r"data/"
    save_dir = "data/"
    # splitting_images(root_dir, save_dir, val_samples=100)

    # splitting_features(root_dir, save_dir, val_samples=100)


    with open(r"data\labels.json", "r") as f:
        labels = json.load(f)

    print(labels.keys())
