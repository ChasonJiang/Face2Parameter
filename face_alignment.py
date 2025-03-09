

import glob
import json
import multiprocessing
import os
import random
import shutil
from typing import List
import uuid
import cv2

import numpy as np
import tqdm
from src.face_data_utils.FaceCrop import FaceCrop
from asyncio import Lock

locker = Lock()


def readImage(filename):
    try:
        img=cv2.imdecode(np.fromfile(filename,dtype=np.uint8),-1)
        # img = cv2.imread(filename=filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img[:, :, [2, 1, 0]]
    except:
        raise Exception("图片读取失败！")
        
    return img


def face_detect(pid, file_paths, save_dir, other_dir):
    # assert len(args) == 3
    
    # file_paths, save_dir, other_dir = args[0], args[1], args[2]
    face_crop = FaceCrop()
    for idx, file_path in enumerate(file_paths):
        try:
            print(f"pid:{pid} | task: {idx}/{len(file_paths)}")
            img=readImage(file_path)


            faces:list=face_crop.crop(img,alignment=True)
            if len(faces)==0:
                # print(f"跳过图片{file_path}，未检测到人脸！",end="\r")
                # shutil.move(file_path,other_dir)
                continue
            if len(faces)>1:
                # print(f"跳过图片{file_path}，存在多张人脸！",end="\r")
                # shutil.move(file_path,other_dir)
                continue
            
            face=cv2.cvtColor(faces[0], cv2.COLOR_RGB2BGR)
            os.makedirs(os.path.join(save_dir,"face"),exist_ok=True)
            filename = file_path.split(os.sep)[-1].replace(".jpg",".png")
            
            cv2.imwrite(os.path.join(save_dir,"face",filename), face)
            
            # raw_save_dir = os.path.join(save_dir,"raw")
            # os.makedirs(raw_save_dir,exist_ok=True)
            # shutil.move(file_path,raw_save_dir)
            # print(f"图片{file_path}处理完成！",end="\r")
            
        except:
            continue

    





def face_alignment(root_dir:str, save_dir:str, max_workers:int = 16):
    # images_dir = "dataset/raw"
    face_save_dir = os.path.join(save_dir, "aligned")
    other_save_dir = os.path.join(save_dir, "other")

    # os.makedirs(images_dir,exist_ok=True)
    os.makedirs(face_save_dir,exist_ok=True)
    os.makedirs(other_save_dir,exist_ok=True)
    
    

    raw_images:list=glob.glob(os.path.join(root_dir,"**","*.png"), recursive=True)

    # # with ThreadPoolExecutor(max_workers=max_workers) as pool:
    # for item in tqdm.tqdm(raw_images):
    #         # pool.submit(face_detect, (face_crop, item, face_save_dir, other_save_dir))
    #     face_detect(face_crop, item, face_save_dir, other_save_dir)

    procss=[]
    for i in range(max_workers):
        proc = multiprocessing.Process(target=face_detect, args=(i, raw_images[i::max_workers], face_save_dir, other_save_dir))
        procss.append(proc)
        proc.start()

    for proc in procss:
        proc.join()

    # with multiprocessing.Pool(max_workers) as pool:
    #     _ = list(tqdm.tqdm(pool.imap(face_detect, [(raw_images[i::max_workers], face_save_dir, other_save_dir) for i in range(max_workers)]), total=len(raw_images)//max_workers))





if __name__ =="__main__":
    root_dir = r"E:\Datasets\celeba_hq"
    save_dir = r"D:\Project\dataset_gen\dataset\celeba_hq"
    face_alignment(root_dir, save_dir, 8)