#coding=utf-8
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def split_map(datapath, vid_name):
    print(datapath, vid_name)
    write_path = datapath.strip('labels') + 'body_detail/'
    for name in os.listdir(datapath + '/' + vid_name):
        mask = cv2.imread(datapath + '/' + vid_name + '/' + name,0)
        body = cv2.blur(mask, ksize=(5,5))
        body = cv2.distanceTransform(body, distanceType=cv2.DIST_L2, maskSize=5)
        body = body**0.5

        tmp  = body[np.where(body>0)]
        if len(tmp)!=0:
            body[np.where(body>0)] = np.floor(tmp/np.max(tmp)*255)

        if not os.path.exists(write_path+vid_name):
            os.makedirs(write_path+vid_name)
        cv2.imwrite(write_path + vid_name+'/' + name[:-4]+'b.png', body)
        cv2.imwrite(write_path + vid_name+'/' + name[:-4]+'d.png', mask-body)


if __name__=='__main__':
    you_path_to_visha = ""

    test_paths = {
        "train": you_path_to_visha + "ViSha_release/train/labels",
        "test": you_path_to_visha + "ViSha_release/test/labels",
        "val": you_path_to_visha + "ViSha_release/test/labels"
    }
    for key, values in test_paths.items():
        videos = os.listdir(values)
        for item in videos:
            path = os.path.join(values, item)
            split_map(values, item)
