import os
import cv2
import lmdb
import skimage.io as io
from tqdm import tqdm


lmdb_output = "/path/to/your/AffectNet_lmdb/"
lb_txt = "/path/to/your/lb2.txt"
ori_root = "/path/to/your/align_larger_256/"
size = (256, 256)


lines = open(lb_txt, 'r').readlines()
env_w = lmdb.open(lmdb_output, map_size=6000e7)
txn_w = env_w.begin(write=True)

for i in tqdm(range(len(lines))):
    k = lines[i].split(' ')[0]
    img_path = k.split('_')[1]

    img = io.imread(os.path.join(ori_root, img_path))[:,:,:3]
    img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)

    txn_w.put(key=k.encode('utf-8'), value=img.tobytes())

txn_w.commit()
env_w.close()
