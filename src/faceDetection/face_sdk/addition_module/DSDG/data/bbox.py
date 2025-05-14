from mtcnn import MTCNN
import cv2
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

detector = MTCNN()

root_dir = '/export2/home/wht/oulu_images/train_img_flod/'
bbox_dir = '/export2/home/wht/oulu_images/train_bbox_flod/'

if not os.path.exists(bbox_dir):
    os.makedirs(bbox_dir)

img_list = []
f = open('/export2/home/wht/oulu_images/images_list.txt', 'r')

for line in f.readlines():
    img_list.append(str(line[:-1]))

for i, img in enumerate(img_list):
    imgpath = img.split(',')[1]
    imgdir = img.split(',')[0]
    image = cv2.cvtColor(cv2.imread(root_dir + imgpath), cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(image)
    if len(result) < 0.5:
        continue
    bounding_box = result[0]['box']
    bounding_box = ' '.join(str(x) for x in bounding_box)
    dat_path = os.path.join(bbox_dir + imgdir)
    if not os.path.exists(dat_path):
        os.makedirs(dat_path)
    fp = open(bbox_dir + imgpath[:-4] + '.dat', 'w')
    fp.write(bounding_box)
    fp.close()

