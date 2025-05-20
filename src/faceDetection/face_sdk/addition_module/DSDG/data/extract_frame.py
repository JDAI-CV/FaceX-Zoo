import os
import cv2


def save_image(image, img_dir, vedio_name, num):
    flod_path = img_dir + vedio_name[:-4] + '/'
    if not os.path.exists(flod_path):
        os.makedirs(flod_path)
    address = flod_path + vedio_name[:-4] + '_' + str(num) + '_scene.jpg'
    cv2.imwrite(address, image)


root_dir = '/export2/home/wht/Oulu_Origin/Train_files/'

img_dir = '/export2/home/wht/oulu_images/train_img_flod/'

if not os.path.exists(img_dir):
    os.makedirs(img_dir)

id_list = os.listdir(root_dir)
num = len(id_list)

for id in id_list[:num]:
    vedio_dir = os.path.join(root_dir, id)
    vedio_lists = os.listdir(vedio_dir)
    vedioname_list = []
    for name in vedio_lists:
        if name[-4:] == '.avi':
            vedioname_list.append(name)
    for j, vedio_name in enumerate(vedioname_list):
        videoCapture = cv2.VideoCapture(vedio_dir + '/' + vedio_name)
        success, frame = videoCapture.read()
        i = 1
        while success:
            success, frame = videoCapture.read()
            if success == True:
                save_image(frame, img_dir, vedio_name, i)
