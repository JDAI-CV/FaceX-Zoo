import os
import random

root_dir = '/export2/home/wht/oulu_images/train_img_flod/'

origin_txt = '/export2/home/wht/oulu_images/protocols/Protocol_1/Train.txt'
f = open('./train_list/train_list_oulu_p1.txt', 'w')
ff = open(origin_txt, 'r')
video_names = ff.readlines()

for video_name in video_names:
    video_name = video_name.split(',')[1][:-1]
    img_list = os.listdir(os.path.join(root_dir, video_name))
    img_num = len(img_list)
    if img_num > 50:
        random_img = random.sample(range(1, img_num), 50)
    else:
        random_img = random.sample(range(1, img_num), img_num)
    if int(video_name[-1:]) == 1:
        domain_flag = str(1)
    else:
        domain_flag = str(0)
    spoof_type = video_name[-1:]
    for j in random_img:
        img_name = video_name + '/' + img_list[j]
        label = video_name[4:6]

        f.write(img_name + ' ' + label + ' ' + domain_flag + ' ' + spoof_type + '\n')
f.close()
