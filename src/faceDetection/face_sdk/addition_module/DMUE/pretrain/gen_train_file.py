import os

def gen_train_file(data_root, train_file):
    train_file_buf = open(train_file, 'w')
    id_list = os.listdir(data_root)
    id_list.sort()
    for label, id_name in enumerate(id_list):
        cur_id_folder = os.path.join(data_root, id_name)
        cur_img_list = os.listdir(cur_id_folder)
        cur_img_list.sort()
        for image_name in cur_img_list:
            cur_image_path = os.path.join(id_name, image_name)
            line = cur_image_path + ' ' + str(label)
            train_file_buf.write(line + '\n')
    
if __name__ == '__main__':
    data_root = "/path/to/your/cropped_msra"
    tain_file = './msra_train_file.txt'
    gen_train_file(data_root, tain_file)
