"""
@author: JiXuan Xu, Jun Wang
@date: 20201015
@contact: jun21wangustc@gmail.com 
"""
# it's a approximate map
# 15 --> (99+103)/2
# 17, 19; 20, 22; 16; 9 will be used in face crop(25 points)
lms25_2_lms106 = {1:105, 2:106, 3:34, 4:38, 5:43,
                  6:47, 7:52, 8:55, 9:88, 10:94,
                  11:85, 12:91, 13:63, 14:59, 15:99,
                  16:61, 17:71, 18:73, 19:67, 20:80,
                  21:82, 22:76, 23:36, 24:45, 25:17}

# 1: left eye center
# 2: right eye center
# 3: nose tip
# 4: left mouth corner
# 5: right mouth corner
lms5_2_lms25 = {1:1, 2:2, 3:8, 4:11, 5:12}
lms5_2_lms106 = {1:105, 2:106, 3:55, 4:85, 5:91}

def lms106_2_lms25(lms_106):
    lms25 = []
    for cur_point_index in range(25):
        cur_point_id = cur_point_index + 1
        point_id_106 = lms25_2_lms106[cur_point_id]
        cur_point_index_106 = point_id_106 - 1
        cur_point_x = lms_106[cur_point_index_106 * 2]
        cur_point_y = lms_106[cur_point_index_106 * 2 + 1]
        lms25.append(cur_point_x)
        lms25.append(cur_point_y)
    return lms25

def lms106_2_lms5(lms_106):
    lms5 = []
    for cur_point_index in range(5):
        cur_point_id = cur_point_index + 1
        point_id_106 = lms5_2_lms106[cur_point_id]
        cur_point_index_106 = point_id_106 - 1
        cur_point_x = lms_106[cur_point_index_106 * 2]
        cur_point_y = lms_106[cur_point_index_106 * 2 + 1]
        lms5.append(cur_point_x)
        lms5.append(cur_point_y)
    return lms5

def lms25_2_lms5(lms_25):
    lms5 = []
    for cur_point_index in range(5):
        cur_point_id = cur_point_index + 1
        point_id_25 = lms5_2_lms25[cur_point_id]
        cur_point_index_25 = point_id_25 - 1
        cur_point_x = lms_25[cur_point_index_25 * 2]
        cur_point_y = lms_25[cur_point_index_25 * 2 + 1]
        lms5.append(cur_point_x)
        lms5.append(cur_point_y)
    return lms5
