"""
@author: Yinglu Liu, Jun Wang  
@date: 20201012   
@contact: jun21wangustc@gmail.com 
"""

import numpy as np

def read_landmark_106_file(filepath):
    map = [[1,2],[3,4],[5,6],7,9,11,[12,13],14,16,18,[19,20],21,23,25,[26,27],[28,29],[30,31],33,34,35,36,37,42,43,44,45,46,51,52,53,54,58,59,60,61,62,66,67,69,70,71,73,75,76,78,79,80,82,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103]
    line = open(filepath).readline().strip()
    pts1 = line.split(' ')[58:-1]
    assert(len(pts1) == 106*2)
    pts1 = np.array(pts1, dtype = np.float)
    pts1 = pts1.reshape((106, 2))
    pts = np.zeros((68,2)) # map 106 to 68
    for ii in range(len(map)):
        if isinstance(map[ii],list):
            pts[ii] = np.mean(pts1[map[ii]], axis=0)
        else:
            pts[ii] = pts1[map[ii]]
    return pts

def read_landmark_106_array(face_lms):
    map = [[1,2],[3,4],[5,6],7,9,11,[12,13],14,16,18,[19,20],21,23,25,[26,27],[28,29],[30,31],33,34,35,36,37,42,43,44,45,46,51,52,53,54,58,59,60,61,62,66,67,69,70,71,73,75,76,78,79,80,82,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103]
    pts1 = np.array(face_lms, dtype = np.float)
    pts1 = pts1.reshape((106, 2))
    pts = np.zeros((68,2)) # map 106 to 68
    for ii in range(len(map)):
        if isinstance(map[ii],list):
            pts[ii] = np.mean(pts1[map[ii]], axis=0)
        else:
            pts[ii] = pts1[map[ii]]
    return pts

def read_landmark_106(filepath):
    map = [[1,2],[3,4],[5,6],7,9,11,[12,13],14,16,18,[19,20],21,23,25,[26,27],[28,29],[30,31],33,34,35,36,37,42,43,44,45,46,51,52,53,54,58,59,60,61,62,66,67,69,70,71,73,75,76,78,79,80,82,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103]
    lines = open(filepath).readlines() # load landmarks
    pts1 = [_.strip().split() for _ in lines[1:107]]
    pts1 = np.array(pts1, dtype = np.float)
    pts = np.zeros((68,2)) # map 106 to 68
    for ii in range(len(map)):
        if isinstance(map[ii],list):
            pts[ii] = np.mean(pts1[map[ii]], axis=0)
        else:
            pts[ii] = pts1[map[ii]]
    return pts
    
def read_bbox(filepath):
    lines = open(filepath).readlines()
    bbox = lines[0].strip().split()
    bbox = [int(float(_)) for _ in bbox]
    return np.array(bbox)

