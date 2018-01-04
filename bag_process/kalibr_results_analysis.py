import numpy as np
import pandas as pd
import yaml
import os
import sys
import glob

import utils.utils as ut


path = '/home/viper/Documents/Codes/ros/camera_imu_calibration_kalibr/results_yaml/'
filename1 = 'camchain-imucam-kalibr_raw_RPY_4A3_'

filename2 = 'camchain-imucam-kalibr_raw_RPY_A3_'

extensions = ['Kannala_Omni', 'Mei', 'Eynard']

cam_imu_matrices = np.zeros((3, 4, 4))

for i, ext in enumerate(extensions):
    filePath = path + filename1 + ext + '.yaml'
    stream = open(filePath, 'r')
    docs = yaml.load_all(stream)

    for doc in docs:
        cam_imu_matrices[i, :, :] = np.asarray(doc['cam0']['T_cam_imu'])



trans_matrices = np.zeros((3, 3, 4, 4))
angle_matrices = np.zeros((3, 3, 3))

for i, R1 in enumerate(cam_imu_matrices):
    for j, R2 in enumerate(cam_imu_matrices):
        R12 = np.dot(R2.T, R1)
        trans_matrices[i, j, :, :] = R12
        print('comparison between {} and {} method'. format(extensions[i], extensions[j]))
        print('R12 is {}'.format(R12))
        euler_angle = ut.mat2euler(R12[0:3, 0:3])
        angle_matrices[i, j, :] = euler_angle
        print('euler_angle is {}'.format(euler_angle))
        print ('')
