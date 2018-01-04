#!/usr/bin/env python
import numpy as np
import pandas as pd
import os
import sys
import argparse
import glob

import rosbag
import rospy
import rosbag_pandas

import bag.bag as bp
import utils.utils as ut
import argparse


def buildParser():
    ''' Builds the parser for reading the command line arguments'''
    parser = argparse.ArgumentParser(
        description='Script to parse bagfile to csv file')

    parser.add_argument('-filename', help='Bag file to read',
                        type=str)

    parser.add_argument('-image', '--image_topic',
                        help='list or regex for topics to include',
                        type=str
                        )

    parser.add_argument('-imu', '--imu_topic',
                        help='list or regex for topics to exclude',
                        type=str)

    parser.add_argument('-savedir', '--savedir',
                        help='save directory ',
                        type=str)

    parser.add_argument('-image_type', help='type of images to save',
                        type=str)


    parser.add_argument('-cam', '--camerafile',
                        help='name of camerafile file',
                        type=str)

    parser.add_argument('-cam_model', help='Bag file to read',
                        type=str)


    return parser



def write_kalibr_yaml(filename, imu_topic, camera_file, camera_model):
    """
    Writing yaml file for kalibr based on imu_topic and camera_file
    """
    df = rosbag_pandas.bag_to_dataframe(filename, include=imu_topic,
                                        exclude=None, parse_header=True)
    df_mat = df.values

    imu_quaternion = df_mat[:, -4:]
    print imu_quaternion.shape

    imu_euler = ut.quat2euler(imu_quaternion)

    Ric = ut.euler2mat(*imu_euler[0])
    Rci = Ric.T

    T_cam_imu = np.zeros((4, 4))
    T_cam_imu[0:3, 0:3] = Rci
    T_cam_imu[3, 3] = 1

    print ('The first IMU recording on the camera indicate the rotation of imu\'
    \'when placed on the camera in i frame:\n {}'.format(Ric))

    print ('equivalenty the imu in camera frame will be Rci:\n {} '.format(Rci))

    bag = rosbag.Bag(filename)

    sel_imgs_array, sel_imgs_index, timeshift =
    bp.get_timeshift_imu_images(bag, image_topic, imu_topic)

    print ('timeshift between imu and image recordings:\n\'
    \'{}'.format(timeshift))

    model = 'omni'
    dist_model = 'radtan'

    K, dist_coef, intrinsics = ut.import_calib(camera_file, camera_model)

    #N = 320    #M = 260
    N = K['width']
    M = K['height']
    print('width {}, height {}'.format(N, M))
    bp.write_yaml(model, intrinsics, dist_model, dist_coef, T_cam_imu,
                  timeshift, N, M, 'camchain.yaml')




if __name__ == '__main__':

    """
    Main entry point for the function. Reads the command line arguements
    and performs the requested actions
    """
    # Build the command line argument parser
    parser = buildParser()
    # Read the arguments that were passed in
    args = parser.parse_args()
    print(args)
    filename = args.filename
    image_topic = args.image_topic
    imu_topic = args.imu_topic
    save_dir = args.savedir

    camera_file = args.camerafile
    camera_model = args.cam_model
    bag = rosbag.Bag(filename)

    # images should be saved in tiff format originally
    image_type = '.tiff'

    ############################################################
    # Writing yaml file for kalinr based on camera parameters  #
    ############################################################
    write_kalibr_yaml(filename, imu_topic, camera_file, camera_model)

    ################################
    # Saving the data to csv file  #
    ################################
    df = bp.bag_to_csv(filename, imu_topic, None, None, True, True)

    #########################################################
    # Saving images                                         #
    #    * calculating the timeshift between IMU and images #
    #########################################################
    bp.bag_to_images(bag, image_topic, imu_topic, image_type, save_dir, False, False)


    ######################################################################
    # Converting the images from tiff to normalized png                  #
    # The images will be saved in the same folder as the original images #
    ######################################################################
    origin_ext = '.tiff'
    save_ext = '.png'
    bp.rename_images(save_dir, origin_ext, save_ext, remove_str=None)

    print('If you are interested to use superpixel images rather than raw \'
    \'images you should first:\n')
    print('* use pix2image to save angle images, Note : use\'
    \'the original tiff format\n')
    print('* rename and remove I0/90/45/135 extension using\'
    \'rename_image function\n')

    print('Run kalibr_createbag --folder dir --output-bag name to save name.bag \n',
    'dir should contain -cam0 -imu0.csv')
