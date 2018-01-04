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


    return parser



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
    bag = rosbag.Bag(filename)

    # images should be saved in tiff format originally
    image_type = '.tiff'

    ################################
    # Saving the data to csv file  #
    ################################
    df = bp.bag_to_csv(filename, imu_topic, None, None, True)

    #########################################################
    # Saving images                                         #
    #    * calculating the timeshift between IMU and images #
    #########################################################
    bp.bag_to_images(bag, image_topic, imu_topic, image_type, save_dir, True,
                     False)
