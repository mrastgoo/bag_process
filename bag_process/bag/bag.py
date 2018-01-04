from __future__ import print_function, division
import numpy as np
import pandas as pd
import os
import sys
import argparse
import cv2
import glob
import random

from scipy.io import loadmat
from skimage import io
from skimage import img_as_float, img_as_uint
from itertools import islice
import matplotlib.pyplot as plt
import seaborn as sns


import rospy
import rosbag
import rosbag_pandas
import genpy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

############################################################
# From here forward functions are special for bag_process  #
############################################################

def _find_nearest_value(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def _find_nearest_images(imu_array, imgs_array, k=1):
    """
    Finds a list of images which have the closest timestamp to recording

    Parameters
    ----------
    imu_array: np.array (n,), imu timestamp array

    img_array: np.array (m,), imgs timestamp array

    k: int (default=1) number of nearest neighbors to be considered

    Return
    ------
    imgs_sel_array: np.array, shape (k*n, ) imgs have the closest
        timestamp to imu recording
    diff_time: array, the difference of imu timestamps to its closest image
        timestamps

    idx : array of indexes of selected images

    """
    diff_time = []
    imgs_sel_array = []
    index = []
    darray = np.asarray(imgs_array)
    for t1 in imu_array:
        for i in range(0, k):
            if not np.size(darray) == 0 :
                t2, t2_idx = _find_nearest_value(darray, t1)
                imgs_sel_array.append(t2)
                diff_time.append(t2 - t1)
                darray = np.delete(darray, t2_idx)
                index.append(np.where(imgs_array==t2))
        #darray = np.delete(darray, t2_idx)
    index = np.ravel(np.asarray(index))
    return np.asarray(imgs_sel_array), index, np.asarray(diff_time)

def get_timestamp_of_topic(bag, topic_name):
    timestamp_list = []
    for topic, msg, t in bag.read_messages(topics=topic_name):
        timestamp_list.append(genpy.rostime.Time.to_time(msg.header.stamp))
    timestamp_array = np.asarray(timestamp_list)

    return timestamp_array


def get_timeshift_imu_images(bag, image_topic, imu_topic, show_plot=False):
    """
    find the timeshift between IMU and images

    Parameter
    ---------
    bag: rosbag object from the loaded bag

    images_topic: string, the topics where images are publised

    imu_topic: string, the topic where imu information are published

    show_graph: bool, defalut= False, shows the graph of time difference
    between imu timestamps and closes images timestamps

    Returns
    -------
    sel_imgs_array: output of find_nearest_images(imu_array, images_array)

    sel_imgs_indexes: indexes of the selected images from the original list of
        images from the bag.

    mean_shiftime: float, mean time shift between imu and images timestamps,
        considering after 120 first recording

    """
    imgs_timestamp = get_timestamp_of_topic(bag, image_topic)
    imu_timestamp = get_timestamp_of_topic(bag, imu_topic)

    # number of neighbors
    sel_imgs_array, sel_imgs_indexes, diff_time = _find_nearest_images(
        imu_timestamp, imgs_timestamp)

    mean_timeshift = np.mean(diff_time[120:])
    print ('first difference time {}'.format(diff_time[0]))
    print ('Mean difference time {}'.format(np.mean(diff_time)))
    print ('Mean difference time after the first 120 samples {}'.format(mean_timeshift))

    if show_plot:
        plt.figure()
        plt.plot(diff_time)
        plt.show()


    return sel_imgs_array, sel_imgs_indexes, -mean_timeshift



def bag_to_csv(bag_name, include, exclude=None, output=None, header=False,
               imu_kalibr=False):
    """bag_to_csv saves the csv file based on mentioned topic

    Parameters:
    -----------
    bag_name: string, path of the bag

    include: the topic or topics to be included

    exclude: the topic to be excluded from the final csv file

    output: string, the name of the output file

    header: bool, (default=False), either include the header timestamp or not

    imu_kalibr: bool, (default=False), either the final file is for kalibr
    calibration or not, In case of kalibr calibration `imu0.csv` file, Only
    will be saved which includes the angular velocity (gyroscope data) and
    linear acceleroration (accelerometer data) with their timestamp.

    Returns:
    --------
    Saving an csv file in the current directory

    df: Pandas dataframe of the saved file

    """
    if imu_kalibr :
        df_origin = rosbag_pandas.bag_to_dataframe(
            bag_name, include='/imu_3dm_node/imu/data', exclude=exclude,
            parse_header=header)
        df_origin = rosbag_pandas.clean_for_export(df_origin)

        base_name = 'imu_3dm_node_imu_data__'
        gyro_name = 'angular_velocity_'
        acc_name = 'linear_acceleration_'
        imu0_df = df_origin [
            [base_name + 'header_stamp_secs', base_name + 'header_stamp_nsecs',
             base_name + gyro_name + 'x',
             base_name + gyro_name + 'y', base_name + gyro_name + 'z',
             base_name + acc_name + 'x' ,
             base_name + acc_name + 'y', base_name + acc_name + 'z']]

        imu0_mat = imu0_df.values
        timestamp = []
        for i in range(0, imu0_mat.shape[0]):
            string = str(int(imu0_mat[i, 0])) + str(int(imu0_mat[i, 1]))
            timestamp.append(int(string))

        timestamp = np.asarray(timestamp)
        imu0_mat[:, 0] = timestamp
        imu0_mat = np.delete(imu0_mat, 1, 1)

        df = pd.DataFrame(
            imu0_mat, columns = ['timestamp', 'omega_x', 'omega_y', 'omega_z',
                                 'alpha_x', 'alpha_y', 'alpha_z'])

        df.to_csv('imu0.csv', index=False)

    else:

        df = rosbag_pandas.bag_to_dataframe(
            bag_name, include=include, exclude=exclude,
            parse_header=header)
        df = rosbag_pandas.clean_for_export(df)
        if output is None:
            base, _ = os.path.splitext(bag_name)
            output = base + '.csv'

        df.to_csv(output)

    return df


def bag_to_images(bag, image_topic, imu_topic, image_type, save_dir,
                  indexed=False, index_in_filename=False):
    """bag_to_image saves the images corresponding to the imu recordings in a
    specified directory.  Timestamps of the images is used as their name.

    Parameters:
    -----------
    bag: rosbag object

    image_topic: string, the specified topic where the images are published

    imu_topic: string, topic indicating where imu information is published

    image_type: string, (default='.tiff'), images extension

    save_dir: string, directory to save the images

    indexed: bool, (default=False), if True images are saved with
    frame_index_timestamp.ext

    index_in_filename: bool, (default=False), this option is only used if
    indexed is activated.  if True images are saved as `frame_index.ext`

    Returns:
    --------
    mean_time_shift: float, the mean timeshift between imu and images.

    """

    index_format = "04d"
    image_index = 0

    sel_imgs_array, sel_imgs_indexes, mean_time_shift = get_timeshift_imu_images(
        bag, image_topic, imu_topic, True)

    save_dir = os.path.join(sys.path[0], save_dir)

    # Use a CvBridge to convert ROS images to OpenCV images so they can be
    # saved.
    bridge = CvBridge()

    # Open bag file.
    # with rosbag.Bag(filename, 'r') as bag:
    # bag = rosbag.Bag(filename)
    for topic, msg, t in bag.read_messages(topics=[image_topic]):
        # first part is empty string
        try:
            # cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv_image = bridge.imgmsg_to_cv2(msg, "passthrough")
        except CvBridgeError, e:
            print e

        t = []
        t.append(genpy.rostime.Time.to_time(msg.header.stamp))
        #print t
        # print sel_imgs_array[sel_imgs_array == t]

        if sel_imgs_array[sel_imgs_array == t] :

            #timestr = "%.3f" % msg.header.stamp.to_sec()
            timestrnsec = "%d"   % msg.header.stamp.to_nsec()
            if indexed:
                if index_in_filename:
                    image_name = str(save_dir) + "/" + "frame" + "-" +
                    format(image_index, index_format) + "-" + timestr + image_type
                else:
                    image_name = str(save_dir) + "/" + "frame" + "-" +
                    format(image_index, index_format) + image_type
            else:
                image_name = str(save_dir) + "/" + timestrnsec + image_type

            print 'in process of saving {}'.format(image_name)

            cv2.imwrite(image_name, cv_image)
            image_index = image_index + 1

    return mean_time_shift



def write_yaml(model, intrinsics, distortion_model, distortion_coeffs,
               T_cam_imu, timeshift, N, M, outfilename=None):
    """writes the yaml file based on requirements of kalibr toolbox

    Parameters:
    -----------
    model: string, indicating model of the camera (pinhole/omni)

    intrinsics: list, intrinsic parameters of the camera

    distortion_model: string, radtan/equidistance

    distortion_coeffs: list of distortion parameters

    T_cam_imu: ndarray(4,4), rotation of imu in the camera frame, initial
    calculations

    timeshift: float, the meantimeshift between imu and images

    N: int, width of the image
    M: int, height of the image
    outfilename: default(None), name of the yaml file

    """

    import yaml
    data = dict(cam0 = dict(
        camera_model=model,
        intrinsics=intrinsics,
        distortion_model=distortion_model,
        distortion_coeffs=distortion_coeffs,
        T_cam_imu=T_cam_imu.tolist(),
        timeshift_cam_imu=timeshift.tolist(),
        rostopic='/cam0/image_raw',
        resolution=[N, M],
    ))
    if outfilename :
        with open(outfilename, 'w') as outfile:
            yaml.dump(data, outfile)

    else:
        with open('camchain.yaml', 'w') as outfile:
            yaml.dump(data, outfile)




def rename_images(path, origin_ext, save_ext, remove_str=None):
    """converts the images and saved a new normalized image with the new format.
    if requested remove parts of original names and save the images with their
    new names.

    Parameters:
    -----------
    path: string, location of original images

    origin_ext: string, the original extension of the images

    save_ext: string, the new extension requested by the user

    remove_str: string, the part of the original name requested to remove
    (default=None)

    """

    files = glob.glob( path +'*'+ origin_ext)

    # current_dir = os.getcwd()
    for i in files:
        filename, file_extension = os.path.splitext(i)
        if remove_str:
            path_to_save, new_filename = i.split(remove_str)
            print(path_to_save)
            print(new_filename)

            name, ext = new_filename.split(origin_ext)
            fullname = path_to_save + name
        else:
            fullname = filename
            print(fullname)

        img = cv2.imread(i)
        img = np.float32(img)
        print('Image {} and min  -> {} max -> {} of original image'.format(img.dtype, np.min(img), np.max(img)))_

        #img = img_as_uint(img)
        print('min  -> {} max -> {} of float image'.format(np.min(img), np.max(img)))
        cv2.imwrite(os.path.join(fullname + save_ext), img)


# #############
# # main_part #
# #############


# n = 0
# #bag_names = ['roll-pitch-yaw', 'random', 'LR-UD', 'LR2-UD2']
# #filename = bag_names[n] + '_aprilgridA3.bag'
# filename = 'RPY_ag_4A3.bag'

# image_topic = '/pleora_polarcam_driver/image_raw'
# imu_topic = '/imu_3dm_node/imu/data'
# save_dir = './imgs'


# # images should be saved in tiff format originally
# image_type = '.tiff'

# bag = rosbag.Bag(filename)


# ###################################
# # calculating R_ci of the camera  #
# ###################################
# df, output_name = bag_2_df(filename, imu_topic, None, None, False, True)
# df_mat = df.values

# imu_quaternion = df_mat[:, -4:]
# print imu_quaternion.shape

# imu_euler = quat2euler(imu_quaternion)

# Ric = euler2mat(*imu_euler[0])
# Rci = Ric.T

# T_cam_imu = np.zeros((4, 4))
# T_cam_imu[0:3, 0:3] = Rci
# T_cam_imu[3, 3] = 1

# print 'The first IMU recording on the camera indicate the rotation of imu when placed on the camera in i frame:\n {}'.format(Ric)

# print 'equivalenty the imu in camera frame will be Rci:\n {} '.format(Rci)

# sel_imgs_array, sel_imgs_index, timeshift = get_timeshift_imu_imgs(bag, image_topic, imu_topic)

# print 'timeshift between imu and image recordings:\n {}'.format(timeshift)

# model = 'omni'
# distortion_coeffs = [-0.0578, 0.0052, -0.00030508, 0.000072296]
# distortion_model= 'radtan'
# intrinsics =  [0.8731, 149.781, 149.6935, 160.0177, 115.0287]
# N = 320
# M = 260

# # model = 'pinhole'

# # distortion_coeffs =  [1.2828, 0.0304, 0.0166, -0.0138]
# # distortion_model = 'equidistant'
# # intrinsics =  [103.3344, 103.4761, 163.0093, 102.9141]



# write_yaml(model, intrinsics, distortion_model, distortion_coeffs, T_cam_imu, timeshift, N, M, 'camchain_I0_RPY_4A3_Mei.yaml')







# #########################################################
# # Saving images                                         #
# #    * calculating the timeshift between IMU and images #
# #########################################################

# #bag_2_images(bag, image_topic, imu_topic, '.tiff', save_dir, False, False)


# ################################################################################################
# # saving normalized png imahes instead of tiff and removing if necessary any part of the name  #
# ################################################################################################
# path = '/home/viper/Documents/Datasets/calibration/IMU-camera/aprilgrid-A3/RPY/kalibr_I0/'
# origin_ext = '.tiff'
# save_ext = '.png'
# remove_str = 'I0_'
# #rename_images(path, origin_ext, save_ext, remove_str=None)

# #rename_images(path, origin_ext, save_ext, remove_str)



# ################################
# # Saving the data to csv file  #
# ################################
# #df = bag_2_csv(filename, imu_topic, None, None, False, True, True)
