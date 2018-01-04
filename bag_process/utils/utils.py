#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
utils.py 
The utilities
@author: mrastgoo 
"""
import numpy as np 
import csv 
import os
import sys
import io
import random
import cv2 

from scipy.signal import convolve2d
from skimage.transform import resize 
from nibabel import eulerangles as ea
from nibabel import quaternions as qu
from scipy.io import loadmat

import matplotlib.pyplot as plt
import seaborn as sns


###################################################
# ------------ Conversion functions ------------- #
###################################################


def mat2euler(M):
    """Return Euler angles according to z, y, x
    intrinsic rotation according z, y, x
    M = Rz(gamma).Ry(beta).Rx(alpha)

    Parameters
    ----------
    M : 2D array

    Returns
    -------
    eul : list of Euler angles [gamma, beta, alpha] or [yaw, pitch, roll]
    """
    res = ea.mat2euler(M.T)
    return [-ang for ang in res]



def euler2mat(z=0, y=0, x=0):
    """Return matrix for intrinsic rotations around z, y, x
    Parameters
    ----------
    z : float
    y : float
    x : float
    
    Returns
    -------
    M the intrinsic rotation matrix

    Example
    -------
    >>> zrot = 1.3  # radians
    >>> yrot = -0.1
    >>> xrot = 0.2
    >>> M = euler2mat(zrot, yrot, xrot)
    >>> M1 = euler2mat(z=1.3)
    >>> M2 = euler2mat(y=-0.1)
    >>> M3 = euler2mat(x=0.2)
    >>> composed_M = dot(M3, dot(M2, M1))
    >>> allclose(M, composed_M)
    True
    """

    return np.dot(ea.euler2mat(z=z), np.dot(ea.euler2mat(y=y), ea.euler2mat(x=x)))


def quat2euler(quaternion_angles, show_angles=False, Rinit=np.eye(3)):
    euler_angles = np.zeros((quaternion_angles.shape[0], 3))


    for i, angle in enumerate(quaternion_angles):
        euler_angles[i, :] = mat2euler(np.dot(Rinit, qu.quat2mat(angle)))
    
    if show_angles:
        colors = ['r*-', 'g*-', 'b*-']
        plt.figure()
        for i, c  in enumerate(colors):
            plt.plot(euler_angles[:, i], c)
            plt.legend(['yaw', 'pitch', 'roll'])
            plt.title('The Euler angles from the csv file')
            sns.despine(trim=True)
        plt.show()

    return euler_angles



########################################
# --------- Loading function --------- #
########################################


def import_calib(path, model, **kwargs):
    """
    function that import calibration information from a matlab file
    
    Parameters
    ----------
    path : string , path to the file
    
    model : string 
    Three model of calibration is allowed scaramuzza, kannala, mei 

    f : the focal point of the camera in case kannala model is used 
    
    Returns
    -------
    K : the calibration structure
    if model == 'scaramuzza'
        K ['ss'] : the distortion coefficient 
        K ['A'] : the affine transformation 
        K ['t'] : the tanslation 
        K [height] : Image height
        K [width] : Image width
    if model == 'kannala'
        K ['mu'] : numbr of pixels in unit length in horizontal direction 
        K ['mv'] : number of pixels in unit length in vertical direction
        K ['u0'] : image center 
        K ['v0'] : image center 
        K ['ss'] : distrortion coefficents (5 parameters), 
        [k5, 0, k4, 0, k3, 0, k2, 0, k1]
    if model == 'mei'
        K['ss'] : distortion coefficients (5 parameters)   
    """
    K = {}
    if model == 'scaramuzza':
        ocamcalib = loadmat(path)['ocam_model'][0, 0]
        K['ss'] = ocamcalib[0].reshape((5, ))
        (xc, yc, c, d, e, K['width'], K['height']) = (ocamcalib[i][0, 0]
                                                      for i in range(1, 8))
        K['A'] = np.array([[c, d],
                           [e, 1]])
        K['t'] = np.array([xc, yc])

    elif model == 'kannala':
        P = loadmat(path)['p']
        P = P.ravel()     

        if 'f' in kwargs:
            K['f'] = kwargs.pop('f')
        else:
            K['f'] = P[0]

        if 'Size' in kwargs:
            Size = kwargs.pop('Size')
            K['height'] = Size[0]
            K['width'] = Size[1]

        else:
            if P[5]*2 > 320 :
                K['height'] = 640
                K['width'] = 460

        K['mu'] = P[2]
        K['mv'] = P[3]
        K['u0'] = P[4]
        K['v0'] = P[5]
        K['ss'] = np.array([P[8], 0.0, P[7], 0.0, P[6], 0.0, P[1], 0.0, P[0], 
                            0.0])

        dist_coef = np.zeros((4,))
        dist_coef[0] = P[0]
        dist_coef[1] = P[1]
        dist_coef[2] = P[6]
        dist_coef[3] = P[7]
        dist_coef = dist_coef.tolist()

        intrinsics = np.zeros((5,))
        intrinsics[0] = 1 
        intrinsics[1:3] = P[2:4] * K['f']
        intrinsics[3:] = P[4:6]
        intrinsics = intrinsics.tolist()

 
    elif model == 'mei':
        data = loadmat(path)
        K['ss'] = data['kc']
        K['u0'] = data['cc'][0]
        K['v0'] = data['cc'][1]
        K['pu'] = data['gammac'][0]
        K['pv'] = data['gammac'][1]
        K['xi'] = data['xi']
        K['width'] = data['roi_max'][0]
        K['height'] = data['roi_max'][1]        

        dist_coef = K['ss'].tolist()

        intrinsics = np.zeros((5,))
        intrinsics[0] = K['xi'] 
        intrinsics[1:3] = data['gammac']
        intrinsics[3:] = data['cc']
        intrinsics = intrinsics.tolist()

    else:
        raise ValueError(('The defined model {} is not known').format(model))

        
    return K, dist_coef, intrinsics

def csv2mat(file_path, istr, iend):

    ## --- opening the euler csv file and storing the values in a numpy array
    raw_data = []

    with open(file_path, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            raw_data.append(row[istr:iend])
    mat = np.array([[float(a) for a in b] for b in raw_data[1:]])

    return mat



