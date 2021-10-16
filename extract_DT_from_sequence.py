# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 13:43:56 2017

@author: laura
"""


import numpy as np
import sys
import cv2 
import os
import subprocess
import shutil


if len(sys.argv) < 1:
    sys.exit('Usage: extract_DT_global subject_global')

subject_global=int(sys.argv[1]) 


#map global id to day and subjet per day

##extract dense trajectories
#THIS NEEDS TO BE CHANGED TO FIT USER
traj_dir='/*MYDIR*/changed_dense_trajectory_release/'
seq_dir='/*MYDIR*/sequences/original/'
out_dir_dt='/*MYDIR*/DT/subject'+str(subject_global)+'/'

print('Extracting trajectories, to be saved at '+out_dir_dt)

if not os.path.exists(out_dir_dt):
    os.makedirs(out_dir_dt)

os.chdir(traj_dir)
subprocess.call(['./release/DenseTrack', seq_dir+'subject'+str(subject_global)+'.mp4',out_dir_dt,'-L','20'])
