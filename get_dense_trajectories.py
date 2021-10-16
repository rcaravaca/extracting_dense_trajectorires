#!/usr/bin/env python3

"""
Created on 08/17/2021

@author: Ronald
"""


import numpy as np
import sys
import cv2 
import os
import glob
import subprocess
import argparse
import utils


def get_parser():
	parser = argparse.ArgumentParser(description="This script is wrapper to get dense trajectories from ./dense_trajectories/release/DenseTrack")
	
	parser.add_argument("-i","--input_dir", help="Directory where video secuence are")
	parser.add_argument("-o","--output_dir", help="Directory to save the resutls")

	return parser


def get_dense_trajectories(video, out_dir, L):

	# video = seq_dir+'subject'+str(subject_global)+'.mp4'
	subprocess.call(['./dense_trajectories/release/DenseTrack', video, out_dir, '-L', str(L)])


if __name__ == "__main__":

	#### LOG FILE
	utils.make_log_file("get_dense_trajectories.log")

	#### Get parse arguments
	args = get_parser().parse_args()

	input_dir = args.input_dir
	days = os.listdir(input_dir)


	output_dir = args.output_dir
	utils.mkdir(output_dir)

	days = ["day_1"]
	for day in days:

		current_day = os.path.join(input_dir, day)
		utils.mkdir(os.path.join(output_dir, day))

		for p in os.listdir(current_day):

			current_p = os.path.join(input_dir, day, p)
			

			if os.path.isdir(current_p):

				p_dir = os.path.join(output_dir, day, p)
				utils.mkdir(p_dir)
				p_videos = glob.glob(current_p + '/*.avi')
				
				for video in p_videos:
					
					video_name = os.path.splitext(os.path.basename(video))[0]

					video_trajectory_dir = os.path.join(p_dir, video_name)

					if os.path.exists(video_trajectory_dir):
						utils.msg("Dense trajectories already exist for video: "+str(video_trajectory_dir))
					else:
						utils.mkdir(video_trajectory_dir)
						utils.msg("Running 'get_dense_trajectories' for video : " + str(video_name))
						get_dense_trajectories(video, video_trajectory_dir, 30)
						# exit()


