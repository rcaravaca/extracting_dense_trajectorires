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
from natsort import natsorted
import json, codecs
import pickle


def get_parser():
	parser = argparse.ArgumentParser(description="This script is wrapper to get dense trajectories from ./dense_trajectories/release/DenseTrack")
	
	parser.add_argument("-i","--input_dir", help="Directory where video secuence are")
	parser.add_argument("-o","--output_dir", help="Directory to save the resutls")

	return parser

def get_dt_from_csv(csv_file):

	f=open(csv_file,"r")

	lines=f.readlines()
	list_matrix = []

	for i in lines:
		list_matrix.append(i.split(sep='\t')[:-1])

	f.close()

	Matrix = np.array(list_matrix).astype(np.float)

	return Matrix

def clutering_trajestories(dt_matrix, temp_traject_list):

	rows, cols = dt_matrix.shape

	for dt in range(rows):

		mean_x, mean_y = dt_matrix[dt,1:3]

		trajectory = dt_matrix[dt,:]

		if mean_x <= 114 and mean_x >= 0 and mean_y <= 114 and mean_y >= 0:
			utils.msg("DT to cluster: "+str(0))
			temp_traject_list[0].append(trajectory)
		elif mean_x <= 228 and mean_x > 114 and mean_y <= 114 and mean_y >= 0:
			utils.msg("DT to cluster: "+str(1))
			temp_traject_list[1].append(trajectory)
		elif mean_x <= 340 and mean_x > 228 and mean_y <= 114 and mean_y >= 0:
			utils.msg("DT to cluster: "+str(2))
			temp_traject_list[2].append(trajectory)
		elif mean_x <= 114 and mean_x >= 0 and mean_y <= 228 and mean_y > 114:
			utils.msg("DT to cluster: "+str(3))
			temp_traject_list[3].append(trajectory)
		elif mean_x <= 228 and mean_x > 114 and mean_y <= 228 and mean_y > 114:
			utils.msg("DT to cluster: "+str(4))
			temp_traject_list[4].append(trajectory)
		elif mean_x <= 340 and mean_x > 228 and mean_y <= 228 and mean_y > 114:
			utils.msg("DT to cluster: "+str(5))
			temp_traject_list[5].append(trajectory)
		elif mean_x <= 114 and mean_x >= 0 and mean_y <= 340 and mean_y > 228:
			utils.msg("DT to cluster: "+str(6))
			temp_traject_list[6].append(trajectory)
		elif mean_x <= 228 and mean_x > 114 and mean_y <= 340 and mean_y > 228:
			utils.msg("DT to cluster: "+str(7))
			temp_traject_list[7].append(trajectory)
		elif mean_x <= 340 and mean_x > 228 and mean_y <= 340 and mean_y > 228:
			utils.msg("DT to cluster: "+str(8))
			temp_traject_list[8].append(trajectory)

def get_mean_of_trajectories(windows_trajectories):

	windows = windows_trajectories.keys()
	utils.msg("Count of keys of windows_trajectories: "+str(len(windows)))

	# Get means for all trajectorires for each window
	for window in windows:

		utils.msg("***********************************************************")
		utils.msg("For window: "+str(window))
		utils.msg("***********************************************************")

		clusters = windows_trajectories.get(window)

		for cluster in clusters.keys():

			stack_mtx = np.asarray(clusters.get(cluster))
			utils.msg("In cluster: "+str(cluster)+" there are "+str(stack_mtx.shape)+" DT.")

			if stack_mtx.shape[0] > 0:

				mean_of_mtx = stack_mtx.mean(axis=0)
				clusters.update({cluster: mean_of_mtx})

				aux_mtx = np.asarray(clusters.get(cluster))
				utils.msg("After there are "+str(aux_mtx.shape)+" DT.")


class NumpyArrayEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return JSONEncoder.default(self, obj)

def save_dt(file, dictionary):

	with open(file, "w") as outfile:
		json.dump(dictionary, outfile, cls=NumpyArrayEncoder)


# obj_text = codecs.open("save_dt.json", 'r', encoding = 'utf-8').read()
# b_new = json.loads(obj_text)


if __name__ == "__main__":

	#### LOG FILE
	utils.make_log_file("process_dt.log")

	#### Get parse arguments
	args = get_parser().parse_args()

	input_dir = args.input_dir
	days = os.listdir(input_dir)

	output_dir = args.output_dir
	utils.mkdir(output_dir)

	days = ["day_1"]

	# frames_checkpoints = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 450, 480, 510, 540, 570, 600, 630, 660, 690, 720, 750, 780, 810, 840, 870, 900, 930, 960, 990, 1020, 1050, 1080, 1110, 1140, 1170, 1200, 1230, 1260, 1290, 1320, 1350, 1380, 1410, 1440, 1470, 1500, 1530, 1560, 1590, 1620, 1650, 1680, 1710, 1740, 1770, 1800, 1830, 1860, 1890, 1920, 1950, 1980, 2010, 2040, 2070, 2100, 2130, 2160, 2190, 2220, 2250, 2280, 2310, 2340, 2370, 2400, 2430, 2460, 2490, 2520, 2550, 2580, 2610, 2640, 2670, 2700, 2730, 2760, 2790, 2820, 2850, 2880, 2910, 2940, 2970, 3000, 3030, 3060, 3090, 3120, 3150, 3180, 3210, 3240, 3270, 3300, 3330, 3360, 3390, 3420, 3450, 3480, 3510, 3540, 3570, 3600, 3630, 3660, 3690, 3720, 3750, 3780, 3810, 3840, 3870, 3900, 3930, 3960, 3990, 4020, 4050, 4080, 4110, 4140, 4170, 4200, 4230, 4260, 4290, 4320, 4350, 4380, 4410, 4440, 4470]
	
	windows_trajectories = {}
	for day in days:

		utils.msg("")
		utils.msg("")
		utils.msg("###########################################################")
		utils.msg("Working on day: "+str(day))
		utils.msg("###########################################################")
						

		current_day = os.path.join(input_dir, day)
		utils.mkdir(os.path.join(output_dir, day))

		for p in os.listdir(current_day):

			utils.msg("#----------------------------------------------------------")
			utils.msg("Working on day: "+str(day))
			utils.msg("#----------------------------------------------------------")
			utils.mkdir(os.path.join(output_dir, day, p))

			current_p = os.path.join(input_dir, day, p)

			for video in os.listdir(current_p):

				utils.mkdir(os.path.join(output_dir, day, p, video))

				video_part = os.path.join(current_p, video)
				# all_CSVs = natsorted(os.listdir(video_part))
				all_CSVs = range(4500)

				# frameNum, mean_x, mean_y, var_x, var_y, length, scale, x_pos, y_pos, t_pos, original_x, original_y, norm_trajectory_2x30 ,trajectory_2x30 , HOGx96, HOFx108, MBHxx96, MBHyx96
				window_start = 0
				temp_traject_list = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}

				h = 0
				for csv in all_CSVs:
					h += 1
					# current_frame = csv.replace('.csv','')		# csv number is the current frame number
					current_frame = csv		# csv number is the current frame number
					csv_path = os.path.join(video_part, str(csv)+".csv")

					utils.msg(">>>>>>>>>>>>>>>>>>>>>>>>>>>")
					utils.msg("Get DT Matrix for file: "+str(csv_path))
					utils.msg(">>>>>>>>>>>>>>>>>>>>>>>>>>>")

					if os.path.exists(csv_path):

						dt_matrix = get_dt_from_csv(csv_path)
						
						if int(current_frame) - 16 < window_start + 30:

							clutering_trajestories(dt_matrix, temp_traject_list)

						else:
							windows_trajectories.update({window_start: temp_traject_list})
							temp_traject_list = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
							clutering_trajestories(dt_matrix, temp_traject_list)
							window_start += 30
					else:
						utils.msg("File: "+str(csv_path)+" does not exist!", "W")
					
					if h == 121:
						break

				get_mean_of_trajectories(windows_trajectories)
				save_dt("save_dt.json", windows_trajectories)

				exit()

					# p_dir = os.path.join(output_dir, day, p)
					# utils.mkdir(p_dir)
					# p_videos = glob.glob(current_p + '/*.avi')