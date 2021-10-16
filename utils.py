"""
Created on 08/30/2021

@author: Ronald
"""


import numpy as np
import sys
import cv2 
import os
import glob
import subprocess
import argparse
import logging

def make_log_file(file_name):

	if os.path.exists(file_name):
		msg("Removing existing file: "+str(file_name))
		os.remove(file_name)
		
	logging.basicConfig(filename=file_name, encoding='utf-8', level=logging.DEBUG)

def msg(msg, tp=None):

	if tp == None or tp == "I":
		print("UTILS_INFO==> " + msg )
		logging.info(msg)
	elif tp == "E":
		print("UTILS_ERROR==> " + msg )
		logging.error(msg)
	elif tp == "W":
		print("UTILS_WARNING==> " + msg )
		logging.debug(msg)

def mkdir(directory):
	if not os.path.exists(directory):
		msg("Creating dir: " +str(directory), "I")
		os.makedirs(directory)
	else:
		msg("dir: " +str(directory) + " already exists.", "W")
