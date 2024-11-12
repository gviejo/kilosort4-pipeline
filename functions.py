# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2024-11-08 15:27:57
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-11-08 15:28:06
import numpy as np
import os, sys
import pathlib
import shutil
import subprocess
import re

def loadXML(path):
	"""
	path should be the folder session containing the XML file
	Function returns :
		1. the number of channels
		2. the sampling frequency of the dat file or the eeg file depending of what is present in the folder
			eeg file first if both are present or both are absent
		3. the mappings shanks to channels as a dict
	Args:
		path : string
	Returns:
		int, int, dict
	"""
	if not os.path.exists(path):
		print("The path "+path+" doesn't exist; Exiting ...")
		sys.exit()
	listdir = os.listdir(path)
	xmlfiles = [f for f in listdir if f.endswith('.xml')]
	if not len(xmlfiles):
		print("Folder contains no xml files; Exiting ...")
		sys.exit()
	new_path = os.path.join(path, xmlfiles[0])
	
	from xml.dom import minidom	
	xmldoc 		= minidom.parse(new_path)
	nChannels 	= xmldoc.getElementsByTagName('acquisitionSystem')[0].getElementsByTagName('nChannels')[0].firstChild.data
	fs_dat 		= xmldoc.getElementsByTagName('acquisitionSystem')[0].getElementsByTagName('samplingRate')[0].firstChild.data
	fs_eeg 		= xmldoc.getElementsByTagName('fieldPotentials')[0].getElementsByTagName('lfpSamplingRate')[0].firstChild.data	
	if os.path.splitext(xmlfiles[0])[0] +'.dat' in listdir:
		fs = fs_dat
	elif os.path.splitext(xmlfiles[0])[0] +'.eeg' in listdir:
		fs = fs_eeg
	else:
		fs = fs_eeg
	shank_to_channel = {}
	shank_to_keep = {}	
	groups 		= xmldoc.getElementsByTagName('anatomicalDescription')[0].getElementsByTagName('channelGroups')[0].getElementsByTagName('group')
	for i in range(len(groups)):
		shank_to_channel[i] = []
		shank_to_keep[i] = []
		for child in groups[i].getElementsByTagName('channel'):
			shank_to_channel[i].append(int(child.firstChild.data))
			tmp = child.toprettyxml()
			shank_to_keep[i].append(int(tmp[15]))

		#shank_to_channel[i] = np.array([int(child.firstChild.data) for child in groups[i].getElementsByTagName('channel')])
		shank_to_channel[i] = np.array(shank_to_channel[i])
		shank_to_keep[i] = np.array(shank_to_keep[i])
		shank_to_keep[i] = shank_to_keep[i]==0 #ugly
	return int(nChannels), int(fs), shank_to_channel, shank_to_keep