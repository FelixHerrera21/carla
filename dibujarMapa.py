#!/usr/bin/env python
import glob
import os
import sys
import matplotlib.pyplot as plt

#carpeta
try:
    sys.path.append(glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

#Bibliotecas
import carla
import random
import time
import numpy as np

import cv2
import math
import copy
#import tensorflow as tf
#import tensorflow.keras as kr
import logging
import argparse
import json

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

IM_WIDTH = 600
IM_HEIGHT = 400

vectorSensores = None
imagenGrabar = None

def init_carla():
	client = carla.Client('localhost', 2000)
	client.set_timeout(2.0)

	world = client.get_world()
	
	settings = world.get_settings()
	
	settings.no_rendering_mode = False
	world.apply_settings(settings)

	blueprint_library = world.get_blueprint_library()

	return [world, blueprint_library]

def handleArgs():
	parser = argparse.ArgumentParser(description='Ejemplo: programa.py -e rgb --camera 4')
	parser.add_argument('-e', '--entrada', help='entrada a utilizar', default = '')
	parser.add_argument('-wp', '--waypoints', help='waypoints', default = '')
	args = vars(parser.parse_args())
	return args

def dibujarWaypoints(world, archivoJson):
	f = open ((archivoJson),'r')
	mapa_json = f.read()
	f.close()
	waypoints_json = json.loads(mapa_json)

	for i in range(len(waypoints_json)):
		plt.plot(waypoints_json[i]['x'], waypoints_json[i]['y'], 'bo',  marker = 'o', color = 'b')
	plt.savefig("mapa2.png")

args = handleArgs()

[world, blueprint_library] = init_carla()

archivoJson = args['waypoints']

dibujarWaypoints(world, archivoJson)
