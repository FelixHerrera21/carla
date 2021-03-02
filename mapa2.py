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
	
	settings.no_rendering_mode = True
	world.apply_settings(settings)

	blueprint_library = world.get_blueprint_library()

	return [world, blueprint_library]

def handleArgs():
	parser = argparse.ArgumentParser(description='Ejemplo: programa.py -e rgb --camera 4')
	parser.add_argument('-e', '--entrada', help='entrada a utilizar', default = '')
	args = vars(parser.parse_args())
	return args

def obtenerWaypoints(world):
	map1 = world.get_map()
	waypoints = map1.generate_waypoints(20)

	print("Length: " + str(len(waypoints)))

	x = [p.transform.location.x for p in waypoints]
	y = [p.transform.location.y for p in waypoints]

	plt.plot(x, y, 'bo',  marker = 'o', color = 'b')
	ruta = []
	#wp_prueba = waypoints[90]
	spawn_wp = random.choice(world.get_map().get_spawn_points())

	wp_prueba = map1.get_waypoint(carla.Location(x = spawn_wp.location.x, y = spawn_wp.location.y, z = spawn_wp.location.z))

	ruta.append({'x': spawn_wp.location.x,'y': spawn_wp.location.y, 'z': spawn_wp.location.z, 'pitch':spawn_wp.rotation.pitch, 'yaw':spawn_wp.rotation.yaw, 'roll':spawn_wp.rotation.roll})

	plt.plot([wp_prueba.transform.location.x], [wp_prueba.transform.location.y], 'bo',  marker = 'o', color = 'r')

	next_wp = wp_prueba.next(5)

	plt.plot([next_wp[0].transform.location.x], [-next_wp[0].transform.location.y], 'bo',  marker = 'o', color = 'g')

	

	for i in range(1,100):
		next_wp = next_wp[-1].next(5)
		x = next_wp[-1].transform.location.x
		y = next_wp[-1].transform.location.y
		z = next_wp[-1].transform.location.z
		pitch = next_wp[-1].transform.rotation.pitch
		yaw = next_wp[-1].transform.rotation.yaw
		roll = next_wp[-1].transform.rotation.roll

		plt.plot([x], [-y], 'bo',  marker = 'o', color = 'g')
		ruta.append({'x': x,'y': y, 'z': z, 'pitch':pitch, 'yaw':yaw, 'roll':roll})

	mapa_json = json.dumps(ruta)
	print(type(mapa_json))
	plt.savefig("mapa.png")

	return mapa_json

args = handleArgs()

[world, blueprint_library] = init_carla()

f = open (('ruta'+ '.json'),'w')
mapa_json = obtenerWaypoints(world)
f.write(mapa_json)
f.close()