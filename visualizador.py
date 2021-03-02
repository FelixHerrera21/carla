#!/usr/bin/env python
import glob
import os
import sys

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
import tensorflow as tf
import tensorflow.keras as kr
import logging
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

IM_WIDTH = 600
IM_HEIGHT = 400

vectorSensores = None
imagenGrabar = None

def grabar_img(image):
	global imagenGrabar
	i = np.array(image.raw_data)
	i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
	i3 = i2[:, :, :3]
	imagenGrabar = i3

def filtrarSensores(matrizEntrada, rango1, rango2):
	x = matrizEntrada[:,0]
	y = matrizEntrada[:,1]

	#Conversion
	distanciaRadio = np.sqrt(x**2 + y**2)
	anguloRadianes = np.arctan2(y, x)
	anguloGrados = np.degrees(anguloRadianes)

	#condiciones
	mask = (anguloGrados>rango1) & (anguloGrados<rango2)

	#filtrados
	angulos = anguloRadianes[mask]
	distancias = distanciaRadio[mask]

	if(len(distancias>0)):
		distanciaMinima = np.amin(distancias)
	else:
		distanciaMinima = 15

	x = distancias * np.cos(angulos)
	y = distancias * np.sin(angulos)

	z = np.array([x, y])
	#return z.T, distanciaMinima
	return distanciaMinima


def process_img(image):
	global vectorSensores
	#separar
	points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))#vector 1D

	#reshape
	points = np.reshape(points, (int(points.shape[0] / 3), 3))#cambiar a matriz 2D
	
	points = points[points[:,2] < 1.7]#filtro suelo

	#tomar solo 2 dimensiones
	lidar_data = np.array(points[:, :2])#solo 2 dimen

	#Lectura de los sensores
	lidar1 = copy.deepcopy(lidar_data)
	lidar2 = copy.deepcopy(lidar_data)
	lidar3 = copy.deepcopy(lidar_data)

	izquierda = filtrarSensores(lidar3, -140, -100)
	
	frente = filtrarSensores(lidar2, -100, -80) #mi funcion (-180 a 0 es el frente, y de 0 a 180 es atras, creciente en sentido manecillas del reloj)
	
	derecha = filtrarSensores(lidar1, -80, -40)

	vectorSensores = [izquierda, frente, derecha]
	vectorSensores = np.array(vectorSensores)
	vectorSensores = vectorSensores/15

	#Dibujar mapa local:

	#normalizar
	lidar_data *= IM_HEIGHT / 100.0 #regla de 3 simple
	
	lidar_data += (0.5 * IM_HEIGHT, 0.5 * IM_HEIGHT) #centrar puntos en la imagen
	lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111

	
	lidar_data = lidar_data.astype(np.int32)
	lidar_data = np.reshape(lidar_data, (-1, 2))
	
	#tam imagen
	lidar_img_size = (IM_HEIGHT, IM_HEIGHT, 3)
	lidar_img = np.zeros((lidar_img_size), dtype = int)
	lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

	#convertir
	lidar_img = np.array(lidar_img, dtype=np.uint8)
	#invertir
	lidar_img = cv2.flip(lidar_img, 0)
	#rotar para mostrar de frente
	lidar_img = cv2.rotate(lidar_img, cv2.ROTATE_90_CLOCKWISE)
	#dibujar centro
	lidar_img = cv2.circle(lidar_img, (int(0.5 * IM_HEIGHT), int(0.5 * IM_HEIGHT)), 3, (0,0,255), -1)


def init_carla():
	client = carla.Client('localhost', 2000)
	client.set_timeout(2.0)

	world = client.get_world()
	settings = world.get_settings()

	weather = carla.WeatherParameters(
		cloudiness=30.0,
		wind_intensity = 10.0,
		sun_altitude_angle=70.0)

	world.set_weather(weather)

	#if args['render'] == "no":
	#	settings.no_rendering_mode = True
	#settings.no_rendering_mode = False
	world.apply_settings(settings)
	time.sleep(3)

	blueprint_library = world.get_blueprint_library()

	#########################      Plantillas   #####################################
	#plantilla auto
	return [world, blueprint_library]

def spawn_car(blueprint_library, spawn_Auto, world, actor_list):
	bp_Auto = blueprint_library.filter('model3')[0]
	bandera = True
	#Auto
	while bandera:
		try: 
			#spawn_Auto = random.choice(world.get_map().get_spawn_points())
			#spawn_Auto = ubicacion_spawn
			#print(spawn_Auto.location, spawn_Auto.rotation)

			#ubicacion tunel
			#spawn_Auto.location.x = 245 #100
			#spawn_Auto.location.y = -40 #6
			#spawn_Auto.location.z = 0.3
			#spawn_Auto.rotation.pitch = 0
			#spawn_Auto.rotation.yaw = -91
			#spawn_Auto.rotation.roll = 0

			vehicle = world.spawn_actor(bp_Auto, spawn_Auto)

			"""#vehiculo obstaculo
			spawn_Auto.location.x = 244
			spawn_Auto.rotation.yaw = -88
			spawn_Auto.location.y = -65 #6
			vehicleObstacle = world.spawn_actor(bp_Auto, spawn_Auto)
			"""
			bandera = False
		except:
			spawn_Auto = random.choice(world.get_map().get_spawn_points())
			bandera = True

	actor_list.append(vehicle)

	return vehicle

def spawn_lidar(blueprint_library, world, vehicle, actor_list):

	bp_Lidar = blueprint_library.find('sensor.lidar.ray_cast')
	bp_Lidar.set_attribute('range', '15')
	bp_Lidar.set_attribute('sensor_tick', '0.05')
	bp_Lidar.set_attribute('rotation_frequency','30')
	bp_Lidar.set_attribute('upper_fov','0.0')
	bp_Lidar.set_attribute('lower_fov','-8')
	bp_Lidar.set_attribute('channels', '3')
	spawn_Lidar = carla.Transform(carla.Location(z=2))
	lidar = world.spawn_actor(bp_Lidar, spawn_Lidar, attach_to=vehicle)
	actor_list.append(lidar)
	lidar.listen(lambda data: process_img(data))
	return lidar

def spawn_colision(blueprint_library, world, vehicle, actor_list):
	bp_Collision = blueprint_library.find('sensor.other.collision')
	spawn_Collision = carla.Transform(carla.Location(x=0, y=0, z=0))
	sensor_Collision = world.spawn_actor(bp_Collision, spawn_Collision, attach_to=vehicle)
	actor_list.append(sensor_Collision)
	return sensor_Collision

def spawn_camera(blueprint_library, world, vehicle, actor_list):
	bp_Grabar = blueprint_library.find('sensor.camera.rgb')
	bp_Grabar.set_attribute('image_size_x', f'{IM_WIDTH}')
	bp_Grabar.set_attribute('image_size_y', f'{IM_HEIGHT}')
	bp_Grabar.set_attribute('fov', '110')
	bp_Grabar.set_attribute('sensor_tick', '0.05')

	spawn_Grabar = carla.Transform(carla.Location(x=-5, z=3), carla.Rotation(pitch=-20))
	sensor_Grabar = world.spawn_actor(bp_Grabar, spawn_Grabar, attach_to=vehicle)
	actor_list.append(sensor_Grabar)
	sensor_Grabar.listen(lambda data: grabar_img(data))
	return sensor_Grabar

#funcion aceleracion y direccion
def funcion1(model, world, vehicle, actor_list):
	global vectorSensores, imagenGrabar
	while (True):
		arregloEntrada = np.array([vectorSensores])
		direccion, velocidad = model.predict(arregloEntrada)[0]
		direccion = direccion * 2 - 1

		v = vehicle.get_velocity()
		kmh = 3.6 * math.sqrt(v.x **2 + v.y**2 + v.z**2)

		if(kmh > 30):
			velocidad = 0
		
		#vehicle.apply_control(carla.VehicleControl(throttle=float(velocidad), steer=float(direccion)))
		vehicle.set_autopilot(True)
		
		#if(len(collision_hist)>0):
		#	break
		if(imagenGrabar is not None):
			cv2.imshow("camara", imagenGrabar)
		#salidaVideo.write(imagenGrabar)
		tecla = cv2.waitKey(1) 
		if (tecla & 0xFF) == ord("q"):
			cv2.destroyAllWindows()
			break

	for actor in actor_list:
	    actor.destroy()

	


def handleArgs():
	parser = argparse.ArgumentParser(description='Ejemplo: programa.py -e rgb --camera 4')
	parser.add_argument('-e', '--entrada', help='entrada a utilizar', default = '')
	args = vars(parser.parse_args())
	return args

args = handleArgs()

modelo = kr.models.load_model(args['entrada'])

[world, blueprint_library] = init_carla()

map1 = world.get_map()

waypoints = map1.generate_waypoints(2.0)

print(waypoints[:5])

actor_list = []
vehicle = spawn_car(blueprint_library, random.choice(world.get_map().get_spawn_points()), world, actor_list)

lidar = spawn_lidar( blueprint_library, world, vehicle, actor_list)

collision = spawn_colision(blueprint_library, world, vehicle, actor_list)

camera = spawn_camera(blueprint_library, world, vehicle, actor_list)

funcion1(modelo, world, vehicle, actor_list)