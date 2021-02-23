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

def handleArgs():
	parser = argparse.ArgumentParser(description='Ejemplo: programa.py -e rgb --camera 4')
	parser.add_argument('-e', '--entrada', help='entrada a utilizar', default = 'lidar')
	parser.add_argument('-r', '--render', help='renderizar', default = 'si')

	parser.add_argument('-i', '--iter', help='iteraciones', default = '2')
	parser.add_argument('-p', '--pob', help='poblacion', default = '5')
	parser.add_argument('-g', '--gen', help='generaciones', default = '5')

	parser.add_argument('-c', '--car', help='carros', default = '10')
	#parser.add_argument('-g', '--gen', help='generaciones', default = '40')

	args = vars(parser.parse_args())

	return args

args = handleArgs()

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

def grabar_img(image):
	global imagenGrabar
	i = np.array(image.raw_data)
	i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
	i3 = i2[:, :, :3]
	imagenGrabar = i3

def process_img(image):
	global frame, args, vectorSensores

	if args['entrada'] == "lidar":
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
		i3 = lidar_img
		frame = i3

#Detector Colisiones
def collision_data(event):
	global collision_hist
	collision_hist.append(event)

#Detector invadir lineas
def on_invasion(event):
	global line_hist
	lane_types = set(x.type for x in event.crossed_lane_markings)
	text = ['%r' % str(x).split()[-1] for x in lane_types]
	#print(text[0])
	if(text[0] == "\'Solid\'"):
		line_hist.append(event)

##########################     Conexion     #################################################

client = carla.Client('localhost', 2000)
client.set_timeout(2.0)

world = client.get_world()
settings = world.get_settings()

weather = carla.WeatherParameters(
	cloudiness=30.0,
	wind_intensity = 10.0,
	sun_altitude_angle=70.0)

world.set_weather(weather)

if args['render'] == "no":
	settings.no_rendering_mode = True
#settings.no_rendering_mode = False
world.apply_settings(settings)
time.sleep(3)

blueprint_library = world.get_blueprint_library()

#########################      Plantillas   #####################################
#plantilla auto
bp_Auto = blueprint_library.filter('model3')[0]

#### Camara grabar
bp_Grabar = blueprint_library.find('sensor.camera.rgb')
bp_Grabar.set_attribute('image_size_x', f'{IM_WIDTH}')
bp_Grabar.set_attribute('image_size_y', f'{IM_HEIGHT}')
bp_Grabar.set_attribute('fov', '110')
bp_Grabar.set_attribute('sensor_tick', '0.05')

######################

#plantilla entrada
if args['entrada'] == "rgb":
	bp_Lidar = blueprint_library.find('sensor.camera.rgb')
	bp_Lidar.set_attribute('image_size_x', f'{IM_HEIGHT}')
	bp_Lidar.set_attribute('image_size_y', f'{IM_HEIGHT}')
	bp_Lidar.set_attribute('fov', '110')
	bp_Lidar.set_attribute('sensor_tick', '0.05')

if args['entrada'] == "lidar":
	bp_Lidar = blueprint_library.find('sensor.lidar.ray_cast')
	bp_Lidar.set_attribute('range', '15')
	bp_Lidar.set_attribute('sensor_tick', '0.05')
	bp_Lidar.set_attribute('rotation_frequency','30')
	bp_Lidar.set_attribute('upper_fov','0.0')
	bp_Lidar.set_attribute('lower_fov','-8')
	bp_Lidar.set_attribute('channels', '3')

bp_Collision = blueprint_library.find('sensor.other.collision')

bp_line_invasion = blueprint_library.find('sensor.other.lane_invasion')

####################### Red nn #################################################

nn = [3, 7, 2]  # número de neuronas por capa

# secuencia de capas
model = kr.Sequential()

# Añadimos la capa 1
l1 = model.add(kr.layers.Dense(nn[1], input_dim=nn[0], activation='sigmoid'))

# Añadimos la capa 2
l3 = model.add(kr.layers.Dense(nn[2], activation='sigmoid'))

# Compilamos el modelo
model.compile(loss='mse', optimizer=kr.optimizers.SGD(lr=0.05), metrics=['acc'])

#################################################################################

#Cargar ventanas 
cv2.namedWindow('sensor lidar', cv2.WINDOW_AUTOSIZE)

if args['render'] == "si":
	cv2.namedWindow('camara', cv2.WINDOW_AUTOSIZE)

cv2.moveWindow("sensor lidar", IM_HEIGHT+150, IM_HEIGHT+150)
cv2.moveWindow("camara", 0, 0)

##################################################################################

#inicializaciones obligatorias
frame = np.zeros((IM_HEIGHT, IM_HEIGHT), np.uint8)
imagenGrabar = np.zeros((IM_HEIGHT, IM_HEIGHT), np.uint8)

vectorSensores = np.array([1, 1, 1])
collision_hist = []
line_hist = []

#salidaVideo = cv2.VideoWriter('videoSalida.avi',cv2.VideoWriter_fourcc(*'XVID'),20.0,(IM_WIDTH,IM_HEIGHT))

def funcion_Aptitud_Recta(vector, ubicacion_spawn):
	global bp_Auto, bp_Lidar, bp_Collision, frame, collision_hist, model, nn, world, line_hist, bp_line_invasion, vectorSensores, imagenGrabar, bp_Grabar#, salidaVideo

	###########################      Spawns   ###########################################
	vector = np.array(vector)
	actor_list = []
	bandera = True
	#Auto
	while bandera:
		try: 
			#spawn_Auto = random.choice(world.get_map().get_spawn_points())
			spawn_Auto = ubicacion_spawn
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
			bandera = True

	actor_list.append(vehicle)
	############################################################################################

	#'vehicle.*'
	#blueprintsWalkers = world.get_blueprint_library().filter('walker.pedestrian.*')
	#walker_bp = random.choice(blueprintsWalkers)
	#SpawnActor(walker_bp, spawn_point)

	############################################################################################


	#actor_list.append(vehicleObstacle)
	#vehicleObstacle.apply_control(carla.VehicleControl(throttle=float(1), steer=float(0)))
	#vehicleObstacle.set_autopilot(True)
	#Entrada
	if args['entrada'] == "rgb":
		spawn_Lidar = carla.Transform(carla.Location(x=2.5, z=2), carla.Rotation(pitch=-30))
	if args['entrada'] == "lidar":
		spawn_Lidar = carla.Transform(carla.Location(z=2))
	lidar = world.spawn_actor(bp_Lidar, spawn_Lidar, attach_to=vehicle)
	actor_list.append(lidar)

	#grabar
	spawn_Grabar = carla.Transform(carla.Location(x=-5, z=3), carla.Rotation(pitch=-20))
	sensor_Grabar = world.spawn_actor(bp_Grabar, spawn_Grabar, attach_to=vehicle)
	actor_list.append(sensor_Grabar)

	#collision
	spawn_Collision = carla.Transform(carla.Location(x=0, y=0, z=0))
	sensor_Collision = world.spawn_actor(bp_Collision, spawn_Collision, attach_to=vehicle)
	actor_list.append(sensor_Collision)

	sensor_line_invasion = world.spawn_actor(bp_line_invasion, carla.Transform(), attach_to=vehicle)
	actor_list.append(sensor_line_invasion)
	
	############################  Ejecucion   #########################################  

	frame = np.zeros((IM_HEIGHT, IM_HEIGHT), np.uint8)
	imagenGrabar = np.zeros((IM_WIDTH, IM_HEIGHT), np.uint8)

	collision_hist = []
	line_hist = []
	#llamada lidar
	lidar.listen(lambda data: process_img(data))

	if args['render'] == "si":
		sensor_Grabar.listen(lambda data: grabar_img(data))

	#llamada collision
	sensor_Collision.listen(lambda event: collision_data(event))

	sensor_line_invasion.listen(lambda event2: on_invasion(event2))
	############################################################################
	pesosSeparados = []
	separador = 0
	for i in range(0,len(nn)-1):

		pesosSeparados.append(vector[separador:separador + nn[i]*nn[i+1]])
		separador = separador + nn[i]*nn[i+1]

		pesosSeparados.append(vector[separador:separador + nn[i+1]])
		separador = separador + nn[i+1]

	j = 0
	for i in range(0, len(pesosSeparados), 2):
		pesosSeparados[i] = pesosSeparados[i].reshape((nn[j], nn[j+1]))
		j = j+1

	model.set_weights(pesosSeparados)

	velocidad = 0
	direccion = 0

	freno = 0
	acelerador = 0

	time.sleep(2)
	
	pos_Inicial = vehicle.get_transform()
	pos_Inicial = np.array((pos_Inicial.location.x, pos_Inicial.location.y))

	last_Point = pos_Inicial

	d0 = 0
	fx = 0
	con=0

	while (True):
		pos_Actual = vehicle.get_transform()
		pos_Actual = np.array((pos_Actual.location.x, pos_Actual.location.y))

		arregloEntrada = np.array([vectorSensores])
		direccion, velocidad = model.predict(arregloEntrada)[0]
		direccion = direccion * 2 - 1

		v = vehicle.get_velocity()
		kmh = 3.6 * math.sqrt(v.x **2 + v.y**2 + v.z**2)

		if(vectorSensores[1] > 0.8) and velocidad < 0.16 and kmh < 4:
			fx = fx - 100
			break

		if(kmh > 30):
			velocidad = 0
		
		vehicle.apply_control(carla.VehicleControl(throttle=float(velocidad), steer=float(direccion)))
		#vehicleObstacle.set_autopilot(True)
		
		if(len(collision_hist)>0):
			break

		if(len(line_hist)>0):
			line_hist = []
			fx = fx - 3

		d1 = np.linalg.norm(pos_Actual - last_Point)
		
		if(kmh < 10):
			fx = fx - 1

		if(d1 < d0):
			fx = fx - 5
		else:
			fx = fx + 1

		d0=d1

		#Mostrar datos en tiempo real
		font = cv2.FONT_HERSHEY_SIMPLEX
		texto = ' throttle: ' + str("%.4f" % velocidad)
		cv2.putText(frame,texto,(0,25), font, 0.8,(255,255,255),1,cv2.LINE_AA)
		texto = ' steer:' + str("%.4f" % direccion)
		cv2.putText(frame,texto,(0,50), font, 0.8,(255,255,255),1,cv2.LINE_AA)
		texto = ' score:' + str(fx)
		cv2.putText(frame,texto,(0,75), font, 0.8,(255,255,255),1,cv2.LINE_AA)
		texto = ' kmh:' + str("%.3f" % kmh)
		cv2.putText(frame,texto,(0,100), font, 0.8,(255,255,255),1,cv2.LINE_AA)
		
		cv2.imshow("sensor lidar", frame)
		
		if args['render'] == "si":
			cv2.imshow("camara", imagenGrabar)
			#salidaVideo.write(imagenGrabar)
		tecla = cv2.waitKey(1)

		if(fx < -100):
			break

	for actor in actor_list:
	    actor.destroy()

	return fx
###########################################################################################	

class Particula:
	def __init__(self, cantidadVariables, minimo, maximo, sp_points):
		self.particula_pos = []
		self.particula_velocidad = []
		self.particula_aptitud = 0
		self.particula_pbest = None
		self.particula_pbest_pos = []
		self.sp_points = sp_points
		for i in range(cantidadVariables):
			self.particula_pos.append(random.uniform(minimo,maximo)) 
			self.particula_velocidad.append(random.uniform(-1,1))

		for x in range(len(self.sp_points)):
			self.particula_aptitud = self.particula_aptitud + funcion_Aptitud_Recta(self.particula_pos, self.sp_points[x])
		#self.particula_aptitud = self.particula_aptitud + funcion_Aptitud_Recta(self.particula_pos)
		self.particula_aptitud = self.particula_aptitud/len(self.sp_points)
		print(self.particula_aptitud)
		self.particula_pbest=self.particula_aptitud
		self.particula_pbest_pos=self.particula_pos

	def evaluar(self):
		self.particula_aptitud = 0
		for x in range(len(self.sp_points)):
			self.particula_aptitud = self.particula_aptitud + funcion_Aptitud_Recta(self.particula_pos, self.sp_points[x])
		#self.particula_aptitud=funcion_Aptitud_Recta(self.particula_pos)
		#print("segunda")
		#self.particula_aptitud=self.particula_aptitud + funcion_Aptitud_Recta(self.particula_pos)
		self.particula_aptitud = self.particula_aptitud/len(self.sp_points)
		print(self.particula_aptitud)

		if (self.particula_aptitud > self.particula_pbest):
			self.particula_pbest=self.particula_aptitud
			self.particula_pbest_pos= self.particula_pos

	def actualizar_pos(self, minimo, maximo):
		for i in range(cantidadVariables):
			self.particula_pos[i]=self.particula_pos[i]+self.particula_velocidad[i]

			if self.particula_pos[i]>maximo:
				self.particula_pos[i]=maximo

			if self.particula_pos[i]<minimo:
				self.particula_pos[i]=minimo
	
	def actualizar_vel(self, gbest_pos):
		for i in range(cantidadVariables):
			r1=random.random()
			r2=random.random()
			vel_individual = c1*r1*(self.particula_pbest_pos[i] - self.particula_pos[i])
			vel_social = c2*r2*(gbest_pos[i] - self.particula_pos[i])
			self.particula_velocidad[i] = w*self.particula_velocidad[i] + vel_individual + vel_social

	def imprimirDatos(self):
		print(self.particula_pos)
		print(self.particula_aptitud)
		print(self.particula_velocidad)

#####################################################################################################

relleno_lista = []
for vehiculosRelleno in range(int(args['car'])):
	relleno_bp = random.choice(blueprint_library.filter('vehicle.*'))
	spawn_relleno = random.choice(world.get_map().get_spawn_points())
	vehiculo_relleno = world.spawn_actor(relleno_bp, spawn_relleno)
	vehiculo_relleno.set_autopilot(True)
	relleno_lista.append(vehiculo_relleno)

puntos_de_spawn =[]
for iter_aptitud in range(int(args['iter'])):
	puntos_de_spawn.append(random.choice(world.get_map().get_spawn_points()))

puntos_de_spawn[0].location.x = 245 #100
puntos_de_spawn[0].location.y = -40 #6
puntos_de_spawn[0].location.z = 0.3
puntos_de_spawn[0].rotation.pitch = 0
puntos_de_spawn[0].rotation.yaw = -91
puntos_de_spawn[0].rotation.roll = 0

print("puntos spawn: ")
print(puntos_de_spawn)
#####################################################################################################

identificadorPrueba = '_' + args['iter'] + '_' + args['pob'] +'_'+ args['gen']+'_'+ args['car']

numeroPrueba = 1

target_dir = './pruebas/prueba_' + str(numeroPrueba) + "_id"+ identificadorPrueba + '/'

while os.path.exists(target_dir):
	numeroPrueba = numeroPrueba +1
	target_dir = './pruebas/prueba_' + str(numeroPrueba) + "_id"+ identificadorPrueba + '/'

os.mkdir(target_dir)

minimo = -3
maximo = 3

#############################################################################################

cantidadParticulas = int(args['pob'])
generaciones = int(args['gen'])
cantidadVariables = 0

for i in range(0,len(nn)-1):
	cantidadVariables = cantidadVariables + nn[i]*nn[i+1] + nn[i+1]
print("cantidad variables a optimizar: ", cantidadVariables)

w = 0.3
c1 = 0.3
c2 = 0.7

################################################################################################
f = open ((target_dir + 'generacion: ' + str(0) + '.txt'),'w')

print("Poblacion Inicial")
enjambre = []
for i in range(cantidadParticulas):
	print("\t individuo: ", i+1, " calif: ", end="")
	enjambre.append(Particula(cantidadVariables, minimo, maximo, puntos_de_spawn))
	if(i==0):
		gbest = enjambre[0].particula_aptitud
		gbest_pos = enjambre[0].particula_pos
	else:
		if(enjambre[i].particula_aptitud > gbest):
			gbest = enjambre[i].particula_aptitud
			gbest_pos = enjambre[i].particula_pos
			model.save(target_dir + 'gen_' + str(0) + '_red_' + identificadorPrueba + '.h5')
	f.write(str(enjambre[i].particula_aptitud) + "\n")
print("\tmejor generacion", gbest)
f.close()
############################################################################################
#comienza ciclo
for g in range(generaciones):
	print("Generacion: ", g+1)
	#f = open (('hola' + '.txt'),'w')
	f = open ((target_dir + 'generacion: ' + str(g+1) + '.txt'),'w')

	for i in range(cantidadParticulas):
		print("\t individuo: ", i+1, " calif: ", end="")
		enjambre[i].evaluar()
		if enjambre[i].particula_aptitud > gbest:
			gbest = enjambre[i].particula_aptitud
			gbest_pos = enjambre[i].particula_pos
			model.save(target_dir + 'gen_' + str(g) + '_red_' + identificadorPrueba + '.h5')
			#f2 = open ((target_dir + 'gen:_' + str(g) + '_red_' + identificadorPrueba +'.txt'),'w')
			#f2.write(str(gbest_pos))
			#f2.close()
		f.write(str(enjambre[i].particula_aptitud) + "\n")
		
	for i in range(cantidadParticulas):
		enjambre[i].actualizar_vel(gbest_pos)
		enjambre[i].actualizar_pos(minimo, maximo)
	print("\tmejor generacion", gbest)
	f.close()

print("\nmejor final")
print(gbest)

#############################################################################################

#gbest_pos = [-1.3678269181074858, -1.6485398139559544, 0.9456611926068601, -0.9601122784830124, -1.3003309500122735, -1.751250334549164, 0.2566273365499324, -1.6723611235926146, -1.0394542544369563, 2.270266082820752, -0.47701116395626075, 2.2632167881858845, -1.6154864234228485, -1.8322161552500744, -0.8140422064131451, 0.7951039633226861, 1.4656231411831737, -2.587421879657094, 1.8898053233270067, -0.358324863379295, -1.5407267079550537, -0.18985739312944466, -0.6341347783026146, 0.04999568811450844, -1.560706647467091, 1.248847913410546, 0.6116362580143901, -1.1054287550764659, 0.16735692433253835, -0.5461494406258491, -1.9985925014562302, -1.6718420475953544, -0.8623390071529567, -0.9514414020874804, 1.8689558005192994, 0.6741040502908736, 0.3277390341565009, -0.8905061856889659, 2.3819636257144134, 0.18629102170215245, 0.032669769901775975, -0.5974687592019717, 1.0838839178454873, 0.08762433877530827, 0.4956167000327521, -2.5374541902334573, -2.150524242199497, 1.005486765599056, -1.1089252772854206, 1.4054664596897806, 1.4983115208142077, 1.0749744962294578, 1.3337280869640302, -1.639357144162334, -0.41261662163620694, 1.2943556697948422, 0.47565655480823843, 2.2586282574007295, -2.59495908789121, -0.277134632857685, 0.5681627363039031, -0.18785628050938394, 0.48456100016974074, -0.10980491827866126, 0.9431083535091886, 0.5638441672270761, 1.3983219308829544, -0.4376886200249245, 1.547030999249325, -1.397215187372595, -0.9002864102375698]

settings.no_rendering_mode = False
world.apply_settings(settings)

"""
while (True):
	print("probar mejor resultado?")
	prueba = input()
	if(prueba != "n"):
		funcion_Aptitud_Recta(gbest_pos)
	else:
		break
"""
#salidaVideo.release()
cv2.destroyAllWindows()