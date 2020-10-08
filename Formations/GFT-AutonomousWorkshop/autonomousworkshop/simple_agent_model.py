import abc
from collections import deque

import sys
import os
import math
import numpy as np
from enum import Enum
import random
import networkx

import tensorflow as tf

import carla
import glob

from carla import ColorConverter as cc

import collections
import datetime
import logging
import re
import weakref

import cv2

import time
import threading

import PIL.Image
from io import BytesIO
import IPython.display
import ipywidgets as widgets

# Constants
WIDTH = 200
HEIGHT = 88


class Agent(abc.ABC):

    @abc.abstractmethod
    def run_step(self, measurements, sensor_data, directions, target):
        """
        Function to be implemented by an agent..
        :param measurements The measurements like speed, the image data and a target
        :param sensor_data Data depending on sensor
        :param directions The direction of the car
        :param target Target waypoint
        :returns A carla Control object, with the steering/gas/break for the agent
        """


class ForwardAgent(Agent):
    # Implementation

    def run_step(self, measurements, sensor_data, directions, target):
        current_speed = measurements['speed']
        control = carla.VehicleControl();
        control.throttle = 0.75
        control.brake = 0.0
        return control


class ModelAgent(Agent):

    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        print('Model Loaded!')

    @staticmethod
    def process_image(image):
        processed_image = image[:, :, :3]

    def run_step(self, measurements, sensor_data, directions, target):
        current_speed = measurements['speed']
        input = sensor_data.camera['rgb']
        [[steer, throttle]] = self.model.predict(np.array([input]))

        control = carla.VehicleControl()
        control.throttle = throttle.item()
        control.steer = steer.item()
        control.brake = 0.0
        return control


agent = ForwardAgent()
control = agent.run_step({'speed': 10}, None, None, None)
print(control)


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\2026') if len(name) > truncate else name


class Sensor(object):
    def __init__(self):
        self.camera = dict()
        self.collision = None
        self.lane_invasion = None
        self.gnss = None
        self.imu = None

###############################
# WORLD
###############################
class World(object):
    restarted = False

    def __init__(self, client: carla.Client):
        self.client: carla.Client = client
        self.world: carla.World = client.get_world()
        self.roles_names = ['ego', 'hero', 'hero1', 'hero2', 'hero3', 'hero4', 'hero5', 'hero6']
        self.actor_role_name = random.choice(self.roles_names)
        self._sensors: Sensor = Sensor()
        self._measurements = dict()

        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            logging.error('RuntimeError: {}'.format(error))
            sys.exit(-1)

        self.player = None
        self.camera_manager = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None

        self._actor_filter = 'mustang'
        self._gamma = 2.2
        self.restart()

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep some camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get random blueprint
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', True)
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])
        else:
            print('No recommended values for speed')
        # Spawn the player
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            spawn_points = self.map.get_spawn_points()
            if not spawn_points:
                print('There are no spawn points available in your map.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(-1)
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        cam_index = 0
        cam_pos_index = 1
        # Set up the sensors
        self.collision_sensor = CollisionSensor(self.player)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self._gamma, WIDTH, HEIGHT)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)

    def render(self):
        self.camera_manager.render()

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        actors = [
            self.camera_manager.sensor,
            self.player
        ]
        for actor in actors:
            if actor is not None:
                actor.destroy()

    def get_sensors(self):
        self._sensors.camera.update({
            'rgb': self.camera_manager.surface,
            'semantic': self.camera_manager.surface_semantic,
            'depth': self.camera_manager.surface_depth
        })
        self._sensors.collision = self.collision_sensor
        self._sensors.lane_invasion = self.lane_invasion_sensor
        self._sensors.gnss = self.gnss_sensor
        self._sensors.imu = self.imu_sensor
        return self._sensors

    def _get_speed(self):
        vel = self.player.get_velocity()
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y **2 + vel.z ** 2)

    def get_measurements(self):
        self._measurements.update({
            'speed': self._get_speed()
        })
        return self._measurements

###############################
# Sensors
###############################
class CollisionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        world: carla.World = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular reference
        weak_self = weakref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2+impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

class LaneInvasionSensor(object):
    '''Class for lane invasion sensors'''

    def __init__(self, parent_actor, hud):
        '''Constructor method'''
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        '''On invasion method'''
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


class IMUSensor(object):
    pass

class GnssSensor(object):
    ''' Class for GNSS sensors'''

    def __init__(self, parent_actor):
        '''Constructor method'''
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        '''GNSS method'''
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

class CameraManager(object):
    ''' Class for camera management'''

    def __init__(self, parent_actor, hud, gamma_correction):
        '''Constructor method'''
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(
                carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=1.6, z=1.7)), attachment.Rigid),
            (carla.Transform(
                carla.Location(x=5.5, y=1.5, z=1.5)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-1, y=-bound_y, z=0.5)), attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                blp.set_attribute('image_size_x', str(hud.dim[0]))
                blp.set_attribute('image_size_y', str(hud.dim[1]))
                if blp.has_attribute('gamma'):
                    blp.set_attribute('gamma', str(gamma_correction))
            elif item[0].startswith('sensor.lidar'):
                blp.set_attribute('range', '50')
            item.append(blp)
        self.index = None

    def toggle_camera(self):
        '''Activate a camera'''
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        '''Set a sensor'''
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
                force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])

            # We need to pass the lambda a weak reference to
            # self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        '''Get the next sensor'''
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        '''Toggle recording on or off'''
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        '''Render method'''
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(self.sensors[self.index][1])
        array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return array.copy()

###############################
# Client
###############################

log_level = logging.DEBUG
logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

try:
    logging.info('Starting tests')
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    world = World(client)

    while True:
        world.render()
        sensor_data = world.get_sensors()
        measurements = world.get_measurements()
        control = agent.run_step(measurements, sensor_data, None, None)
        world.player.apply_control(control)
except:
    logging.info('\nCancelled by user. Bye!')
    if world is not None:
        world.destroy()

