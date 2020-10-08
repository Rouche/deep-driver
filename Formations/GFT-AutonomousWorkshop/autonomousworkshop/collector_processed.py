import datetime
import logging
import os
import random

import carla
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# Constants
WIDTH = 200
HEIGHT = 88

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(10.0)
client.reload_world()
world = client.get_world()
# world.set_weather(carla.WeatherParameters())

# Directory to save
today = datetime.datetime.now()
h = str(today.hour)
if today.hour < 10:
    h = "0" + h
m = str(today.minute)
if today.minute < 10:
    m = "0" + m

directory = "./generated/data/" + today.strftime('%Y%m%d_') + h + m + "_npy"

try:
    os.makedirs(directory)
except:
    print("Directory already exists: [" + directory + "]")
try:
    inputs_file = open(directory + "/inputs.npy", "ba+")
    outputs_file = open(directory + "/outputs.npy", "ba+")
except:
    print("File could not be openned: [" + directory + "]")

# Add a car
ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
ego_bp.set_attribute('role_name', 'ego')
ego_color = random.choice(ego_bp.get_attribute('color').recommended_values)
ego_bp.set_attribute('color', ego_color)

spawn_points = world.get_map().get_spawn_points()
number_of_spawn_points = len(spawn_points)

if number_of_spawn_points > 0:
    random.shuffle(spawn_points)
    ego_transform = spawn_points[0]
    ego_vehicle = world.spawn_actor(ego_bp, ego_transform)
    ego_vehicle.set_autopilot(True)
else:
    logging.warning('Could not find any spawn point')

# Add a sensor RGB Camera
cam_bp = None
cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
cam_bp.set_attribute('image_size_x', str(WIDTH))
cam_bp.set_attribute('image_size_y', str(HEIGHT))
cam_bp.set_attribute('fov', str(105))
cam_location = carla.Location(2, 0, 1)
cam_rotation = carla.Rotation(0, 0, 0)
cam_transform = carla.Transform(cam_location, cam_rotation)

# Spawn it
ego_cam = world.spawn_actor(
    cam_bp,
    cam_transform,
    attach_to=ego_vehicle,
    attachment_type=carla.AttachmentType.Rigid
)

current = 0

# Function to convert image to a nupy array
def process_image(image):
    # Get raw image 8bit format
    raw_image = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    # Reshape image to RGBA
    raw_image = np.reshape(raw_image, (image.height, image.width, 4))
    # Take only RGB
    processed_image = raw_image[:, :, :3]

    # Save the processed image to see how it looks.
    plt.imsave(f"{directory}/{image.frame:06}.jpg", processed_image)

    return processed_image


def save_image(carla_image):
    image = process_image(carla_image)
    ego_control = ego_vehicle.get_control()
    data = [ego_control.steer, ego_control.throttle, ego_control.brake]
    np.save(inputs_file, image)
    np.save(outputs_file, data)
    # carla_image.save_to_disk(f"{directory}/{carla_image.frame:06}.jpg")
    # display(f"{str(current)} saved in np")


# Record data
ego_cam.listen(save_image)

try:
    while current < 302:
        world_snapshot = world.wait_for_tick()
        clear_output(wait=True)
        # display(f"{str(current)} frames saved")
        current += 1
except:
    print("\nSimulation error")

display(f"{str(current)} frames saved")
# print("Sleeping a bit.....")
# time.sleep(5)

if ego_vehicle is not None and ego_cam is not None:
    ego_cam.stop()
    ego_cam.destroy()
    ego_vehicle.destroy()

inputs_file.close()
outputs_file.close()

# control_file.close()
print("Data retrieval finished")
print(directory)
