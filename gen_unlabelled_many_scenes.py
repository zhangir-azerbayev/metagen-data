import os
import csv
import math
import random
import json
from random import randrange
from tqdm import tqdm
random.seed(19)

import numpy as np
from scipy.spatial.distance import cdist as cdist

from pathlib import Path
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.librarian import ModelLibrarian
from tdw.librarian import SceneLibrarian
from tdw.output_data import OutputData, Bounds, Images, Collision, EnvironmentCollision, Environments

num_videos = 100
num_frames = 20
#place_dimensions = [7, 7]
delta_too_close = 1 #3
#room_dimensions = [10, 10]
#traj_radius = 5
camera_var = .7
camera_l = 2.5
lookat_var = 0.7
lookat_l = 2
camera_height = 2
data = []


# Trajectory generators
def CircleTrajectory(num_frames, radius, camera_height):
    angles = [2 * math.pi * i / num_frames for i in range(num_frames)]

    for angle in angles:
        camera = {"x": radius * math.cos(angle), "y": camera_height, "z": radius * math.sin(angle)}
        lookat = {"x": -radius * math.cos(angle), "y": 0, "z": -radius * math.sin(angle)}
        yield camera, lookat


def gp_noise_trajectory(num_frames,
                        radius_x,
                        radius_y,
                        camera_var,
                        camera_l,
                        camera_height,
                        lookat_var,
                        lookat_l,
                        room_dimensions):
    def rbf_kernel(xa, xb, var, l):
        sq_norm = -0.5 * cdist(xa, xb, 'sqeuclidean') / l**2
        return var * np.exp(sq_norm)

    angles = [2 * math.pi * i / num_frames for i in range(num_frames)]
    ts = np.expand_dims(np.linspace(0, 10, num_frames), 1)
    camera_cov = rbf_kernel(ts, ts, camera_var, camera_l)
    lookat_cov = rbf_kernel(ts, ts, lookat_var, lookat_l)

    camera_dx = np.random.multivariate_normal(mean=np.zeros(num_frames),
            cov=camera_cov, size=2)

    lookat_dx = np.random.multivariate_normal(mean=np.zeros(num_frames),
            cov=lookat_cov, size=3)
    lookat_dx[2] = lookat_dx[2] #+ (camera_height - 0.5) #trying to have the camera looking down a bit

    for t in range(num_frames):
        camera_x = radius_x * math.cos(angles[t]) + camera_dx[0, t],
        camera_z = radius_y * math.sin(angles[t]) + camera_dx[1, t]

        wall_offset = .8

        camera = {"x": np.clip(camera_x, -room_dimensions[0]/2 + wall_offset, room_dimensions[0]/2 - wall_offset).item(),
                  "y": camera_height,
                  "z": np.clip(camera_z, -room_dimensions[1]/2 + wall_offset, room_dimensions[1]/2 - wall_offset).item()}
        lookat = {"x": np.clip(lookat_dx[0, t], -room_dimensions[1]/2, room_dimensions[1]/2).item(),
                  "y": np.clip(lookat_dx[2, t], 0, camera_height).item(),# clamping to not look below the floor or above the camera's height
                  "z": np.clip(lookat_dx[1, t], -room_dimensions[1]/2, room_dimensions[1]/2).item()}

        yield camera, lookat

def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)



# Creates objects dictionary
lib = ModelLibrarian(library="models_full.json")
scene_lib = SceneLibrarian()

# objects = dict({'chair': ['blue_club_chair', 'chair_willisau_riale', 'tan_side_chair'],
# 'plant': ['b03_pot_plant_20_2013__vray', 'monstera_plant', 'potted_plant', 'potted_plant_trellis'],
# 'tv': ['sm_tv'],
# 'umbrella': ['b05_umbrella_(open_and_close)', 'umbrella_fix'],
# 'bowl': ['round_bowl_large_metal_perf']}) #padauk gets mistaken for umbrella a lot

objects = dict({'chair': ['chair_willisau_riale'],
'plant': ['monstera_plant'],
'tv': ['sm_tv'],
'umbrella': ['umbrella_fix'],
'bowl': ['round_bowl_large_metal_perf']})


records = scene_lib.search_records("")
to_use = [2, 34, 35, 36, 37] #box_room_2018, mm_kitchen_2a, mm_kitchen_2b, mm_kitchen_3a, mm_kitchen_3b
#records = lib.search_records("box_room_2018")
#url = [record for record in records][0].get_url()

print(objects)

for video in tqdm(range(num_videos)):
    print("video ", video)

    scene_num = random.choice(to_use)
    url = records[scene_num].get_url()
    scene_name = records[scene_num].name
    print("scene_num: ", scene_num)
    print("scene_name: ", scene_name)

    c = Controller(launch_build=True)
    c.communicate({"$type": "set_render_quality", "render_quality": 3})
    c.communicate({"$type": "simulate_physics", "value": False})#switched
    c.communicate({"$type": "set_time_step", "time_step": 1})

    #print("video {}".format(video))
    # Loads scene, creates empty room

    c.communicate({"$type": "add_scene",
            "name": scene_name,
            "url": url})

    resp = c.communicate({"$type": "send_environments"})

    for r in resp:
        r_id = OutputData.get_data_type_id(r)
        if r_id == "envi":
            env = Environments(r)
            #print(env.get_center(0))
            #print(env.get_bounds(0))
            room_dim_x = env.get_bounds(0)[0]#getting x and z, which are the floor
            room_dim_z = env.get_bounds(0)[2]#getting x and z, which are the floor
            room_dimensions = [room_dim_x, room_dim_z]
            print(room_dimensions)
            print(env.get_center(0))

    print("created empty scene")
    # Populates scene with objects (placement is uniform random)
    num_objects = randrange(1, 4)
    print("num_objects: ", num_objects)

    #initialize a tracker for where object get placed
    placed_objects_x = []
    placed_objects_z = []

    labels = []
    category_names = []
    placement_height = 0
    x_lim = room_dimensions[0]/2 - delta_too_close
    z_lim = room_dimensions[1]/2 - delta_too_close

    for obj_id in range(num_objects):
        x = random.uniform(-x_lim, x_lim)
        z = random.uniform(-z_lim, z_lim)

        j = 1 #indexes through the objects that have already been placed and accepted
        while j <= len(placed_objects_x):
            print("in while loop")
            print("distance ", distance(placed_objects_x[j-1], placed_objects_z[j-1], x, z))
            if distance(placed_objects_x[j-1], placed_objects_z[j-1], x, z) < delta_too_close: #if too close, resample. j-1 is for zero-indexing
                print("resampling")
                x = random.uniform(-x_lim, x_lim)
                z = random.uniform(-z_lim, z_lim)
                j = 1 #restart loop over objects already placed
            else:
                j = j + 1

        placed_objects_x.append(x)
        placed_objects_z.append(z)

        angle = random.uniform(0, 360)

        coco_name = random.choice(list(objects.keys()))
        category_names.append(coco_name)
        tdw_name = random.choice(objects[coco_name])

        print({tdw_name})

        resp = c.communicate([c.get_add_object(model_name = tdw_name, object_id = obj_id, library="models_full.json"),
                              {"$type": "teleport_object", "id": obj_id, "position": {"x": x, "y": placement_height, "z": z}},
                              {"$type": "rotate_object_to_euler_angles", "euler_angles": {"x": 0, "y": angle, "z": 0},
                                  "id": obj_id}])

        print("placed object {}".format(obj_id))
        #print(tdw_name)
        print(x, z)
        print('#'*20)

    # saves labels
    resp = c.communicate({"$type": "send_bounds", "ids": list(range(num_objects))})
    for r in resp:
        r_id = OutputData.get_data_type_id(r)
        if r_id == 'boun':
            b = Bounds(r)

    print('saving labels')
    print('number of responses: ', len(resp))
    for index in range(num_objects): #order coming out of response might not match order of object ids
        obj = b.get_id(index)
        position = b.get_center(index)
        labels.append({"category_name": category_names[obj], "position": position})

    # Creates frames
    angles = [2 * math.pi * i / num_frames for i in range(num_frames)]

    avatar_id = "a"

    resp = c.communicate([{"$type": "create_avatar",
                           "type": "A_Img_Caps_Kinematic",
                           "avatar_id": avatar_id},
                           {"$type": "set_aperture",
                           "aperture": 26.0},#removes bluriness
                          {"$type": "set_pass_masks",
                           "avatar_id": avatar_id,
                           "pass_masks": ["_img"]}])
    print("created avatar")

    views = []
    trajectories = gp_noise_trajectory(num_frames, room_dim_x/2, room_dim_z/2,
            camera_var, camera_l, camera_height, lookat_var, lookat_l,
            room_dimensions)
    #trajectories = CircleTrajectory(num_frames, min(room_dim_x/2, room_dim_z/2))
    for frame, (camera, lookat) in zip(range(num_frames), trajectories):
        resp = c.communicate([{"$type": "teleport_avatar_to",
                                "position": camera,
                                "avatar_id": avatar_id},
                                {"$type": "set_field_of_view",
                                "field_of_view": 55,
                                "avatar_id": "a"},
                                {"$type": "set_aperture",
                                "aperture": 16.0},#removes bluriness
                                {"$type": "look_at_position",
                                "avatar_id": avatar_id,
                                "position": lookat }])

        resp = c.communicate({"$type": "send_images", "frequency": "once"})
        views.append({"camera": camera, "lookat": lookat})

        for r in resp:
            r_id = OutputData.get_data_type_id(r)
            if r_id == "imag":
                img = Images(r)

        TDWUtils.save_images(img, filename=f"{video:03}_{frame:03}")

    # saves scene to data list
    data.append({"views": views, "labels": labels, "path": f"{video:03}"})
    # Terminates build
    c.communicate({"$type": "terminate"})
    del c

with open("unlabelled_data.json", "w") as f:
    json.dump(data, f)
