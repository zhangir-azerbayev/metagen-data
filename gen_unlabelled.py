import os
import csv
import math
import random
import json
from random import randrange
from tqdm import tqdm
random.seed(17)

from pathlib import Path
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.librarian import ModelLibrarian
from tdw.output_data import OutputData, Bounds, Images, Collision, EnvironmentCollision

num_videos = 20
num_frames = 250
dimensions = [10, 10]
data = []


# Creates objects dictionary
lib = ModelLibrarian(library="models_core.json")

objects = dict({})
search_terms = ["chair", "sofa"]

for search_term in search_terms:
    records = lib.search_records(search_term)
    objects[search_term] = [record.name for record in records]


print(objects)


for video in tqdm(range(num_videos)):
    c = Controller(launch_build=True)
    c.communicate({"$type": "set_render_quality", "render_quality": 3})
    c.communicate({"$type": "simulate_physics", "value": True})
    c.communicate({"$type": "set_time_step", "time_step": 1})

    print("video {}".format(video))
    # Loads scene, creates empty room

    resp = c.communicate([{"$type": "load_scene",
                           "scene_name": "ProcGenScene"},
                          TDWUtils.create_empty_room(*dimensions)])
    print("created empty scene")
    # Populates scene with objects (placement is uniform random)
    num_objects = randrange(2, 6)
    print("num_objects: ", num_objects)

    labels = []
    category_names = []
    for id in range(num_objects):
        x = random.uniform(-(dimensions[0]/2 - 2), dimensions[0]/2 - 2)
        z = random.uniform(-(dimensions[1]/2 - 2), dimensions[1]/2 - 2)

        angle = random.uniform(0, 360)

        coco_name = random.choice(list(objects.keys()))
        category_names.append(coco_name)
        tdw_name = random.choice(objects[coco_name])

        resp = c.communicate([c.get_add_object(model_name = tdw_name, object_id = id),
                              {"$type": "teleport_object", "id": id, "position": {"x": x, "y": 0, "z": z}},
                              {"$type": "rotate_object_to_euler_angles", "euler_angles": {"x": 0, "y": angle, "z": 0}, "id": id}])

        resp = c.communicate([{"$type": "send_collisions", "stay": True}])
                              #{"$type": "send_bounds", "id": id}])

        # Check that object does not collide with other objects and the wall
        col = None
        enco = None
        for r in resp:
            r_id = OutputData.get_data_type_id(r)
            print(r_id)
            if r_id == "coll":
                col = Collision(r)
            if r_id == "enco":
                enco = EnvironmentCollision(r)

        x_lim = dimensions[0]/2 - 2
        z_lim = dimensions[1]/2 - 2


        while enco or col:
            print(f"object{id} hitting something")
            x = random.uniform(-(dimensions[0]/2 - 2), dimensions[0]/2 - 2)
            z = random.uniform(-(dimensions[1]/2 - 2), dimensions[1]/2 - 2)
            if enco:
                print(f"hitting wall, moving object{id}")
                collider = enco.get_object_id()
                resp = c.communicate([{"$type": "teleport_object", "id": collider, "position": {"x": x, "y": 0, "z": z}},
                                      {"$type": "rotate_object_to_euler_angles", "euler_angles": {"x": 0, "y": angle, "z": 0}, "id": collider},
                                      {"$type": "send_collisions", "stay": True}])
            else: # col
                print(f"hitting object, moving object{id}")
                x_1 = random.uniform(-(dimensions[0]/2 - 2), dimensions[0]/2 - 2)
                z_1 = random.uniform(-(dimensions[1]/2 - 2), dimensions[1]/2 - 2)
                collider = col.get_collider_id()
                collidee = col.get_collidee_id()
                angle_1 = random.uniform(0, 360)
                resp = c.communicate([{"$type": "teleport_object", "id": collider, "position": {"x": x, "y": 0, "z": z}},
                                      {"$type": "teleport_object", "id": collidee, "position": {"x": x_1, "y": 0, "z": z_1}},
                                      {"$type": "rotate_object_to_euler_angles", "euler_angles": {"x": 0, "y": angle, "z": 0}, "id": collider},
                                      {"$type": "rotate_object_to_euler_angles", "euler_angles": {"x": 0, "y": angle_1, "z": 0}, "id": collidee},
                                      {"$type": "send_collisions", "stay": True}])

            # refresh col and b
            col = None
            enco = None
            for r in resp:
                r_id = OutputData.get_data_type_id(r)
                print(r_id)
                if r_id == "coll":
                    col = Collision(r)
                if r_id == "enco":
                    enco = EnvironmentCollision(r)




        print("placed object {}".format(id))
        print(tdw_name)
        print(x, z)
        print('#'*20)

    # saves labels
    resp = c.communicate({"$type": "send_bounds", "ids": list(range(num_objects))})
    b = Bounds(resp[0])

    for obj in range(num_objects):
        position = b.get_center(obj)
        labels.append({"category_name": category_names[obj], "position": position})


    c.communicate({"$type": "simulate_physics", "value": True})
    # Creates frames
    radius = dimensions[0]/2 - 2
    angles = [2 * math.pi * i / num_frames for i in range(num_frames)]

    avatar_id = "a"

    resp = c.communicate([{"$type": "create_avatar",
                           "type": "A_Img_Caps_Kinematic",
                           "avatar_id": avatar_id},
                          {"$type": "set_pass_masks",
                           "avatar_id": avatar_id,
                           "pass_masks": ["_img"]}])
    print("created avatar")

    views = []
    for frame in range(num_frames):
        angle = angles[frame]

        camera = {"x": radius * math.cos(angle), "y": 3, "z": radius * math.sin(angle)}
        lookat = {"x": -radius * math.cos(angle), "y": 0, "z": -radius*math.cos(angle)}
        resp = c.communicate([{"$type": "teleport_avatar_to",
                               "position": camera,
                               "avatar_id": avatar_id},
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