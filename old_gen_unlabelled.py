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

num_videos = 5
num_frames = 200
dimensions = [10, 10]
data = []


# Trajectory generators
def CircleTrajectory(num_frames, radius):
    angles = [2 * math.pi * i / num_frames for i in range(num_frames)]

    for angle in angles:
        camera = {"x": radius * math.cos(angle), "y": 2, "z": radius * math.sin(angle)}
        lookat = {"x": 0.00001, "y": 0.01, "z": -0.001}
        yield camera, lookat


# Creates objects dictionary
lib = ModelLibrarian(library="models_core.json")

# objects = dict({})
# search_terms = ["chair", "microwave", "backpack", "bed"]
#
# for search_term in search_terms:
#     records = lib.search_records(search_term)
#     objects[search_term] = [record.name for record in records]


#objects = dict({'chair': ['blue_club_chair', 'blue_side_chair', 'brown_leather_dining_chair', 'brown_leather_side_chair', 'chair_annabelle', 'chair_billiani_doll', 'chair_willisau_riale', 'dark_red_club_chair', 'emeco_navy_chair', 'green_side_chair', 'lapalma_stil_chair', 'linbrazil_diz_armchair', 'linen_dining_chair', 'red_side_chair', 'tan_lounger_chair', 'tan_side_chair', 'vitra_meda_chair', 'white_club_chair', 'wood_chair', 'yellow_side_chair'], 'sofa': ['arflex_hollywood_sofa', 'arflex_strips_sofa', 'meridiani_freeman_sofa', 'minotti_helion_3_seater_sofa', 'napoleon_iii_sofa', 'on_the_rocks_sofa', 'sayonara_sofa']})
#sink_base_wood_beach_honey

#objects = dict({'table': ['dining_room_table'], 'microwave': ['appliance-ge-profile-microwave'], 'backpack': ['b04_backpack'], 'bowl': ['round_bowl_small_beech'], 'sofa': ['sayonara_sofa']})
#objects = dict({'table': ['dining_room_table'], 'microwave': ['appliance-ge-profile-microwave'], 'backpack': ['b04_backpack'], 'bowl': ['round_bowl_small_beech'], 'sofa': ['sayonara_sofa']})
#objects = dict({'table': ['dining_room_table'], 'microwave': ['appliance-ge-profile-microwave'], 'backpack': ['b04_backpack'], 'bowl': ['round_bowl_small_beech'], 'sofa': ['sayonara_sofa']})
objects = dict({'cup':["coffee_cup", "coffeecup004"]})

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
    num_objects = 1#randrange(1, 5)
    print("num_objects: ", num_objects)

    labels = []
    category_names = []
    placement_height = 0
    for obj_id in range(num_objects):
        x = random.uniform(-(dimensions[0]/2 - 2), dimensions[0]/2 - 2)
        z = random.uniform(-(dimensions[1]/2 - 2), dimensions[1]/2 - 2)

        angle = random.uniform(0, 360)

        coco_name = random.choice(list(objects.keys()))
        category_names.append(coco_name)
        tdw_name = random.choice(objects[coco_name])

        resp = c.communicate([c.get_add_object(model_name = tdw_name, object_id = obj_id),
                              {"$type": "teleport_object", "id": obj_id, "position": {"x": x, "y": placement_height, "z": z}},
                              {"$type": "rotate_object_to_euler_angles", "euler_angles": {"x": 0, "y": angle, "z": 0},
                                  "id": obj_id}])

        resp = c.communicate([{"$type": "send_collisions", "stay": True}])

        # Check that object does not collide with other objects and the wall
        collisions = []
        for r in resp:
            r_id = OutputData.get_data_type_id(r)
            print(r_id)
            if r_id == "coll":
                collisions.append(Collision(r))

        x_lim = dimensions[0]/2 - 2
        z_lim = dimensions[1]/2 - 2


        while collisions:
            for col in collisions:
                x = random.uniform(-(dimensions[0]/2 - 2), dimensions[0]/2 - 2)
                z = random.uniform(-(dimensions[1]/2 - 2), dimensions[1]/2 - 2)

                x_1 = random.uniform(-(dimensions[0]/2 - 2), dimensions[0]/2 - 2)
                z_1 = random.uniform(-(dimensions[1]/2 - 2), dimensions[1]/2 - 2)
                collider = col.get_collider_id()
                collidee = col.get_collidee_id()
                print(f"collision between objects {collider}, {collidee}")
                angle_1 = random.uniform(0, 360)
                resp = c.communicate([{"$type": "teleport_object", "id": collider, "position": {"x": x, "y":
                    placement_height, "z": z}},
                                      {"$type": "teleport_object", "id": collidee, "position": {"x": x_1, "y":
                                          placement_height, "z": z_1}},
                                      {"$type": "rotate_object_to_euler_angles", "euler_angles": {"x": 0, "y": angle, "z": 0}, "id": collider},
                                      {"$type": "rotate_object_to_euler_angles", "euler_angles": {"x": 0, "y": angle_1, "z":
                                          0}, "id": collidee}])
                resp = c.communicate({"$type": "send_collisions", "stay": True})

            # refresh col and b
            collisions = []
            for r in resp:
                r_id = OutputData.get_data_type_id(r)
                print(r_id)
                if r_id == "coll":
                    collisions.append(Collision(r))



        print("placed object {}".format(obj_id))
        print(tdw_name)
        print(x, z)
        print('#'*20)

    for obj_id in range(num_objects):
        resp = c.communicate({"$type": "set_kinematic_state", "id": obj_id, "is_kinematic": True, "use_gravity": False})

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

    c.communicate({"$type": "simulate_physics", "value": False})
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
    trajectories = CircleTrajectory(num_frames, dimensions[0]/2 - 2)
    for frame, (camera, lookat) in zip(range(num_frames), trajectories):
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
