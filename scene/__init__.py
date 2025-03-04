#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from glob import glob
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, SceneInfo
from scene.gaussian_model import GaussianModel
from scene.dataset import FourDGSdataset
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import torch

class Scene:

    gaussians : GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], load_coarse=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            elif load_iteration == -2:
                self.loaded_iter = "best"
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}
        assert os.path.isdir(args.source_path), f"Source path does not exist: {args.source_path}"
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            scene_info = sceneLoadTypeCallbacks["dynerf"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path,"dataset.json")):
            scene_info = sceneLoadTypeCallbacks["nerfies"](args.source_path, False, args.eval)
        else:
            assert False, "Could not recognize scene type!"
        scene_info: SceneInfo
        self.maxtime = scene_info.maxtime
        # if not self.loaded_iter:
        #     with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
        #         dest_file.write(src_file.read())
        #     json_cams = []
        #     camlist = []
        #     if scene_info.test_cameras:
        #         camlist.extend(scene_info.test_cameras)
        #     if scene_info.train_cameras:
        #         camlist.extend(scene_info.train_cameras)
            
        #     for id, cam in enumerate(camlist):
        #         json_cams.append(camera_to_JSON(id, cam))
        #     with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
        #         json.dump(json_cams, file)

        # if shuffle:
        #     random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
        #     random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # for resolution_scale in resolution_scales:
            # print("Loading Training Cameras")
            # self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            # print("Loading Test Cameras")
            # self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            # print("Loading Video Cameras")
            # self.video_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.video_cameras, resolution_scale, args)
        print("Loading Training Cameras...", end="")
        self.train_camera = FourDGSdataset(scene_info.train_cameras, args)
        print(f" Done! {len(self.train_camera)} items loaded.")
        print("Loading Test Cameras", end="")
        self.test_camera = FourDGSdataset(scene_info.test_cameras, args)
        print(f" Done! {len(self.test_camera)} items loaded.")
        print("Loading Video Cameras", end="")
        self.video_camera = cameraList_from_camInfos(scene_info.video_cameras,-1,args)
        print(f" Done! {len(self.video_camera)} items loaded.")
        xyz_max = scene_info.point_cloud.points.max(axis=0)
        xyz_min = scene_info.point_cloud.points.min(axis=0)
        self.gaussians._deformation.deformation_net.grid.set_aabb(xyz_max,xyz_min)
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            self.gaussians.load_model(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                   ))
        # elif load_coarse:
        #     self.gaussians.load_ply(os.path.join(self.model_path,
        #                                                    "point_cloud",
        #                                                    "coarse_iteration_" + str(load_coarse),
        #                                                    "point_cloud.ply"))
        #     self.gaussians.load_model(os.path.join(self.model_path,
        #                                             "point_cloud",
        #                                             "coarse_iteration_" + str(load_coarse),
        #                                            ))
        #     print("load coarse stage gaussians")
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.maxtime)

    def save(self, iteration, stage):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))

        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_deformation(point_cloud_path)

    def save_best(self, iteration: int, psnr: float):
        best_pth = glob(os.path.join(self.model_path, "best_*.pth"))
        try:
            pth_last = best_pth[0]
            psnr_last = float(os.path.basename(pth_last).removesuffix(".pth").split("_")[2])
        except:
            pth_last = None
            psnr_last = 0.0
        if psnr > psnr_last:
            print(f"Saving new best model with test PSNR {psnr:.4f} at iteration {iteration}")
            self.save(f"best", "fine")
            torch.save((self.gaussians.capture(), iteration), os.path.join(self.model_path, f"best_{iteration}_{psnr:.4f}.pth"))
            if pth_last:
                os.system(f"rm {pth_last}")

    def getTrainCameras(self, scale=1.0):
        return self.train_camera

    def getTestCameras(self, scale=1.0):
        return self.test_camera

    def getVideoCameras(self, scale=1.0):
        return self.video_camera