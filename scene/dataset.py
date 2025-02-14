import os

import numpy as np
import torch
from arguments import ModelParams
from PIL import Image
from scene.cameras import Camera
from scene.dataset_readers import CameraInfo
from scene.neural_3D_dataset_NDC import Neural3D_NDC_Dataset
from torch.utils.data import Dataset

from utils.graphics_utils import focal2fov


class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset: "list[CameraInfo] | Neural3D_NDC_Dataset",
        args: ModelParams,
    ):
        self.dataset = dataset
        self.args = args

    def __getitem__(self, index):
        try:
            image, w2c, time = self.dataset[index]
            R, T = w2c
            FovX = focal2fov(self.dataset.focal[0], image.shape[2])
            FovY = focal2fov(self.dataset.focal[0], image.shape[1])
            image_path = self.dataset.image_paths[index]
        except:
            caminfo: CameraInfo = self.dataset[index]
            image = caminfo.image
            R = caminfo.R
            T = caminfo.T
            FovX = caminfo.FovX
            FovY = caminfo.FovY
            time = caminfo.time
            image_path = caminfo.image_path
            if not (image_path.endswith(".jpg") or image_path.endswith(".png")):
                image_path = os.path.join(image_path, caminfo.image_name)
            time_index = caminfo.time_index
            view_index = caminfo.view_index

        # from time import perf_counter
        # tic = perf_counter()

        if self.args.read_flow_fwd or self.args.read_flow_bwd:
            flow_dir = "flow_bidir"
            flow_ext = "npy"
            if "dynerf" in image_path:
                flow_path = image_path.replace("/images/", f"/{flow_dir}/").replace(".png", f"_pred.{flow_ext}")
            elif "hypernerf" in image_path:
                flow_path = image_path.replace("/rgb/", f"/{flow_dir}/").replace(".png", f"_pred.{flow_ext}")
            elif "sv4d" in image_path:
                flow_path = image_path.replace("/images/", f"/{flow_dir}/").replace(".png", f"_pred.{flow_ext}")
            else:
                raise NotImplementedError

        flow = None
        occ = None
        if self.args.read_flow_fwd and os.path.isfile(flow_path):
            from utils.flow_utils import readFlow
            flow = readFlow(flow_path)
            occ_path = flow_path.replace(f"_pred.{flow_ext}", "_occ_fwd.png")
            if self.args.use_flow_occ and os.path.isfile(occ_path):
                occ = (np.array(Image.open(occ_path)) != 0).astype(np.uint8)
        # else:
        #     if self.args.read_flow_fwd:
        #         print(f"No flow_fwd for {image_path}")

        # backward flow
        flow_bwd = None
        occ_bwd = None
        if self.args.read_flow_bwd:
            flow_path_bwd = flow_path.replace(f".{flow_ext}", f"_bwd.{flow_ext}")
            if os.path.isfile(flow_path_bwd):
                from utils.flow_utils import readFlow
                flow_bwd = readFlow(flow_path_bwd)
                occ_bwd_path = flow_path_bwd.replace(f"_pred_bwd.{flow_ext}", "_occ_bwd.png")
                if self.args.use_flow_occ and os.path.isfile(occ_bwd_path):
                    occ_bwd = (np.array(Image.open(occ_bwd_path)) != 0).astype(np.uint8)
            # else:
            #     if self.args.read_flow_bwd:
            #         print(f"No flow_bwd for {image_path}")

        # print("time to read flow", perf_counter() - tic)
        # pass

        return Camera(
            colmap_id=index,
            R=R,
            T=T,
            FoVx=FovX,
            FoVy=FovY,
            image=image,
            gt_alpha_mask=None,
            image_name=f"{index}",
            uid=index,
            time_index=time_index,
            view_index=view_index,
            data_device=torch.device("cuda"),
            time=time,
            image_path=image_path,
            flow=[flow, flow_bwd, occ, occ_bwd],
        )

    def __len__(self):
        return len(self.dataset)
