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
from argparse import ArgumentParser
from os import makedirs
from time import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from arguments import ModelHiddenParams, ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from scene.cameras import Camera
from tqdm import tqdm

from utils.general_utils import safe_state

# mesh
from utils.mesh_utils import MiniCam, OrbitCamera, mipmap_linear_grid_put_2d, orbit_camera, safe_normalize

to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    render_images = []
    gt_list = []
    render_list = []
    means3D_list = []
    # gaussians.prune_points(gaussians.get_opacity.squeeze(-1) < 0.01)
    gaussians.eval()
    for idx, view in enumerate(tqdm(views, desc="Rendering progress", dynamic_ncols=True)):
        if idx == 0:
            time1 = time()
        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        means3D_deform = render_pkg["means3D_deform"]

        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        render_images.append(to8b(rendering).transpose(1, 2, 0))
        # print(to8b(rendering).shape)
        render_list.append(rendering)
        means3D_list.append(means3D_deform)
        if name in ["train", "test"]:
            gt = view.original_image[0:3, :, :]
            # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            gt_list.append(gt)
    time2 = time()
    print("FPS:", (len(views) - 1) / (time2 - time1))
    count = 0
    print("writing training images.")
    if len(gt_list) != 0:
        for image in tqdm(gt_list):
            torchvision.utils.save_image(image, os.path.join(gts_path, "{0:05d}".format(count) + ".png"))
            count += 1
    count = 0
    print("writing rendering images.")
    if len(render_list) != 0:
        for image in tqdm(render_list):
            torchvision.utils.save_image(image, os.path.join(render_path, "{0:05d}".format(count) + ".png"))
            count += 1
    imageio.mimwrite(
        os.path.join(model_path, name, "ours_{}".format(iteration), "video_rgb.mp4"), render_images, fps=10, quality=8
    )


def render_orbit(model_path, iteration, views: "list[Camera]", gaussians, pipeline, background):
    name = "orbit"
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    os.system(f"rm -r {render_path}")
    makedirs(render_path, exist_ok=True)

    H, W, radius = 576, 576, 4
    view = views[0]
    cam = OrbitCamera(W, H, r=radius, fovy=view.FoVy * 180 / np.pi, near=view.znear, far=view.zfar)
    render_resolution = 576
    elevation = 0
    # azimuths = list(range(-30, -30 + 360, 5))
    azimuths = [-45] * 25 + [0] * 25 + [45] * 25
    # azi_min, azi_max = -45, 20
    # azimuths = list(range(azi_min, azi_max + 1, 2)) + list(range(azi_min + 1, azi_max, 2))[::-1]
    # times = np.linspace(0, 1, len(azimuths))
    times = list(np.linspace(0, 1, 25)) + list(np.linspace(1, 0, 25))

    render_images = []
    render_list = []
    gaussians.eval()
    for i, hor in enumerate(tqdm(azimuths, desc="Rendering progress", dynamic_ncols=True)):
        # render image
        pose = orbit_camera(elevation, hor, cam.radius)
        cur_cam = MiniCam(
            pose,
            render_resolution,
            render_resolution,
            cam.fovy,
            cam.fovx,
            cam.near,
            cam.far,
            time=times[i % len(times)],
        )
        render_pkg = render(cur_cam, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        render_images.append(to8b(rendering).transpose(1, 2, 0))
        # print(to8b(rendering).shape)
        render_list.append(rendering)
    count = 0
    print("writing rendering images.")
    if len(render_list) != 0:
        for image in tqdm(render_list):
            torchvision.utils.save_image(image, os.path.join(render_path, "{0:05d}".format(count) + ".png"))
            count += 1
    imageio.mimwrite(
        os.path.join(model_path, name, "ours_{}".format(iteration), "video_rgb.mp4"), render_images, fps=10, quality=8
    )


@torch.no_grad()
def export_mesh(model_path, name, iteration, views: "list[Camera]", gaussians: GaussianModel, pipeline, background):
    mesh_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    makedirs(mesh_path, exist_ok=True)

    texture_size = 1024
    density_thresh = 0.5
    device = "cuda"
    H, W, radius = 576, 576, 4

    # gaussians.prune_points(gaussians.get_opacity.squeeze(-1) < 0.01)
    gaussians.eval()
    for t, view in enumerate(tqdm(views, desc="Exporting meshes", dynamic_ncols=True)):
        cam = OrbitCamera(W, H, r=radius, fovy=view.FoVy * 180 / np.pi, near=view.znear, far=view.zfar)
        path = os.path.join(mesh_path, f"mesh_{t:03d}.obj")
        mesh = gaussians.extract_mesh_t(density_thresh, t=view.time)
        mesh = mesh.filter_laplacian(iter=3)

        # perform texture extraction
        print(f"[INFO] unwrap uv...")
        h = w = texture_size
        mesh.auto_uv()
        mesh.auto_normal()

        albedo = torch.zeros((h, w, 3), device=device, dtype=torch.float32)
        cnt = torch.zeros((h, w, 1), device=device, dtype=torch.float32)
        # mesh.albedo = albedo
        # mesh.write(path)
        # print(os.path.abspath(path))

        vers_len = 9
        hors_len = 36
        vers = (
            [0] * hors_len
            + [10] * hors_len
            + [20] * hors_len
            + [30] * hors_len
            + [-10] * hors_len
            + [-20] * hors_len
            + [-30] * hors_len
            + [-45] * hors_len
            + [45] * hors_len
            + [-89, 89]
        )
        hors = list(range(-180, 180, 10)) * vers_len + [0] * 2
        assert len(vers) == len(hors)
        render_resolution = 576

        import nvdiffrast.torch as dr

        glctx = dr.RasterizeCudaContext()

        for ver, hor in zip(vers, hors):
            # render image
            pose = orbit_camera(ver, hor, cam.radius)
            cur_cam = MiniCam(
                pose,
                render_resolution,
                render_resolution,
                cam.fovy,
                cam.fovx,
                cam.near,
                cam.far,
                time=view.time,
            )
            render_pkg = render(cur_cam, gaussians, pipeline, background)
            rgbs = render_pkg["render"]
            rgbs = torch.clamp(rgbs, min=0.0, max=1.0)
            # torchvision.utils.save_image(rgbs, "test.png")

            # enhance texture quality with zero123 [not working well]
            # if self.opt.guidance_model == 'zero123':
            #     rgbs = self.guidance.refine(rgbs, [ver], [hor], [0])
            # import kiui
            # kiui.vis.plot_image(rgbs)

            # get coordinate in texture image
            pose = torch.from_numpy(pose.astype(np.float32)).to(device)
            proj = torch.from_numpy(cam.perspective.astype(np.float32)).to(device)

            v_cam = (
                torch.matmul(F.pad(mesh.v, pad=(0, 1), mode="constant", value=1.0), torch.inverse(pose).T)
                .float()
                .unsqueeze(0)
            )
            v_clip = v_cam @ proj.T
            rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (render_resolution, render_resolution))

            depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f)  # [1, H, W, 1]
            depth = depth.squeeze(0)  # [H, W, 1]

            alpha = (rast[0, ..., 3:] > 0).float()

            uvs, _ = dr.interpolate(mesh.vt.unsqueeze(0), rast, mesh.ft)  # [1, 512, 512, 2] in [0, 1]

            # use normal to produce a back-project mask
            normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
            normal = safe_normalize(normal[0])

            # rotated normal (where [0, 0, 1] always faces camera)
            rot_normal = normal @ pose[:3, :3]
            viewcos = rot_normal[..., [2]]

            mask = (alpha > 0) & (viewcos > 0.5)  # [H, W, 1]
            mask = mask.view(-1)

            uvs = uvs.view(-1, 2).clamp(0, 1)[mask]
            rgbs = rgbs.view(3, -1).permute(1, 0)[mask].contiguous()

            # update texture image
            cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
                h,
                w,
                uvs[..., [1, 0]] * 2 - 1,
                rgbs,
                min_resolution=256,
                return_count=True,
            )

            albedo += cur_albedo
            cnt += cur_cnt

        mask = cnt.squeeze(-1) < 0.1
        albedo[mask] += cur_albedo[mask]
        cnt[mask] += cur_cnt[mask]

        mask = cnt.squeeze(-1) > 0
        albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3)

        mask = mask.view(h, w)

        albedo = albedo.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()

        # dilate texture
        from scipy.ndimage import binary_dilation, binary_erosion
        from sklearn.neighbors import NearestNeighbors

        inpaint_region = binary_dilation(mask, iterations=32)
        inpaint_region[mask] = 0

        search_region = mask.copy()
        not_search_region = binary_erosion(search_region, iterations=3)
        search_region[not_search_region] = 0

        search_coords = np.stack(np.nonzero(search_region), axis=-1)
        inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

        knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(search_coords)
        _, indices = knn.kneighbors(inpaint_coords)

        albedo[tuple(inpaint_coords.T)] = albedo[tuple(search_coords[indices[:, 0]].T)]

        mesh.albedo = torch.from_numpy(albedo).to(device)
        mesh.write(path)

    print(f"[INFO] save model to {path}.")


def render_sets(
    dataset: ModelParams,
    hyperparam,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    skip_video: bool,
    mesh=False,
    orbit=False,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if mesh:
            export_mesh(
                dataset.model_path, "mesh", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background
            )

        if orbit:
            render_orbit(
                dataset.model_path, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background
            )

        if not skip_train:
            ref_time = scene.gaussians._deformation.deformation_net.args.render_ref_time
            scene.gaussians._deformation.deformation_net.args.render_ref_time = -1
            render_set(
                dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background
            )
            scene.gaussians._deformation.deformation_net.args.render_ref_time = ref_time

        if not skip_test:
            render_set(
                dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background
            )
        if not skip_video:
            render_set(
                dataset.model_path, "video", scene.loaded_iter, scene.getVideoCameras(), gaussians, pipeline, background
            )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-2, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--orbit", action="store_true")
    parser.add_argument("--configs", type=str)
    args = get_combined_args(parser)
    print("Rendering ", os.path.abspath(args.model_path))
    if args.configs:
        import mmcv

        from utils.params_utils import merge_hparams

        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        hyperparam.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.skip_video,
        mesh=args.gs2d,
        orbit=args.orbit,
    )
