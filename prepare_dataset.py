import argparse
import json
import os
from glob import glob
from math import cos, radians, sin

import cv2
import imageio
import numpy as np
from tqdm import tqdm


def extract_video_keyframes(video_path: str, output_dir: str = None, frame_interval=1, save=True):
    if save:
        os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    # rate = cap.get(5)
    # frame_num = cap.get(7)
    frame_index = 0
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if frame_index % frame_interval == 0 and not (frame == 0).all():
                frames.append(frame)
                if save:
                    cv2.imwrite(os.path.join(output_dir, f"{frame_index:06d}.jpg"), frame)
            frame_index += 1
        else:
            break
    cap.release()
    return frames


def cmd_output(cmd):
    with os.popen(cmd) as result:
        context = result.read()
        lines = context.splitlines()
    return "\n".join(lines)


def process_videos(
    data_dir: str,
    raw_subdir="raw",
    interp=False,
    interp_subdir="interp",
    interp_ratio=4,
    rife_dir="rife-ncnn-vulkan",
    colmap=False,
    verbose=False,
):
    frame_dirs = [f.replace(".mp4", "") for f in sorted(glob(os.path.join(data_dir, "00*.mp4")))]
    for frame_dir in tqdm(frame_dirs, desc="Processing videos", dynamic_ncols=True, leave=True):
        input_video = f"{frame_dir}.mp4"
        input_dir = os.path.join(frame_dir, raw_subdir)
        input_frames_pattern = os.path.join(input_dir, "frame_%08d.png")
        interp_dir = os.path.join(frame_dir, interp_subdir)
        interp_frames_pattern = os.path.join(interp_dir, "%08d.png")
        interp_video = os.path.join(interp_dir, "interp.mp4")
        colmap_dir = os.path.join(frame_dir, "colmap")
        # os.system(f"rm -r {frame_dir}")
        os.makedirs(input_dir, exist_ok=True)

        # decode all frames
        extract_cmd = f"ffmpeg -i {input_video} {input_frames_pattern}"
        ffmpeg_quiet = " -hide_banner -loglevel error"
        if not verbose:
            extract_cmd += ffmpeg_quiet
        os.system(extract_cmd)

        if interp:
            os.makedirs(interp_dir)
            ffprobe_cmd = f"ffprobe -show_streams {input_video} 2>&1"
            ffprobe_cmd += """ | grep fps | awk '{split($0,a,"fps")}END{print a[1]}' | awk '{print $NF}'"""
            fps = int(cmd_output(ffprobe_cmd))
            frame_count = int(
                cmd_output(
                    f'ffmpeg -i {input_video} -v trace -hide_banner 2>&1 | grep -A 10 codec_type=0 | grep -oP "(?<=sample_count=)\d+"'
                )
            )

            interp_cmd = f"{os.path.join(rife_dir, 'rife-ncnn-vulkan')} -i {input_dir} -o {interp_dir} -m rife-v4.6 -n {frame_count * interp_ratio}"
            if not verbose:
                interp_cmd += ">/dev/null 2>&1"
            os.system(interp_cmd)

            encode_cmd = f"ffmpeg -framerate {fps * interp_ratio} -i {interp_frames_pattern} -c:a copy -crf 20 -c:v libx264 -pix_fmt yuv420p {interp_video}"
            if not verbose:
                encode_cmd += ffmpeg_quiet
            os.system(encode_cmd)

        if colmap:
            colmap_input_dir = interp_dir if interp else input_dir
            os.makedirs(colmap_dir)
            colmap_cmd = f"colmap automatic_reconstructor --image_path {colmap_input_dir} --workspace_path {colmap_dir} --dense=0 --single_camera=1"
            if not verbose:
                colmap_cmd += ">/dev/null 2>&1"
            os.system(colmap_cmd)
            # os.system(f"ns-process-data images --data {colmap_input_dir} --output-dir {frame_dir} --skip-colmap")


def generate_transform_matrix(num_view=21, elevation_deg=10, r=4):
    theta = radians(elevation_deg)  # 转换为弧度

    output = []
    for i in range(0, num_view):
        # 方位角均分，计算每份的角度增量
        phi_increment = 360 / num_view
        phi_deg = i * phi_increment  # 第一个方位角
        phi = radians(phi_deg)  # 转换为弧度

        # 计算球面坐标系中的坐标
        x = r * cos(theta) * cos(phi)
        y = r * cos(theta) * sin(phi)
        z = r * sin(theta)

        # 相机位置向量
        camera_position = np.array([x, y, z])

        # 为了计算旋转矩阵，定义球心为原点
        origin = np.array([0, 0, 0])

        # 计算从相机位置指向原点的向量
        look_direction = origin - camera_position  # look-at
        look_direction = look_direction / np.linalg.norm(look_direction)  # 单位化
        z = -look_direction
        # 假设上方向为世界坐标系的Z轴
        up_direction = np.array([0, 0, 1])

        # 计算右向量
        right_direction = np.cross(up_direction, z)
        right_direction = right_direction / np.linalg.norm(right_direction)  # 单位化

        # 重新计算正确的上方向
        up_direction = np.cross(z, right_direction)  # z, x --> y

        # 构建旋转矩阵
        rotation_matrix = np.vstack([right_direction, up_direction, z]).T

        # 构建4x4变换矩阵
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = camera_position

        output.append(transform_matrix.tolist())
    return output


def create_meta(
    frame_list,
    data_dir,
    image_subdir,
    elevation=0.0,
    organize_by_view=True,
    image_ext="png",
    view_subdir="views",
    restore_subdir="restore",
):
    view_list = sorted(glob(os.path.join(frame_list[0], image_subdir, f"*.{image_ext}")))
    view_list = [os.path.basename(p) for p in view_list]
    time_list = list(np.linspace(0, 1, len(frame_list)))

    # # Get transform_matrix from colmap-based nerfstudio transforms.json
    # # nerfstudio_json_path = os.path.join(frame_list[0], "transforms.json")
    # nerfstudio_json_path = os.path.join(args.output_dir, "transforms.json")
    # with open(nerfstudio_json_path, "r") as f:
    #     meta = json.load(f)
    # views = meta["frames"]
    # views.sort(key=lambda x: x["file_path"])
    # assert len(view_list) == len(views)
    transform_matrix_list = generate_transform_matrix(num_view=len(view_list), elevation_deg=elevation)

    test_views = (0,)
    data_train = []
    data_test = []
    for i_frame, frame in enumerate(frame_list):
        for i_view, view in enumerate(view_list):
            image_path = os.path.join(frame, image_subdir, view)
            if organize_by_view:
                frame_id = os.path.basename(frame)
                view_id = view.replace(f".{image_ext}", "").replace("frame_", "")
                image_path_new = os.path.join(data_dir, view_subdir, view_id, "images", f"{frame_id}.{image_ext}")
                soft_link(image_path, image_path_new, relpath=True)
                image_path = image_path_new
            assert os.path.isfile(image_path), image_path
            restore_path = os.path.join(frame, restore_subdir, view)
            # assert os.path.isfile(restore_path), restore_path
            data_dict = {
                "file_path": os.path.splitext(os.path.relpath(image_path, data_dir))[0],
                "restore_path": (
                    os.path.splitext(os.path.relpath(restore_path, data_dir))[0]
                    if os.path.isfile(restore_path)
                    else None
                ),
                "time": time_list[i_frame],
                "time_index": i_frame,
                "view_index": i_view,
                "transform_matrix": transform_matrix_list[i_view],
            }
            if i_view in test_views:
                data_test.append(data_dict)
            else:
                data_train.append(data_dict)
    data_train.sort(key=lambda x: x["file_path"])
    data_test.sort(key=lambda x: x["file_path"])
    return data_train, data_test


def soft_link(src, dst, relpath=False):
    if os.path.exists(dst) or os.path.islink(dst):
        os.remove(dst)
    dst_dir = os.path.dirname(dst)
    os.makedirs(dst_dir, exist_ok=True)
    if relpath:
        src = os.path.relpath(src, os.path.dirname(dst))
    os.system(f"ln -s {src} {dst}")
    assert os.path.islink(dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="data_train/tiger/tiger3/dynamic.mp4")
    parser.add_argument("--output_dir", type=str, default="data_train/tiger/tiger3_ref_s0.5")
    parser.add_argument("--gm_path", type=str, default="generative-models")
    parser.add_argument("--model", type=str, default="sv3d_p")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--elevation", type=float, default=0.0)
    parser.add_argument("--remove_bg", action="store_true", default=False)
    parser.add_argument("--ref", action="store_true", default=False)
    parser.add_argument("--ref_strength", type=float, default=0.5)
    parser.add_argument("--ref_ema", action="store_true", default=False)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--skip_sv3d", action="store_true", default=False)
    parser.add_argument("--skip_process_videos", action="store_true", default=False)
    parser.add_argument("--interp", action="store_true", default=False)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    DATA_SUBDIR = "multiviews"
    RAW_SUBDIR = "raw"
    INTERP_SUBDIR = "interp"
    VIEW_SUBDIR = "views"
    RESTORE_SUBDIR = "restore/0.177"
    IMAGE_EXT = "png"
    assert os.path.isfile(args.video_path)
    os.makedirs(args.output_dir, exist_ok=True)
    data_dir = os.path.join(args.output_dir, DATA_SUBDIR)

    ### Run SV3D for each frame
    if not args.skip_sv3d:
        os.system(f"cp -a {args.video_path} {os.path.join(args.output_dir, 'dynamic.mp4')}")
        extract_video_keyframes(args.video_path, data_dir)
        cmd = f"cd {args.gm_path} && python scripts/sampling/simple_video_sample.py"
        cmd += f" --input_path {os.path.abspath(data_dir)} --output_folder {os.path.abspath(data_dir)} --version {args.model} --decoding_t 1 --elevations_deg {float(args.elevation)}"
        if args.seed is not None:
            cmd += f" --seed {args.seed}"
        if not args.remove_bg:
            cmd += " --keep_bg"
        if args.ref:
            cmd += " --ref"
        if args.ref_ema:
            cmd += " --ref_ema"
        cmd += f" --ref_strength {args.ref_strength}"
        print(cmd)
        os.system(cmd)

    # Generate pseudo 4D video
    frame_list = sorted(glob(os.path.join(data_dir, "00*.mp4")))
    assert len(frame_list) > 0
    frames = [extract_video_keyframes(vid_path, save=False) for vid_path in frame_list]
    repeat = 5
    imageio.mimwrite(
        os.path.join(args.output_dir, "4d.mp4"),
        [
            cv2.cvtColor(frames[i % len(frames)][i // 2 % len(frames[0])], cv2.COLOR_BGR2RGB)
            for i in range(len(frames) * repeat)
        ],
        fps=args.fps,
    )
    imageio.mimwrite(
        os.path.join(args.output_dir, "back.mp4"),
        [
            cv2.cvtColor(frames[i % len(frames)][len(frames[0]) // 2], cv2.COLOR_BGR2RGB)
            for i in range(len(frames) * repeat)
        ],
        fps=args.fps,
    )

    ### Extract images from SV3D results (and optionally perform interpolation & COLMAP)
    if not args.skip_process_videos:
        process_videos(
            data_dir,
            raw_subdir=RAW_SUBDIR,
            interp=args.interp,
            interp_subdir=INTERP_SUBDIR,
            colmap=False,
            verbose=False,
        )

    ### Create D-NeRF like dataset structure
    data_train, data_test = create_meta(
        [f.replace(".mp4", "") for f in frame_list],
        data_dir,
        image_subdir=INTERP_SUBDIR if args.interp else RAW_SUBDIR,
        elevation=args.elevation,
        organize_by_view=True,
        image_ext=IMAGE_EXT,
        view_subdir=VIEW_SUBDIR,
        restore_subdir=RESTORE_SUBDIR,
    )
    camera_angle_x = 0.6911112070083618
    with open(os.path.join(data_dir, "transforms_train.json"), "w") as f:
        json.dump({"camera_angle_x": camera_angle_x, "frames": data_train}, f, indent=4, ensure_ascii=False)
    with open(os.path.join(data_dir, "transforms_test.json"), "w") as f:
        json.dump({"camera_angle_x": camera_angle_x, "frames": data_test}, f, indent=4, ensure_ascii=False)
    soft_link("transforms_test.json", os.path.join(data_dir, "transforms_val.json"), relpath=False)

    print(f"Done: {data_dir}")
