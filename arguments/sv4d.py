_base_ = "dnerf/dnerf_default.py"

ModelHiddenParams = dict(
    kplanes_config={
        "grid_dimensions": 2,
        "input_coordinate_dim": 4,
        "output_coordinate_dim": 32,
        "resolution": [64, 64, 64, 25],
    },
    multires=[1, 2, 4, 8],
    defor_depth=2,
    net_width=256,
    plane_tv_weight=0.0002,
    time_smoothness_weight=0.001,
    l1_time_planes=0.001,
    predict_color_deform=True,
)

OptimizationParams = dict(
    dataloader=True,
    coarse_iterations=6000,
    iterations=30000,
    # densify_until_iter = 45_000,
    # opacity_reset_interval = 6000,
    # opacity_threshold_coarse = 0.005,
    # opacity_threshold_fine_init = 0.005,
    # opacity_threshold_fine_after = 0.005,
    # # pruning_interval = 2000
)

PipelineParams = dict(convert_SHs_python=True)
