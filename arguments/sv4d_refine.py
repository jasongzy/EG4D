_base_ = "sv4d.py"

OptimizationParams = dict(
    iterations=35000,
    # densify_until_iter=40000,
    # densification_interval=200,
    # pruning_interval=500,
    # opacity_reset_interval=3000,
    # pruning_from_iter=99999,
    # opacity_reset_interval=99999,
)

ModelHiddenParams = dict(
    do=True,
)