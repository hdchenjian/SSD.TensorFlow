from utility import anchor_manipulator

out_shape = [300] * 2

anchor_creator = anchor_manipulator.AnchorCreator(
    out_shape,
    layers_shapes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
    anchor_scales = [(0.1,), (0.2,), (0.375,), (0.55,), (0.725,), (0.9,)],
    extra_anchor_scales = [(0.1414,), (0.2739,), (0.4541,), (0.6315,), (0.8078,), (0.9836,)],
    anchor_ratios = [(1., 2., .5), (1., 2., 3., .5, 0.3333), (1., 2., 3., .5, 0.3333),
                     (1., 2., 3., .5, 0.3333), (1., 2., .5), (1., 2., .5)],
    #anchor_ratios = [(2., .5), (2., 3., .5, 0.3333), (2., 3., .5, 0.3333),
    #(2., 3., .5, 0.3333), (2., .5), (2., .5)],
    layer_steps = [8, 16, 32, 64, 100, 300])
all_anchors, all_num_anchors_depth, all_num_anchors_spatial = anchor_creator.get_all_anchors()

anchor_encoder_decoder = anchor_manipulator.AnchorEncoder(allowed_borders = [1.0] * 6,
                                                          positive_threshold = None,
                                                          ignore_threshold = None,
                                                          prior_scaling=[0.1, 0.1, 0.2, 0.2])
