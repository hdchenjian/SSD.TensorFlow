import tensorflow as tf
from utility import draw_toolbox
from utility import anchor_manipulator
import cv2
import numpy as np

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
      graph_def.ParseFromString(f.read())
    with graph.as_default():
      tf.import_graph_def(graph_def)

    return graph

model_file = 'model/ssd300_vgg16_short_mlu.pb'
graph = load_graph(model_file)
#for tensor in tf.get_default_graph().as_graph_def().node: print(tensor.name)

config = tf.ConfigProto(allow_soft_placement=True, inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
config.mlu_options.data_parallelism = 1
config.mlu_options.model_parallelism = 1
config.mlu_options.core_num = 1
config.mlu_options.core_version = 'MLU270'
config.mlu_options.precision = 'int8'

with tf.Session(config = config, graph = graph) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    #for op in tf.get_default_graph().get_operations(): print(str(op.name))
    for tensor in tf.get_default_graph().as_graph_def().node: print(tensor.name)

    image_input = graph.get_tensor_by_name('import/define_input/image_input:0')
    cls_pred = graph.get_tensor_by_name("import/ssd300/cls_pred/concat:0" )
    location_pred = graph.get_tensor_by_name("import/ssd300/location_pred/concat:0" )

    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    means = [_B_MEAN, _G_MEAN, _R_MEAN, ]
    np_image = cv2.imread('demo/test.jpg')
    np_image = np.float32(np_image)
    image = cv2.resize(np_image, (300, 300))
    #cv2.imwrite('demo/test_out2.jpg', image)
    image = (image - means)# / 255.0
    #print('image', type(image), image.shape, image)
    image = np.expand_dims(image, axis=0)
    #print('image', type(image), image.shape, image)

    cls_pred_, location_pred_ = sess.run([cls_pred, location_pred], feed_dict = {image_input : image})
    print('cls_pred', type(cls_pred_), cls_pred_.shape)
    print('location_pred', type(location_pred_), location_pred_.shape)


g2 = tf.Graph()
with g2.as_default():
    with tf.device('/cpu:0'):
        def select_bboxes(scores_pred, bboxes_pred, num_classes, select_threshold):
            selected_bboxes = {}
            selected_scores = {}
            with tf.name_scope('select_bboxes', values = [scores_pred, bboxes_pred]):
                for class_ind in range(1, num_classes):
                    class_scores = scores_pred[:, class_ind]
        
                    select_mask = class_scores > select_threshold
                    select_mask = tf.cast(select_mask, tf.float32)
                    selected_bboxes[class_ind] = tf.multiply(bboxes_pred, tf.expand_dims(select_mask, axis=-1))
                    selected_scores[class_ind] = tf.multiply(class_scores, select_mask)
        
            return selected_bboxes, selected_scores
        
        def clip_bboxes(ymin, xmin, ymax, xmax, name):
            with tf.name_scope(name, 'clip_bboxes', [ymin, xmin, ymax, xmax]):
                ymin = tf.maximum(ymin, 0.)
                xmin = tf.maximum(xmin, 0.)
                ymax = tf.minimum(ymax, 1.)
                xmax = tf.minimum(xmax, 1.)
        
                ymin = tf.minimum(ymin, ymax)
                xmin = tf.minimum(xmin, xmax)
        
                return ymin, xmin, ymax, xmax
        
        def filter_bboxes(scores_pred, ymin, xmin, ymax, xmax, min_size, name):
            with tf.name_scope(name, 'filter_bboxes', [scores_pred, ymin, xmin, ymax, xmax]):
                width = xmax - xmin
                height = ymax - ymin
        
                filter_mask = tf.logical_and(width > min_size, height > min_size)
        
                filter_mask = tf.cast(filter_mask, tf.float32)
                return tf.multiply(ymin, filter_mask), tf.multiply(xmin, filter_mask), \
                    tf.multiply(ymax, filter_mask), tf.multiply(xmax, filter_mask), tf.multiply(scores_pred, filter_mask)
        
        def sort_bboxes(scores_pred, ymin, xmin, ymax, xmax, keep_topk, name):
            with tf.name_scope(name, 'sort_bboxes', [scores_pred, ymin, xmin, ymax, xmax]):
                cur_bboxes = tf.shape(scores_pred)[0]
                scores, idxes = tf.nn.top_k(scores_pred, k=tf.minimum(keep_topk, cur_bboxes), sorted=True)
        
                ymin, xmin, ymax, xmax = tf.gather(ymin, idxes), tf.gather(xmin, idxes), \
                                         tf.gather(ymax, idxes), tf.gather(xmax, idxes)
        
                paddings_scores = tf.expand_dims(tf.stack([0, tf.maximum(keep_topk-cur_bboxes, 0)], axis=0), axis=0)
        
                return tf.pad(ymin, paddings_scores, "CONSTANT"), tf.pad(xmin, paddings_scores, "CONSTANT"),\
                        tf.pad(ymax, paddings_scores, "CONSTANT"), tf.pad(xmax, paddings_scores, "CONSTANT"),\
                        tf.pad(scores, paddings_scores, "CONSTANT")
        
        def nms_bboxes(scores_pred, bboxes_pred, nms_topk, nms_threshold, name):
            with tf.name_scope(name, 'nms_bboxes', [scores_pred, bboxes_pred]):
                idxes = tf.image.non_max_suppression(bboxes_pred, scores_pred, nms_topk, nms_threshold)
                return tf.gather(scores_pred, idxes), tf.gather(bboxes_pred, idxes)
        
        def parse_by_class(cls_pred, bboxes_pred, num_classes, select_threshold, min_size,
                           keep_topk, nms_topk, nms_threshold):
            with tf.name_scope('select_bboxes', values = [cls_pred, bboxes_pred]):
                scores_pred = tf.nn.softmax(cls_pred)
                selected_bboxes, selected_scores = select_bboxes(scores_pred, bboxes_pred, num_classes, select_threshold)
                for class_ind in range(1, num_classes):
                    ymin, xmin, ymax, xmax = tf.unstack(selected_bboxes[class_ind], 4, axis=-1)
                    #ymin, xmin, ymax, xmax = tf.squeeze(ymin), tf.squeeze(xmin), tf.squeeze(ymax), tf.squeeze(xmax)
                    ymin, xmin, ymax, xmax = clip_bboxes(ymin, xmin, ymax, xmax, 'clip_bboxes_{}'.format(class_ind))
                    ymin, xmin, ymax, xmax, selected_scores[class_ind] = filter_bboxes(
                        selected_scores[class_ind], ymin, xmin, ymax, xmax, min_size, 'filter_bboxes_{}'.format(class_ind))
                    ymin, xmin, ymax, xmax, selected_scores[class_ind] = sort_bboxes(
                        selected_scores[class_ind], ymin, xmin, ymax, xmax, keep_topk, 'sort_bboxes_{}'.format(class_ind))
                    selected_bboxes[class_ind] = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
                    selected_scores[class_ind], selected_bboxes[class_ind] = nms_bboxes(
                        selected_scores[class_ind], selected_bboxes[class_ind], nms_topk,
                        nms_threshold, 'nms_bboxes_{}'.format(class_ind))
        
                return selected_bboxes, selected_scores
        
        
        
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
        
        def decode_fn(pred):
            return anchor_encoder_decoder.ext_decode_all_anchors(
                pred, all_anchors, all_num_anchors_depth, all_num_anchors_spatial)

        with tf.name_scope('g2_cls_pred'):
            g2_cls_pred = tf.placeholder(tf.float32, shape=(8732, 21), name='g2_cls_pred')
        with tf.name_scope('g2_location_pred'):
            g2_location_pred = tf.placeholder(tf.float32, shape=(8732, 4), name='g2_location_pred')
        bboxes_pred = decode_fn(g2_location_pred)
        bboxes_pred = tf.concat(bboxes_pred, axis=0)
        num_classes = 21
        select_threshold = 0.2
        min_size = 0.03
        keep_topk = 200
        nms_topk = 20
        nms_threshold = 0.45
        selected_bboxes, selected_scores = parse_by_class(g2_cls_pred, bboxes_pred,
                                                        num_classes, select_threshold, min_size,
                                                        keep_topk, nms_topk, nms_threshold)

        labels_list = []
        scores_list = []
        bboxes_list = []
        for k, v in selected_scores.items():
            labels_list.append(tf.ones_like(v, tf.int32) * k)
            scores_list.append(v)
            bboxes_list.append(selected_bboxes[k])
        all_labels = tf.concat(labels_list, axis=0)
        all_scores = tf.concat(scores_list, axis=0)
        all_bboxes = tf.concat(bboxes_list, axis=0)

print('sess2 start')
with tf.Session(graph=g2) as sess2:
    print('sess2 end')
    labels_, scores_, bboxes_ = sess2.run([all_labels, all_scores, all_bboxes],
                                          feed_dict = {g2_cls_pred: cls_pred_, g2_location_pred: location_pred_})
    #print('labels_', labels_, type(labels_), labels_.shape)
    #print('scores_', scores_, type(scores_), scores_.shape)
    #print('bboxes_', bboxes_, type(bboxes_), bboxes_.shape, bboxes_.shape[0])

    img_to_draw = draw_toolbox.bboxes_draw_on_img(np_image, labels_, scores_, bboxes_, thickness=2)
    cv2.imwrite('demo/test_out.jpg', img_to_draw)

