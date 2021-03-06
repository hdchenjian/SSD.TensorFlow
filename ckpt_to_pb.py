import tensorflow as tf

def freeze_graph(input_checkpoint,output_graph):
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    for tensor in tf.get_default_graph().as_graph_def().node: print(tensor.name)
 
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=['ssd300/cls_pred/concat', 'ssd300/location_pred/concat'])
 
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))
 

#freeze_graph('model/SSD300-VGG16/model.ckpt-120000', 'model/ssd_vgg16')
freeze_graph('model/ssd300_vgg16/ssd300_vgg16_short-0', 'model/ssd300_vgg16_short.pb')
