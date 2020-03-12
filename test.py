import tensorflow as tf
from imageio import imread, imsave
from utility import draw_toolbox

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
      graph_def.ParseFromString(f.read())
    with graph.as_default():
      tf.import_graph_def(graph_def)

    return graph

model_file = 'model/ssd300_vgg16.pb'
graph = load_graph(model_file)
#for tensor in tf.get_default_graph().as_graph_def().node: print(tensor.name)

with tf.Session(graph = graph) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    #for op in tf.get_default_graph().get_operations(): print(str(op.name))

    image_input = graph.get_tensor_by_name('import/define_input/image_input:0')
    all_labels = graph.get_tensor_by_name("import/all_labels/concat:0" )
    all_scores = graph.get_tensor_by_name("import/all_scores/concat:0" )
    all_bboxes = graph.get_tensor_by_name("import/all_bboxes/concat:0" )
    
    np_image = imread('demo/test.jpg')
    labels_, scores_, bboxes_ = sess.run([all_labels, all_scores, all_bboxes], feed_dict = {image_input : np_image})
    #print('labels_', labels_, type(labels_), labels_.shape)
    #print('scores_', scores_, type(scores_), scores_.shape)
    #print('bboxes_', bboxes_, type(bboxes_), bboxes_.shape, bboxes_.shape[0])

    img_to_draw = draw_toolbox.bboxes_draw_on_img(np_image, labels_, scores_, bboxes_, thickness=2)
    imsave('demo/test_out.jpg', img_to_draw)
