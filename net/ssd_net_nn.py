from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class VGG16Backbone(object):
    def __init__(self, _input_data, _num_classes, _num_anchors_depth_per_layer, _data_format='channels_first'):
        super(VGG16Backbone, self).__init__()
        self.print_tensor_shape = False
        if _data_format == 'channels_first': self.data_format = 'NCHW'
        else: self.data_format = 'NHWC'
        self.input_data = _input_data
        self.num_classes = _num_classes
        self.num_anchors_depth_per_layer = _num_anchors_depth_per_layer
        
        # VGG layers
        self._conv1_block = self.conv_block(self.input_data, 2, 64, 3, 3, [1, 1, 1, 1], 'conv1')
        self._pool1 = tf.nn.max_pool(self._conv1_block, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                     data_format=self.data_format, name='pool1')
        self._conv2_block = self.conv_block(self._pool1, 2, 128, 3, 64, [1, 1, 1, 1], 'conv2')
        self._pool2 = tf.nn.max_pool(self._conv2_block, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                     data_format=self.data_format, name='pool2')
        self._conv3_block = self.conv_block(self._pool2, 3, 256, 3, 128, [1, 1, 1, 1], 'conv3')
        self._pool3 = tf.nn.max_pool(self._conv3_block, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                     data_format=self.data_format, name='pool3')
        self._conv4_block = self.conv_block(self._pool3, 3, 512, 3, 256, [1, 1, 1, 1], 'conv4')
        self._pool4 = tf.nn.max_pool(self._conv4_block, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                     data_format=self.data_format, name='pool4')
        self._conv5_block = self.conv_block(self._pool4, 3, 512, 3, 512, [1, 1, 1, 1], 'conv5')
        self._pool5 = tf.nn.max_pool(self._conv5_block, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME',
                                     data_format=self.data_format, name='pool5')
        self._conv6 = self.conv_block(self._pool5, 1, 1024, 3, 512, [1, 1, 1, 1], name='fc6', dilations=6)
        self._conv7 = self.conv_block(self._conv6, 1, 1024, 1, 1024, [1, 1, 1, 1], name='fc7')

        # SSD layers
        with tf.variable_scope('additional_layers') as scope:
            self._conv8_block = self.ssd_conv_block(self._conv7, 256, 1024, [1, 2, 2, 1], 'conv8')
            self._conv9_block = self.ssd_conv_block(self._conv8_block, 128, 512, [1, 2, 2, 1], 'conv9')
            self._conv10_block = self.ssd_conv_block(
                self._conv9_block, 128, 256, [1, 1, 1, 1], 'conv10', padding='VALID')
            self._conv11_block = self.ssd_conv_block(
                self._conv10_block, 128, 256, [1, 1, 1, 1], 'conv11', padding='VALID')

        # conv4_3
        with tf.variable_scope('conv4_3_scale') as scope:
            weight_scale = tf.Variable([20.] * 512, trainable=False, name='weights')
            weight_scale = tf.reshape(weight_scale, [1, 1, 1, -1], name='reshape')
            self._feature1 = tf.multiply(
                weight_scale, self.l2_normalize(self._conv4_block, name='norm'), name='rescale')

        feature_layers = [self._feature1, self._conv7, self._conv8_block,
                          self._conv9_block,  self._conv10_block,  self._conv11_block]
        feature_layers_pre_channel_size = [512, 1024, 512, 256, 256, 256]
        self.cls_preds = []
        self.loc_preds = []
        for ind, feat in enumerate(feature_layers):
            _loc, _cls = self.get_predication(ind, feat, feature_layers_pre_channel_size[ind])
            self.loc_preds.append(_loc)
            self.cls_preds.append(_cls)
        
    def l2_normalize(self, x, name):
        with tf.name_scope(name, "l2_normalize", [x]) as name:
            axis = -1 if self.data_format == 'channels_last' else 1
            square_sum = tf.reduce_sum(tf.square(x), axis, keep_dims=True)
            x_inv_norm = tf.rsqrt(tf.maximum(square_sum, 1e-10))
            return tf.multiply(x, x_inv_norm, name=name)

    def weight_variable(self, shape, name='name'):
        with tf.variable_scope(name):
            initial = tf.glorot_uniform_initializer(seed=None, dtype=tf.float32)
            return tf.Variable(initial(shape), name='kernel')

    def bias_variable(self, shape, name='name'):
        print('bias_variable', name)
        with tf.variable_scope(name):
            initial = tf.glorot_uniform_initializer(seed=None, dtype=tf.float32)
            return tf.Variable(initial(shape), name='bias')

    #self._conv1_block = self.conv_block(self.input_data, 2, 64, 3, 3, [1, 1, 1, 1], 'conv1')
    def conv_block(self, bottom, num_blocks, filters, kernel_size, pre_channel_size, strides, name, dilations=None):
        with tf.variable_scope(name):
            _input = bottom
            for ind in range(1, num_blocks + 1):
                w = self.weight_variable([kernel_size, kernel_size, pre_channel_size, filters],
                                         name='{}_{}'.format(name, ind))
                b = self.bias_variable([filters], name='{}_{}'.format(name, ind))
                conv = tf.nn.relu(tf.nn.conv2d(
                    _input, w, strides, padding='SAME', data_format=self.data_format, dilations=dilations,
                    name='{}_{}'.format(name, ind)) + b)
                _input = conv
                pre_channel_size = filters
            return _input

    #self._conv8_block = self.ssd_conv_block(self._conv7, 256, 1024, [1, 2, 2, 1], 'conv8')
    def ssd_conv_block(self, bottom, filters, pre_channel_size, strides, name, padding='SAME'):
        print('ssd_conv_block', name)
        with tf.variable_scope(name):
            w = self.weight_variable([1, 1, pre_channel_size, filters], name='{}_{}'.format(name, 1))
            b = self.bias_variable([filters], name='{}_{}'.format(name, 1))
            conv = tf.nn.relu(tf.nn.conv2d(
                bottom, w, [1, 1, 1, 1], padding=padding, data_format=self.data_format,
                name='{}_{}'.format(name, 1)) + b)

            pre_channel_size = filters
            w2 = self.weight_variable([3, 3, pre_channel_size, filters * 2], name='{}_{}'.format(name, 2))
            b2 = self.bias_variable([filters * 2], name='{}_{}'.format(name, 2))
            conv2 = tf.nn.relu(tf.nn.conv2d(
                conv, w2, strides, padding=padding, data_format=self.data_format,
                name='{}_{}'.format(name, 2)) + b2)
            return conv2

    def get_predication(self, ind, feat, pre_channel_size):
        with tf.variable_scope('multibox_head'):
            w = self.weight_variable([3, 3, pre_channel_size, self.num_anchors_depth_per_layer[ind] * 4],
                                     name='loc_{}'.format(ind))
            b = self.bias_variable([self.num_anchors_depth_per_layer[ind] * 4], name='loc_{}'.format(ind))
            loc = tf.nn.conv2d(feat, w, [1, 1, 1, 1], padding='SAME', data_format=self.data_format,
                                name='loc_{}'.format(ind)) + b

            w2 = self.weight_variable([3, 3, pre_channel_size,
                                       self.num_anchors_depth_per_layer[ind] * self.num_classes],
                                      name='cls_{}'.format(ind))
            b2 = self.bias_variable([self.num_anchors_depth_per_layer[ind] * self.num_classes],
                                    name='cls_{}'.format(ind))
            cls = tf.nn.conv2d(feat, w2, [1, 1, 1, 1], padding='SAME', data_format=self.data_format,
                                name='cls_{}'.format(ind)) + b2
            return  loc, cls
