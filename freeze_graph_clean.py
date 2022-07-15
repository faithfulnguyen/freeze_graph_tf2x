#tf 2.6
import argparse
import string
import tensorflow as tf
import numpy as np
import time
import keras
from tensorflow.keras import Model, layers
import os

import config

def define_model(img_h, img_w, channel):
    epsilon_batchnorm = 0.0001
    input_images = layers.Input(shape=[img_h,
                                       img_w,
                                       channel], name='input', dtype=tf.float32)
    # def arch here
    net = layers.Conv2D(32, kernel_size=5, activation=tf.nn.relu, strides=(2, 2),
                            kernel_regularizer=tf.keras.regularizers.l2(l=5e-4),
                            padding="SAME", name='conv_1')(input_images)
    net = layers.BatchNormalization(axis=-1, epsilon=epsilon_batchnorm)(net)
    net = tf.reduce_mean(net, axis=[1, 2], keepdims=False)
    net = layers.Dense(10, activation=None, name='fc_embbed')(net)
    

    logits_sf = tf.nn.softmax(net, axis=-1, name="sf")
    logits_sf = tf.identity(logits_sf, name="sf")
    model_obj = tf.keras.Model(inputs=[input_images], outputs=[logits_sf])
    return model_obj

def freeze_from_h5file_clean_graph(file_model, file_out):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    import tensorflow.keras.backend as K

    def freeze_graph_def(sess, output_node_names):
        graph = sess.graph
        with graph.as_default():
            input_graph_def = graph.as_graph_def()
            for node in input_graph_def.node:
                node.device = '' # clean device
                if node.op == 'RefSwitch':
                    node.op = 'Switch'
                    for index in range(len(node.input)):
                        if 'moving_' in node.input[index]:
                            node.input[index] = node.input[index] + '/read'
                elif node.op == 'AssignSub':
                    node.op = 'Sub'
                    if 'use_locking' in node.attr: del node.attr['use_locking']
                elif node.op == 'AssignAdd':
                    node.op = 'Add'
                    if 'use_locking' in node.attr: del node.attr['use_locking']

            # Get the list of important nodes
            whitelist_names = []
            for node in input_graph_def.node:
                if (node.name.startswith('conv_1') or node.name.startswith('fc_embbed')):
                    whitelist_names.append(node.name)

            # Replace all the variables in the graph with constants of the same values
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, input_graph_def, output_node_names.split(","),
                variable_names_whitelist=whitelist_names)
            return output_graph_def

    K.clear_session()
    K.set_learning_phase(False)

    m = define_model(img_h=224, img_w=224, channel=3)
    m.load_weights(file_model)

    frozen_graph = freeze_graph_def(tf.keras.backend.get_session(),
                                    output_node_names="sf")
    current_dir = os.path.split(file_model)[0]
    tf.train.write_graph(frozen_graph, current_dir, file_out, as_text=False)


if __name__ == "__main__":
    h5_model = 'model.h5' # h5 format
    name_pb = "model.pb" # output
    freeze_from_h5file_clean_graph(h5_model, name_pb)
