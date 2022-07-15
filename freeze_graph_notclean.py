#tf 2.6

import argparse
import string
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np
import time
import keras
import os

def freeze_model_tf(model_dir, name_pb):
    m = tf.saved_model.load(model_dir)
    tfm = tf.function(lambda x: m(inputs=x))  # full model
    tfm = tfm.get_concrete_function(tf.TensorSpec(name="input:0",
                                                  shape=[None, h, w, c],
                                                  dtype=tf.float32))
    frozen_func = convert_variables_to_constants_v2(tfm)
    output_freeze_pb = os.path.join(model_dir, name_pb)
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=model_dir,
                      name=output_freeze_pb,
                      as_text=False)
    return


if __name__ == "__main__":
    saved_dir = 'saved_dir' # saved_format
    name_pb = "./output.pb" # output
    freeze_model_tf(saved_dir, name_pb)
