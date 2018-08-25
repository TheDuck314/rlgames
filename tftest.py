#!/usr/bin/python

import tensorflow as tf
import numpy as np

def main():
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            choice_logits = tf.placeholder(tf.float32, shape=[1, 2], name='choice_logits')
            choice_probs = tf.nn.softmax(choice_logits)

            inputs = np.array([[1.99, 2]])
            print "inputs:\n{}".format(inputs)
            print "inputs.shape: {}".format(inputs.shape)
            feed_dict = {choice_logits: inputs}
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
            results = sess.run(choice_probs, feed_dict)
            print "results: {}".format(results)

if __name__ == "__main__":
    main()
