"""
This file is designed for testing the speed of the speed-up strategy.
Since it is not easy to modify the state size of the road network, here I emulate the computation pass of CSSRNN model
with every thing the same as the one in `trajmodel.py` except the input data is feeded by meaningless random tensors.
Note that in such a way, we can only test the forward pass of the neural net.
"""

from __future__ import print_function
import numpy as np
import tensorflow as tf
import time

try: # python2
    import ConfigParser as configparser
except ImportError: # python3
    import configparser
import sys
if sys.version > '3':
    PY3 = True
else:
    PY3 = False

class Config:
  state_size = 60000
  batch_size = 100
  max_seq_len = 80
  emb_dim = 400
  hidden_dim = 600

  def __init__(self):
    return

class CSSRNN:
  def __init__(self, config):
    self.config = config
    adj_mat_array = np.random.randint(0, config.state_size-1, size=[config.state_size, 6])
    self.adj_mat = tf.constant(adj_mat_array, dtype=tf.int32)
    self.adj_mask = tf.ones([config.state_size, 6], dtype=tf.float32)
    self.target = tf.placeholder(tf.int32, [config.batch_size, config.max_seq_len], name="target") #[b,t]
    self.sub_target = tf.placeholder(tf.float32, name="sub_target") #[b,t,6], one-hot
    self.input = tf.placeholder(tf.int32, [config.batch_size, config.max_seq_len], name="input") # [b,t]
    self.w = tf.Variable(np.random.uniform(-0.05, 0.05, size=[config.state_size, config.hidden_dim]), dtype=tf.float32) # [s, h]
    self.b = tf.Variable(np.random.uniform(-0.05, 0.05, size=[config.state_size]), dtype=tf.float32) #[s]

  def feed(self):
    config = self.config
    input = np.random.randint(0, config.state_size-1, size=[config.batch_size, config.max_seq_len])
    target = np.random.randint(0, config.state_size-1, size=[config.batch_size, config.max_seq_len])
    sub_target = np.random.randint(0, 1, size=[config.batch_size, config.max_seq_len, 6])

    feed_dict = {self.input.name:input, self.target.name:target, self.sub_target.name:sub_target}
    return feed_dict

  def build(self, fast):
    h = self.build_rnn()
    if fast:
      loss = self.build_fast(h)
    else:
      loss = self.build_slow(h)
    return loss

  def run(self, loss, sess):
    fetch_dict = {loss.name: loss}
    feed_dict = self.feed()
    sess.run(fetch_dict, feed_dict)

  def build_rnn(self):
    config = self.config
    self.emb = tf.Variable(np.random.uniform(-0.05, 0.05, size=[config.state_size, config.emb_dim]), name="emb", dtype=tf.float32)
    emb_input = tf.nn.embedding_lookup(self.emb, self.input) # [b, t, emb]
    cell = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_dim)
    h, h_last = tf.nn.dynamic_rnn(cell, emb_input, dtype=tf.float32)
    h = tf.reshape(h, [-1, config.hidden_dim]) # [b*t, h]
    return h

  def build_fast(self, h):
    config = self.config
    # h: [b*t, h]
    target_flat = tf.expand_dims(tf.reshape(self.target, [-1]), 1) # [b*t]
    input_flat = tf.reshape(self.input, [-1]) # [b*t]
    sub_adj_mat = tf.nn.embedding_lookup(self.adj_mat, input_flat) # [b*t, 6]
    sub_adj_mask = tf.nn.embedding_lookup(self.adj_mask, input_flat) # [b*t, 6]
    print(target_flat.get_shape)
    print(sub_adj_mat.get_shape)
    target_and_sub_adj_mat = tf.concat(1, [target_flat, sub_adj_mat]) # [b*t, 7]
    sub_w = tf.nn.embedding_lookup(self.w, target_and_sub_adj_mat) # [b*t, 7, h]
    sub_b = tf.nn.embedding_lookup(self.b, target_and_sub_adj_mat) # [b*t, 7]
    h = tf.expand_dims(h, 1) #[b*t, 1, h]
    tiled_h = tf.tile(h, [1,7,1]) # [b*t, 7, h]
    h_flat = tf.reshape(tiled_h, [-1, int(tiled_h.get_shape()[2])]) # [b*t*7, h]
    w_flat = tf.reshape(sub_w, [-1, int(sub_w.get_shape()[2])]) # [b*t*7, h]
    w_h = tf.reshape(tf.reduce_sum(h_flat * w_flat, 1), [-1, 7]) # [b*t, 7]
    target_logit_and_sub_logits = w_h + sub_b # [b*t, 7]

    scale = tf.reduce_max(target_logit_and_sub_logits, 1) # [b*t]
    scaled_target_logit_and_sub_logits = tf.transpose(
      tf.transpose(target_logit_and_sub_logits) - scale) # [b*t,7]

    exp_logit = tf.exp(scaled_target_logit_and_sub_logits[:,1:]) # [b*t, 6]
    log_denominator = tf.log(tf.reduce_sum(exp_logit * sub_adj_mask, 1)) # [b*t]
    log_numerator = tf.reshape(scaled_target_logit_and_sub_logits[:, 0:1], [-1]) # [b*t]
    loss = log_denominator - log_numerator
    return loss

  def build_slow(self, h):
    config = self.config
    onehot_target = tf.one_hot(tf.reshape(self.target, [-1]), config.state_size, dtype=tf.float32) # [b*t, s]
    logit = tf.matmul(h, tf.transpose(self.w)) + self.b # [b*t, s]
    scale = tf.reduce_max(logit, 1) # [b*t]
    scaled_logit = tf.transpose(tf.transpose(logit) - scale) # [b*t, s]
    log_denominator = tf.log(tf.reduce_sum(tf.exp(scaled_logit), 1)) # [b*t]
    print(scaled_logit.get_shape)
    print(onehot_target.get_shape)
    print(log_denominator.get_shape)
    #return log_denominator
    log_numerator = tf.reduce_sum(scaled_logit * onehot_target, 1) # [b*t]
    loss = log_denominator - log_numerator
    #print(loss.get_shape())
    return loss

  def benchmark(self, fast, samples_for_benchmark):
    config = self.config
    steps = samples_for_benchmark // config.batch_size + 1
    loss = self.build(fast)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
      sess.run(init)
      t1 = time.time()
      for _ in range(steps):
        self.run(loss, sess)
      t2 = time.time()
      samples_per_sec = steps * config.batch_size / float(t2 - t1)
      ms_per_sample = float(t2 - t1) * 1000.0 / (steps * config.batch_size)
    print("%d samples per sec, %.4f ms per sample, batch_size = %d" %
          (samples_per_sec, ms_per_sample, config.batch_size))

# set to any value you want
tmp_conf = Config()
tmp_conf.state_size = 60000
tmp_conf.batch_size = 64
tmp_conf.max_seq_len = 80
tmp_conf.emb_dim = 400
tmp_conf.hidden_dim = 600

with tf.Graph().as_default():
  demo = CSSRNN(tmp_conf)
  demo.benchmark(True, 20000) # with speed up
  #demo.benchmark(False, 20000) # no speed up
input()

