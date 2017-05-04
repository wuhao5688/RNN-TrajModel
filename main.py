from __future__ import print_function
import numpy as np
import math
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from geo import Map, GeoPoint
from ngram_model import N_gram_model
from trajmodel import TrajModel
import time
import os
import distutils.util as du
import copy
from tensorflow.python.client import timeline
try: # python2
    import ConfigParser as configparser
except ImportError: # python3
    import configparser
import sys
if sys.version > '3':
    PY3 = True
else:
    PY3 = False
from tensorflow.contrib.tensorboard.plugins import projector

routeFile = "/data/porto/porto_cleaned_mm_edges.txt"
map_path = "/data/porto/map/"

"""
label = [1,2,3]
feature = [[0.1,0.1,0.2], [0.2,0.2,0.2],[0.3,0.2,0.1]]
np.savetxt("label.txt", label, '%d')
np.savetxt("feature.txt", feature)

const_tensor = tf.constant([1,2,3])
v = tf.Variable()
saver = tf.train.Saver([const_tensor])

target = tf.constant([1,2,3,4,5])
predict = tf.constant([1,2,3,4,3])
acc = tf.reduce_sum(tf.cast(tf.equal(target, predict), tf.float32))
with tf.Session() as sess:
  ans = acc.eval()
  print(ans)
raw_input()
"""
"""
def softmax(x):
  x -= np.max(x)
  ex = np.exp(x)
  sum = np.sum(ex)
  return ex / sum

def test(label):
  fx = open('50/x_' + str(label) + '.raw', 'rb')
  fy = open('50/y_' + str(label) + '.raw', 'rb')
  bx = fx.read()
  by = fy.read()
  x = np.fromstring(bx, dtype=np.float32).reshape((-1, 60))
  y = np.fromstring(by, dtype=np.float32).reshape((-1, 6))
  count = y.shape[0]
  s0 = 0
  s1 = 0

  for i in range(count):
    plt.plot(softmax(x[i]))
    plt.show()
    plt.plot(y[i])
    plt.show()
  '''
  for i in range(count):
    if y[i][1] > 0.5:
      sl = 0
      sr = 0
      for j in range(2, 20):
        sl += x[i][j]
      for j in range(42, 60):
        sr += x[i][j]
      if sl > sr:
        s0 += 1
      else:
        s1 += 1
  print s0, ' ', s1
  '''
  '''
  '''
"""
"""
class CSSRNN:
  def __init__(self, config):
    self.config = config
    adj_mat_array = np.random.randint(0, config.state_size-1, size=[config.state_size, 6])
    self.adj_mat = tf.constant(adj_mat_array)
    self.adj_mask = tf.ones([config.state_size, 6])
    self.target = tf.placeholder(tf.float32) #[b,t]
    self.sub_target = tf.placeholder(tf.float32) #[b,t,6], one-hot
    self.input = tf.placeholder(tf.int32, name="input") # [b,t]
    self.w = tf.ones([config.state_size, config.hidden_dim]) #[s,h]
    self.b = tf.ones([config.state_size]) #[b]

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
    self.emb = tf.Variable(np.random.uniform(-0.05, 0.05, size=[config.state_size, config.emb_dim]), name="emb")
    emb_input = tf.nn.embedding_lookup(self.emb, self.input) # [b, t, emb]
    cell = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_dim)
    h, h_last = tf.nn.dynamic_rnn(cell, emb_input)
    h = tf.reshape(h, [-1, config.hidden_dim]) # [b*t, h]
    return h

  def build_fast(self, h):
    config = self.config
    # h: [b*t, h]
    target_flat = tf.reshape(self.target, [-1]) # [b*t]
    sub_adj_mat = tf.nn.embedding_lookup(self.adj_mat, self.input) # [b*t, 6]
    sub_adj_mask = tf.nn.embedding_lookup(self.adj_mask, self.input) # [b*t, 6]
    target_and_sub_adj_mat = tf.concat(1, [target_flat, sub_adj_mat]) # [b*t, 7]
    sub_w = tf.nn.embedding_lookup(self.w, sub_adj_mat) # [b*t, 7, h]
    sub_b = tf.nn.embedding_lookup(self.b, sub_adj_mat) # [b*t, 7]
    h = tf.expand_dims(h, 1) #[b*t, 1, h]
    tiled_h = tf.tile(h, [1,7,1]) # [b*t, 7, h]
    h_flat = tf.reshape(tiled_h, [-1, int(tiled_h.get_shape()[2])]) # [b*t*7, h]
    w_flat = tf.reshape(sub_b, [-1, int(sub_w.get_shape()[2])]) # [b*t*7, h]
    w_h = tf.reshape(tf.reduce_sum(h_flat * w_flat, 1), [-1, 7]) # [b*t, 7]
    target_logit_and_sub_logits = w_h + sub_b # [b*t, 7]

    scale = tf.reduce_max(target_logit_and_sub_logits, 1) # [b*t]
    scaled_target_logit_and_sub_logits = tf.transpose(
      tf.transpose(target_logit_and_sub_logits) - scale) # [b*t,7]

    exp_logit = tf.exp(scaled_target_logit_and_sub_logits[:,1:]) # [b*t, 6]
    log_fenmu = tf.log(tf.reduce_sum(exp_logit * sub_adj_mask, 1)) # [b*t]
    log_fenzi = tf.reshape(scaled_target_logit_and_sub_logits[:, 0:1], [-1]) # [b*t]
    loss = log_fenmu - log_fenzi
    return loss

  def build_slow(self, h):
    config = self.config
    onehot_target = tf.one_hot(tf.reshape(self.target, [-1]), config.state_size, dtype=tf.float32) # [b*t, s]
    logit = tf.matmul(h, tf.transpose(self.w)) + self.b # [b*t, s]
    scale = tf.reduce_max(logit, 1) # [b*t]
    scaled_logit = tf.transpose(tf.transpose(logit) - scale) # [b*t, s]
    log_fenmu = tf.log(tf.reduce_sum(tf.exp(scaled_logit), 1)) # [b*t]
    log_fenzi = tf.reduce_sum(scaled_logit * onehot_target, 1) # [b*t]
    loss = log_fenmu - log_fenzi
    return loss

  def benchmark(self, samples_for_benchmark):
    config = self.config
    steps = samples_for_benchmark // config.batch_size + 1
    with tf.Graph().as_default():
      loss = self.build(True)
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
"""

def test_angle(map, angle_granularity = 60, sigma = 5.0):
  while(True):
    # edge = np.random.choice(map.edges)
    edge = map.edges[config.trace_input_id]
    print("pick up edge " + str(edge.id))

    def calc_deg(p1, p2):
      cos_val = (p2[0] - p1[0]) / math.sqrt((p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1]))
      rad = math.acos(cos_val)
      if p2[1] < p1[1]:  # [pi, 2pi]
        rad = 2 * math.pi - rad
      return rad / math.pi * 180.0

    adj_dists = []
    for adj_edge in edge.adjList:

      p1 = map.nodes[adj_edge.startNodeId]
      p2 = map.nodes[adj_edge.endNodeId]
      deg = calc_deg(p1, p2)
      print("adj_edge %d deg = %.3f" % (adj_edge.id, deg))
      probs = []
      for bucket_deg in np.linspace(0.0, 360.0, angle_granularity + 1)[0:-1]:
        if abs(bucket_deg - deg) < 360.0 - abs(bucket_deg - deg):
          delta = abs(bucket_deg - deg)
        else:
          delta = 360.0 - abs(bucket_deg - deg)
        probs.append(math.exp(-delta * delta / (2 * sigma * sigma)))
      # normalization
      summation = sum(probs)
      for i in range(len(probs)):
        probs[i] /= summation
      adj_dists.append(probs)
    buckets = np.linspace(0.0, 360.0, angle_granularity + 1)[0:-1]
    for i in range(len(adj_dists)):
      probs = adj_dists[i]
      plt.plot(buckets, probs)
      plt.title(str(edge.adjList_ids[i]))
      plt.show()
    for i in range(len(adj_dists)):
      probs = adj_dists[i]
      plt.plot(buckets, probs)

    plt.show()

"""
mask__ = tf.sparse_placeholder(tf.float32)
targets_ = tf.constant([0,1,1], tf.int32)
outputs = tf.constant([[1,2,3,4],[3,4,5,6],[6,7,8,9]], dtype=tf.float32)
onehot_targets_ = tf.one_hot(targets_,4, dtype=tf.float32)
#mask = tf.SparseTensor([[0,0],[0,1], [1,1], [2,2]], 4 * [1.0], [3,4])
logits = tf.sparse_softmax(mask__ * outputs)
xent = logits * onehot_targets_
loss = tf.sparse_reduce_sum(xent, 1)


indices = np.array([[0,0],[0,1], [1,1], [2,2]], dtype=np.int32)
values = np.array(4 * [1.0], dtype=np.float32)
shape = np.array([3,4], dtype=np.int64)

feed = {mask__: (indices, values, shape)}
with tf.Session() as sess:
  tensor = loss
  val = sess.run(loss, feed_dict=feed)
  print(val)
"""

def enum(*args):
  enums = dict(zip(args, range(len(args))))
  return type('Enum', (), enums)


# RNN_TYPE = enum('rnn', 'gru', 'lstm')
# OPT_TYPE = enum('sgd', 'rmsprop', 'adam', 'adagrad')
# CANDIDATE_SAMPLER_TYPE = enum('default', 'uniform', 'log_uniform', 'learned_unigram', 'fixed_unigram')
# SAMPLED_SOFTMAX_TYPE = enum('sampled_softmax', 'nce')
# CONSTRAINED_SOFTMAX_STRATEGY = enum('sparse_tensor', 'adjmat_dense', 'adjmat_adjmask')

class Config(object):
  # dataset configuration
  dataset_name = "porto_40k" # 'porto_40k', 'porto_6k'
  dataset_path = None
  workspace = '/data'
  data_size = -1
  dataset_ratio = [0.7, 0.2, 0.1]
  state_size = None
  EOS_ID = None
  PAD_ID = 0
  TARGET_PAD_ID = None
  float_type = tf.float32  # `tf.float16` is two times slower than `tf.float32` in GTX TITAN
  int_type = tf.int32

  # ckpt
  load_path = None # do not set this manually
  save_path = "/data/porto/ckpt"
  encoder_save_path_forward_part = None # do not set this manually
  encoder_load_path = None # do not set this manually
  decoder_load_path = None # do not set this manually

  loss_for_filename = "loss_p" # note that ppl will also compute for this name
  max_ckpt_to_keep = 100
  save_ckpt_duration_sec = 3600 # not used
  load_ckpt = True
  save_ckpt = True
  compute_ppl = True
  direct_stdout_to_file = False # if True, all stuffs will be printed into log file
  log_filename = None
  log_file = None
  use_v2_saver = False

  # model configuration
  hidden_dim = 800  # 600
  emb_dim = 800  # 600
  num_layers = 1  # 3
  rnn = 'lstm' #'rnn', 'gru', 'lstm'
  use_bidir_rnn = False
  eval_mode = False

  build_multitask_in_train = True
  use_constrained_softmax_in_train = True
  build_unconstrained_in_train = False
  build_multitask_in_test = False
  use_constrained_softmax_in_test = True
  build_unconstrained_in_test = True
  constrained_softmax_strategy = 'adjmat_adjmask' # suggested  # 'sparse_tensor', 'adjmat_dense', 'adjmat_adjmask'
  input_dest = True # if False, append the destination feature on the input feature
  dest_emb = True # if False, use geo coordinate of the end point of the destination edge as the additonal feature


  # direction method
  predict_dir = True
  dir_sigma = 10.0
  dir_granularity = 36
  encoder_decoder = 'end2end' # 'encoder', 'decoder', 'end2end', 'end2end_multitask'
  decoder_regularizer = 0.0001 # L2 regularizer on weight matrix, set this value to the one smaller than or equal to 0.0 to avoid using L2 regularization.
  decoder_keep_prob = 1.0 # Dropout on weight matrix set this value to the one larger than or equal to 1.0 to avoid using dropout.

  # params for training
  batch_size = 50  # 100 is faster on simple model but need more memory
  lr = 0.0003
  lr_decay = 0.9 # useless
  keep_prob = 0.7  # for dropout.opt, set a value > 1 for avoid using dropout
  max_grad_norm = 1.0  # for grad clipping
  init_scale = 0.03
  fix_seq_len = False  # if True, make each batch with the size [batch_size, config.max_seq_len] (False is faster)
  use_seq_len_in_rnn = False  # whether to use seq_len_ when unrolling rnn. Useless if `fix_seq_len` is `True` (False is faster)
  max_seq_len = 80
  opt = 'rmsprop' # 'sgd', 'rmsprop', 'adam', 'adagrad'

  # for epoch
  epoch_count = 1000
  # steps_per_ckpt = 10000 # depreciated
  # steps_per_epoch_in_train = None # depreciated
  samples_per_epoch_in_train = None
  # steps_per_epoch_in_valid = None  # depreciated
  samples_for_benchmark = 1000
  run_options = None
  run_metadata = None
  trace_filename = "timeline.json"
  time_trace = False

  # for sampled softmax
  use_sampled_softmax = False
  sample_count_for_sampled_softmax = 10000
  candidate_sampler = 'default' # 'default', 'uniform', 'log_uniform', 'learned_unigram', 'fixed_unigram'
  sampled_softmax_alg = 'sampled_softmax' # 'sampled_softmax', 'nce'

  # miscellaneous
  eval_ngram_model = False

  #debug
  trace_hid_layer = False
  trace_input_id = None


  def __init__(self, config_path = None):
    if config_path is not None:
      self.load(config_path)
    if self.time_trace:
      self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      self.run_metadata = tf.RunMetadata()
    # set workspace
    self.workspace = os.path.join(self.workspace, self.dataset_name)
    self.dataset_path = os.path.join(self.workspace, "data/cleaned_mm_edges.txt")
    self.map_path = os.path.join(self.workspace, "map/")
    self.set_save_path()
    if self.predict_dir and self.encoder_decoder == 'encoder':
      if self.loss_for_filename != "err_dir":
        print("Warning, in encoder training mode, automatically set config.loss_for_filename to \"err_dir\"")
        self.loss_for_filename = "err_dir"''
      if self.compute_ppl:
        print("Warning, in encoder training mode, automatically set config.compute_ppl to False")
        self.compute_ppl = False
    if self.eval_mode and self.save_ckpt:
      print("Warning, in evaluation mode, automatically set config.save_ckpt to False")
      self.save_ckpt = False

  def printf(self):
    print("========================================\n")
    print("dataset configuration:\n" \
          "\tdataset_name = {dataset_name}\n" \
          "\tdata_size = {data_size}\n" \
          "\tstate_size = {state_size}\n" \
          "\tratio = {ratio}\n" \
          "\tsamples_per_epoch_in_train = {samples_per_epoch_in_train}" \
      .format(dataset_name=self.dataset_name, samples_per_epoch_in_train=self.samples_per_epoch_in_train,
              data_size=self.data_size, state_size=self.state_size, ratio=self.dataset_ratio))

    print("\nmodel configuration:\n" \
          "\temb_dim = {emb_dim}\n" \
          "\thid_dim = {hid_dim}\n" \
          "\tdeep = {deep}\n" \
          "\tuse_bidir_rnn = {use_bidir_rnn}\n" \
          "\tbuild_multitask_in_train = {build_multitask_in_train}\n" \
          "\tuse_constrained_softmax_in_train = {use_constrained_softmax_in_train}\n" \
          "\tbuild_unconstrained_in_train = {build_unconstrained_in_train}\n" \
          "\tinput_dest = {input_dest}\n" \
          "\tdest_emb = {dest_emb}" \
      .format(emb_dim=self.emb_dim, hid_dim=self.hidden_dim, deep=self.num_layers,
              build_multitask_in_train=self.build_multitask_in_train,
              use_constrained_softmax_in_train=self.use_constrained_softmax_in_train,
              build_unconstrained_in_train=self.build_unconstrained_in_train,
              input_dest=self.input_dest, dest_emb=self.dest_emb, use_bidir_rnn=self.use_bidir_rnn))

    if self.predict_dir:
      print("\ndirection_method:\n" \
            "\tencoder_decoder = {encoder_decoder}\n" \
            "\tdir_sigma = {dir_sigma}\n" \
            "\tdir_granularity = {dir_granularity}\n" \
            "\tdecoder_regularizer = {decoder_regularizer}\n" \
            "\tdecoder_keep_prob = {decoder_keep_prob}" \
      .format(encoder_decoder=self.encoder_decoder, dir_sigma=self.dir_sigma,
              dir_granularity=self.dir_granularity, decoder_regularizer = self.decoder_regularizer,
              decoder_keep_prob = self.decoder_keep_prob))

    print("\nparams:\n" \
          "\tbatch = {batch_size}\n" \
          "\tlr = {lr}\n" \
          "\tlr_decay = {lr_decay}\n" \
          "\tmax_grad_norm = {max_grad_norm}\n" \
          "\tinit_scale = {init_scale}\n" \
          "\tkeep_prob = {keep_prob}\n" \
          "\tfix_seq_len = {fix_seq_len}\n" \
          "\tuse_seq_len_in_rnn = {use_seq_len_in_rnn}\n" \
          "\tmax_seq_len = {max_seq_len}\n" \
          "\toptimizer = {opt}" \
      .format(batch_size=self.batch_size, lr=self.lr, lr_decay=self.lr_decay, keep_prob=self.keep_prob,
              max_grad_norm=self.max_grad_norm, init_scale=self.init_scale,
              fix_seq_len=self.fix_seq_len, max_seq_len=self.max_seq_len,
              use_seq_len_in_rnn=self.use_seq_len_in_rnn, opt=self.opt))
    print("\n========================================")

  def set_config(self, routes, roadnet):
    """
    decide some attributes in config by dataset `routes`
    :param routes: the whole dataset, list of lists
    :return: Nothing
    """
    self.data_size = len(routes)
    max_edge_id = max([max(route) for route in routes])  # 40266
    min_edge_id = min([max(route) for route in routes])  # 330
    print("min_edge_id = %d, max_edge_id = %d" % (min_edge_id, max_edge_id))
    max_route_len = max([len(route) for route in routes])
    print("max seq_len = %d" % max_route_len)
    self.EOS_ID = max_edge_id + 1
    self.state_size = max_edge_id + 2

    # if self.steps_per_epoch_in_train is None:
    #  self.steps_per_epoch_in_train = int(self.dataset_ratio[0] * len(routes) / self.batch_size)
    if self.samples_per_epoch_in_train is None:
      self.samples_per_epoch_in_train = int(self.dataset_ratio[0] * len(routes))
    # self.steps_per_epoch_in_valid = int(self.dataset_ratio[1] * len(routes) / self.batch_size)

    # Note that we should pad target by the adjacent state of state `PAD_ID`.
    # Since if we also pad the target by `PAD_ID`, when computing using constrained softmax,
    # the logit of the target state `PAD_ID` will result in 0 since it is impossible to
    # transfer from `PAD_ID` to `PAD_ID` and if we use `-log(logit(PAD_ID))` to compute the cross-entropy,
    # `nan` will emerge and unfortunately it is useless to apply a mask to the result (`0 * nan` will not result to 0)
    self.TARGET_PAD_ID = roadnet.edges[config.PAD_ID].adjList_ids[0]

  def set_save_path(self):
    """
    generate a string represents the capacity of the model
    and append this string to `self.save_path`
    :return:
    """
    model_capacity_str = "emb_{emb}_hid_{hid}_deep_{deep}".format(emb=self.emb_dim,
                                                                  hid=self.hidden_dim,
                                                                  deep=self.num_layers)
    if self.use_bidir_rnn:
      model_capacity_str += "_bidir"
    if self.input_dest:
      if self.dest_emb:
        model_capacity_str = "dest_emb/" + model_capacity_str
      else:
        model_capacity_str = "dest_direct/" + model_capacity_str

    ckpt_home = os.path.join(self.workspace, "ckpt")

    if self.predict_dir:
      # only encoder
      encoder_str = model_capacity_str + ("_dir_%d_sigma_%.1f" % (self.dir_granularity, self.dir_sigma))
      decoder_str = "dir_%d_sigma_%.1f" % (self.dir_granularity, self.dir_sigma)

      if self.encoder_decoder == 'encoder':
        self.encoder_save_path_forward_part = os.path.join(ckpt_home,
                                                         "encoder_decoder/encoder/forward_only/" + encoder_str)
        self.save_path = os.path.join(ckpt_home, "encoder_decoder/encoder/train/" + encoder_str)
        self.encoder_load_path = str(self.save_path)
        self.load_path = self.encoder_load_path
      # encoder-decoder
      elif self.encoder_decoder == 'decoder':
        self.encoder_load_path = os.path.join(ckpt_home,
                                              "encoder_decoder/encoder/forward_only/" + encoder_str)
        self.save_path = os.path.join(ckpt_home, "encoder_decoder/decoder/" + decoder_str )
        self.decoder_load_path = str(self.save_path)
        self.load_path = self.decoder_load_path
      # end_to_end encoder_decoder
      elif self.encoder_decoder == 'end2end':
        self.save_path = os.path.join(ckpt_home, "encoder_decoder/end2end/" + encoder_str)
        self.load_path = str(self.save_path)
      else:
        raise Exception("config.encoder_decoder should be correctly defined")
    else:
      if not self.use_constrained_softmax_in_train:
        model_capacity_str = "full_softmax/" + model_capacity_str
      model_capacity_str = "normal/" + model_capacity_str
      self.load_path = os.path.join(ckpt_home, model_capacity_str)
      self.save_path = os.path.join(ckpt_home, model_capacity_str)

  def reformat(self):
    """
    reformat the attributes from string to the correct type
    used after `load()`
    :return:
    """
    self.data_size = int(self.data_size)
    self.PAD_ID = int(self.PAD_ID)

    self.max_ckpt_to_keep = int(self.max_ckpt_to_keep)
    self.load_ckpt = bool(du.strtobool(self.load_ckpt))
    self.save_ckpt = bool(du.strtobool(self.save_ckpt))
    self.compute_ppl = bool(du.strtobool(self.compute_ppl))
    self.direct_stdout_to_file = bool(du.strtobool(self.direct_stdout_to_file))
    self.samples_per_epoch_in_train = int(self.samples_per_epoch_in_train)
    self.use_v2_saver = bool(du.strtobool(self.use_v2_saver))

    self.hidden_dim = int(self.hidden_dim)
    self.emb_dim = int(self.emb_dim)
    self.num_layers = int(self.num_layers)
    self.use_bidir_rnn = bool(du.strtobool(self.use_bidir_rnn))
    self.eval_mode = bool(du.strtobool(self.eval_mode))

    self.build_multitask_in_train = bool(du.strtobool(self.build_multitask_in_train))
    self.use_constrained_softmax_in_train = bool(du.strtobool(self.use_constrained_softmax_in_train))
    self.build_unconstrained_in_train = bool(du.strtobool(self.build_unconstrained_in_train))
    self.build_multitask_in_test = bool(du.strtobool(self.build_multitask_in_test))
    self.use_constrained_softmax_in_test = bool(du.strtobool(self.use_constrained_softmax_in_test))
    self.build_unconstrained_in_test = bool(du.strtobool(self.build_unconstrained_in_test))
    self.input_dest = bool(du.strtobool(self.input_dest))
    self.dest_emb = bool(du.strtobool(self.dest_emb))

    self.predict_dir = bool(du.strtobool(self.predict_dir))
    self.dir_sigma = float(self.dir_sigma)
    self.dir_granularity = int(self.dir_granularity)
    self.decoder_regularizer = float(self.decoder_regularizer)
    self.decoder_keep_prob = float(self.decoder_keep_prob)

    self.batch_size = int(self.batch_size)
    self.lr = float(self.lr)
    self.lr_decay = float(self.lr_decay)
    self.keep_prob = float(self.keep_prob)
    self.max_grad_norm = float(self.max_grad_norm)
    self.init_scale = float(self.init_scale)
    self.fix_seq_len =bool(du.strtobool(self.fix_seq_len))
    self.use_seq_len_in_rnn = bool(du.strtobool(self.use_seq_len_in_rnn))
    self.max_seq_len = int(self.max_seq_len)

    self.epoch_count = int(self.epoch_count)
    self.samples_for_benchmark = int(self.samples_for_benchmark)

    self.use_sampled_softmax = bool(du.strtobool(self.use_sampled_softmax))
    self.sample_count_for_sampled_softmax = int(self.sample_count_for_sampled_softmax)

    self.eval_ngram_model = bool(du.strtobool(self.eval_ngram_model))

    self.trace_hid_layer = bool(du.strtobool(self.trace_hid_layer))
    self.trace_input_id = int(self.trace_input_id)

  def load(self, config_path):
    """
    Load config file
    Format: standard format supported by `ConfigParser`
    The parameters which do not appear in the config file will be set as the default values.
    :param config_path:
    :return: nothing
    """
    cp = configparser.ConfigParser()
    cp.read(config_path)
    secs =  cp.sections()
    for sec in secs:
      for k,v in cp.items(sec):
        if hasattr(self, k):
          setattr(self, k, v)
        else:
          raise Exception("no attribute named \"%s\" in class \"Config\"" % k)
    self.reformat()
    return

class MapInfo(object):
  def __init__(self, map, config):
    self.map = map
    self.config = config
    self.adj_mat, self.adj_mask = self.get_adjmat_and_mask(config.PAD_ID)
    self.dir_distris = self.get_dir_distributions(config.dir_granularity, config.dir_sigma)
    self.dest_coord = self.get_dest_coord()
    return

  def get_adjmat_and_mask(self, pad_id):
    """
    `adjmat` has the shape of `(#edge, max_len)`, where `max_len` is the maximum of #adjacent_edges of each edge.
    `adjmat[i]` records all the adjacent edges of edge_i (including padding with id = `pad_id`)
    eg: if adjacent edges of edge 1 is [2,3],
           adjacent edges of edge 2 is [3,4,5] and pad_id = 0
        then `adjmat` should be [[0,0,0], [2,3,0], [3,4,5]] (dtype = int)
        and `adjmask` should be [[0,0,0], [1,1,0], [1,1,1]] (dtype = float)
    PS: In real application, it is better to let a useless edge be the `PAD` e.g. edge 0.
        And actually, this function will not fill all zeros for the mask of `PAD` state. [TODO] Try to fix this later.
    :param map: instance of `Map`
    :param pad_id: int
    :return: adjmat and adjmask with shape `(#edge, max_adj_len)`
    """
    map = self.map
    adjmat, adjmask, lens = [], [], []
    for edge in map.edges:
      adjmat.append(list(edge.adjList_ids))
      lens.append(len(edge.adjList_ids))
    max_len = max(lens)
    # max_len = 6 #TODO
    for i in range(len(adjmat)):
      adjmat[i].extend([pad_id] * (max_len - len(adjmat[i])))
      adjmask.append([1.0] * lens[i] + [0.0] * (max_len - lens[i]))
    return np.array(adjmat), np.array(adjmask)

  def get_dir_distributions(self, angle_granularity, sigma):
    map = self.map
    def calc_deg(p1, p2):
      """
      compute the degree of p1->p2 relative to the horizontal line, [0, 360)
      :param p1: Pos type
      :param p2: Pos type
      :return: float
      """
      try:
        cos_val = (p2[0] - p1[0]) / math.sqrt((p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1]))
      except:
        return 0.0  # if dist(p1, p2) = 0
      rad = math.acos(cos_val)
      if p2[1] < p1[1]:  # [pi, 2pi]
        rad = 2 * math.pi - rad
      return rad / math.pi * 180.0

    distributions = []
    for edge in map.edges:
      p1 = map.nodes[edge.startNodeId]
      p2 = map.nodes[edge.endNodeId]
      deg = calc_deg(p1, p2)
      distri = []
      for bucket_deg in np.linspace(0.0, 360.0, angle_granularity + 1)[0:-1]:
        if abs(bucket_deg - deg) < 360.0 - abs(bucket_deg - deg):
          delta = abs(bucket_deg - deg)
        else:
          delta = 360.0 - abs(bucket_deg - deg)
        distri.append(math.exp(-delta * delta / (2 * sigma * sigma)))
      # normalization
      summation = sum(distri)
      for i in range(len(distri)):
        distri[i] /= summation
      distributions.append(distri)
    return np.array(distributions)

  def get_dest_coord(self):
    map = self.map
    dests = []
    for edge in map.edges:
      node = map.nodes[edge.endNodeId]
      dests.append([node.lat, node.lon])
    # normalize
    np_dests = np.array(dests)
    minlat, maxlat = min(np_dests[:,0]), max(np_dests[:,0])
    minlon, maxlon = min(np_dests[:,1]), max(np_dests[:,1])
    for row in range(np_dests.shape[0]):
      np_dests[row][0] = (np_dests[row][0] - minlat) / (maxlat-minlat)
      np_dests[row][1] = (np_dests[row][1] - minlon) / (maxlon-minlon)
    return np_dests

def read_data(file_path, max_count=-1, max_seq_len = None, ratio=[0.7, 0.2, 0.1]):
  """
  Read the route data
  :param file_path: path of the file to load
  :param max_count: maximum count of routes to be loaded, default is -1 which loads all routes.
  :param max_seq_len: samples longer than `max_seq_len` will be skipped.
  :param ratio: ratio[train, valid, test] for split the dataset, automatically normalized if sum(ratio) is not 1.
  :return: three lists of lists, in order: train, valid, test.
  """
  file = open(file_path)
  routes = []
  current_count = 0
  for line in file:
    if current_count == max_count:
      break
    route_str = line.split(',') # including the last blank substr
    if len(route_str)-1 > max_seq_len or len(route_str)-1 < 2:
      continue
    routes.append([int(route_str[i]) for i in range(len(route_str) - 1)]) # last element is an empty string
    current_count += 1

  ratio = [r / sum(ratio) for r in ratio]
  train = [routes[i] for i in range(0, int(len(routes) * ratio[0]))]
  valid = [routes[i] for i in range(int(len(routes) * ratio[0]), int(len(routes) * (ratio[0] + ratio[1])))]
  test = [routes[i] for i in range(int(len(routes) * (ratio[0] + ratio[1])), len(routes))]
  return routes, train, valid, test

# load data
config = Config("config")
timestr = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))
if config.direct_stdout_to_file:
  if config.predict_dir:
    model_str = config.encoder_decoder
  else:
    model_str = "normal"
  config.log_filename = "log_" + timestr + "_" + config.dataset_name + "_" + model_str + ".txt"
  config.log_file = open(config.log_filename, 'w')
  sys.stdout = config.log_file


routes, train, valid, test = read_data(config.dataset_path, config.data_size, config.max_seq_len)
print("successfully read %d routes" % sum([len(train), len(valid), len(test)]))  # 1005031
max_edge_id = max([max(route) for route in routes])  # 40266
min_edge_id = min([max(route) for route in routes])  # 330
print("min_edge_id = %d, max_edge_id = %d" % (min_edge_id, max_edge_id))
max_route_len = max([len(route) for route in routes])
route_lens = [len(route) for route in routes]

print("train:%d, valid:%d, test:%d" % (len(train), len(valid), len(test)))
print(max(route_lens))
plt.hist(route_lens, bins=config.max_seq_len, cumulative=True, normed=True)
plt.show()

def count_trans(roadnet, data):
  # initialization
  print("start initialization")
  trans = []
  for edge in roadnet.edges:
    adjs = {}
    for adj_edge_id in edge.adjList_ids:
      adjs[adj_edge_id] = 0
    trans.append(adjs)

  # do stats
  print("start stats")
  for route in data:
    for i in range(len(route) - 1):
       trans[route[i]][route[i + 1]] += 1

  f = open("count_trans", "w")
  for edge in roadnet.edges:
    f.write(str(edge.id) + " ")
    for adj_edge_id in edge.adjList_ids:
      f.write("|" + str(adj_edge_id) + " :\t" + str(trans[edge.id][adj_edge_id]) + "\t")
    f.write("\n")
  f.close()

# load map
GeoPoint.AREA_LAT = 41.15
roadnet = Map()
roadnet.open(config.map_path)

# set config
config.set_config(routes, roadnet)
config.printf()

if config.trace_hid_layer:
  test_angle(roadnet)
# count_trans(roadnet, routes)

# get map info
mapInfo = MapInfo(roadnet, config)

if config.eval_ngram_model:
  # n-gram model eval
  markov_model = N_gram_model(roadnet, config)
  # markov_model.train_and_eval(train, valid, 5, config.max_seq_len, given_dest=True,use_fast=True, compute_all_gram=True)
  print("======================test set========================")
  markov_model.train_and_eval_given_dest(train, test, 3, 600, use_fast=True)
  #markov_model.train_and_eval(train, test, 4, config.max_seq_len, use_fast=True, compute_all_gram=True)
  #markov_model.train_and_eval_given_dest(train, test, 2, 10, True, False)
  #markov_model.train_and_eval_given_dest(train, test, 3, 40, True, False)
  #markov_model.train_and_eval_given_dest(train, test, 4, 80, True, False)
  print("======================valid set========================")
  markov_model.train_and_eval_given_dest(train, valid, 3, 600, use_fast=True)
  #markov_model.train_and_eval(train, valid, 4, config.max_seq_len, use_fast=True, compute_all_gram=True)
  #markov_model.train_and_eval_given_dest(train, valid, 2, 10, True, False)
  #markov_model.train_and_eval_given_dest(train, valid, 3, 40, True, False)
  #markov_model.train_and_eval_given_dest(train, valid, 4, 80, True, False)
  # markov_model.train_and_eval_given_dest(train, valid, 3, 60) # 4W
  # markov_model.train_and_eval_given_dest(train, valid, 2, 10, use_fast=True) # 4W
  # markov_model.train_and_eval_given_dest(train, valid, 4, 300) # 4W
  # markov_model.train_and_eval_given_dest(train, valid, 3, 10, True) # 6K
  input()

# construct model
with tf.Graph().as_default():
  initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
  model_scope = "Model"
  with tf.name_scope("Train"):
    with tf.variable_scope(model_scope, reuse=None, initializer=initializer):
      model = TrajModel(not config.trace_hid_layer, config, train, model_scope=model_scope, map=roadnet, mapInfo=mapInfo)

  with tf.name_scope("Valid"):
    with tf.variable_scope(model_scope, reuse=True):
      model_valid = TrajModel(False, config, valid, model_scope=model_scope, map=roadnet, mapInfo=mapInfo)

  with tf.name_scope("Test"):
    with tf.variable_scope(model_scope, reuse=True):
      model_test = TrajModel(False, config, test, model_scope=model_scope, map=roadnet, mapInfo=mapInfo)

  # sv = tf.train.Supervisor(logdir=config.load_path)
  # with sv.managed_session() as sess:
  sess_config = tf.ConfigProto()
  # sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4
  # sess_config.gpu_options.allow_growth = True
  with tf.Session(config=sess_config) as sess:

    # stuff for ckpt
    ckpt_path = None
    if config.load_ckpt:
      print('Input training ckpt filename (at %s): ' % config.load_path)
      if PY3:
        ckpt_name = input()
      else:
        ckpt_name = raw_input()
      print(ckpt_name)
      ckpt_path = os.path.join(config.load_path, ckpt_name)
      print('try loading ' + ckpt_path)
    '''
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
      print "restore model params from %s" % ckpt.model_checkpoint_path
      model.saver.restore(sess, ckpt.model_checkpoint_path)'''
    if ckpt_path and tf.gfile.Exists(ckpt_path):
      print("restoring model trainable params from %s" % ckpt_path)
      model.saver.restore(sess, ckpt_path)
    else:
      if config.load_ckpt:
        print("restore model params failed")
      print("initialize all variables...")
      sess.run(tf.initialize_all_variables())
    if config.predict_dir and config.encoder_decoder == 'decoder':
      print('Input forward params for encoder (at %s): ' % config.encoder_load_path)
      if PY3:
        encoder_param_name = input()
      else:
        encoder_param_name = raw_input()
      encoder_param_path = os.path.join(config.encoder_load_path, encoder_param_name)
      print('try loading ' + encoder_param_path)
      model.encoder_forward_saver.restore(sess, encoder_param_path)
      print("restoring pre-trained encoder params from %s" % encoder_param_path)

    """
    # tensorboard
    # Use the same LOG_DIR where you stored your checkpoint.
    summary_writer = tf.train.SummaryWriter(config.save_path)

    # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
    projectorConfig = projector.ProjectorConfig()

    # You can add multiple embeddings. Here we add only one.
    embedding = projectorConfig.embeddings.add()
    embedding.tensor_name = model.dest_emb_.name
    # Link this tensor to its metadata file (e.g. labels).
    #embedding.metadata_path = os.path.join(config.save_path, 'metadata.tsv')

    # Saves a configuration file that TensorBoard will read during startup.
    projector.visualize_embeddings(summary_writer, projectorConfig)

    """
    # benchmark
    print("speed benchmark for get_batch()...")
    how_many_tests = 1000
    t1 = time.time()
    for _ in range(how_many_tests):
      model.get_batch(model.data, config.batch_size, config.max_seq_len)
    t2 = time.time()
    print("%.4f ms per batch, %.4fms per sample, batch_size = %d" % (float(t2-t1)/how_many_tests*1000.0,
                                                                   float(t2-t1)/how_many_tests/config.batch_size*1000.0,
                                                                   config.batch_size))

    # use for timeline trace (unstable, need lots of memory)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    print('start benchmarking...')
    model.speed_benchmark(sess, config.samples_for_benchmark)
    # timeline generation
    model_valid.speed_benchmark(sess, config.samples_for_benchmark)
    print("start training...")

    if config.direct_stdout_to_file:
      config.log_file.close()
      config.log_file = open(config.log_filename, "a+")
      sys.stdout = config.log_file

    # let's go :)
    for ep in range(config.epoch_count):
      if not config.eval_mode:
        model.train_epoch(sess, train)
      model_valid.eval(sess, valid, True, True, model_train=model)
      model_test.eval(sess, test, False, False, model_train=model)
      raw_input()
input()


"""
plt.hist(route_lens, bins=100, cumulative=True, normed=True)
plt.show()
"""