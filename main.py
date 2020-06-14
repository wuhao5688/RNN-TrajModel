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
import types
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
#from tensorflow.contrib.tensorboard.plugins import projector

#routeFile = "/data/porto/porto_cleaned_mm_edges.txt"
#map_path = "/data/porto/map/"

# workspace (e.g., /data)
#     | dataset_name1 (e.g., porto_6k)
#         | data
#         |   | your_traj_file.txt
#         | map  (w.r.t. porto_6k)
#         |   | nodeOSM.txt
#         |   | edgeOSM.txt
#         | ckpt
#     | dataset_name2 (e.g., porto_40k)
#         | data
#         |   | your_traj_file.txt
#         | map  (w.r.t. porto_40k)
#         |   | nodeOSM.txt
#         |   | edgeOSM.txt
#         | ckpt


class Config(object):
  # dataset configuration
  dataset_name = "porto_40k" # e.g., 'porto_40k', 'porto_6k'
  dataset_path = None # do not set this manually
  workspace = '/data' # the true workspace will actually be workspace + '/' + dataset_name
  file_name = "example.txt"
  data_size = -1 # how many trajs you want to read in. `-1` means reading in all trajectories
  dataset_ratio = [0.8, 0.1, 0.1]
  state_size = None # do not set this manually
  EOS_ID = None # do not set this manually
  PAD_ID = 0
  TARGET_PAD_ID = None # do not set this manually
  float_type = tf.float32  # `tf.float16` is two times slower than `tf.float32` in GTX TITAN
  int_type = tf.int32

  # ckpt
  load_path = None # do not set this manually, the path will be automatically generated according to the model
  save_path = None # do not set this manually, the path will be automatically generated according to the model

  loss_for_filename = "loss_p" # use which loss to name the ckpt file. All loss in TrajModel.loss_dict can be set here.
                               # Here, "loss_p" is just the NLL loss of the sequence.
  max_ckpt_to_keep = 100
  load_ckpt = True
  save_ckpt = True
  compute_ppl = True # whether to compute the perplexity (sequence-level, not word-level)
  direct_stdout_to_file = False # if True, all stuffs will be printed into log file
  log_filename = None # do not set this manually
  log_file = None # do not set this manually
  use_v2_saver = False # tensorflow 0.12 starts to use ckpt V2
                       # and the code is written in 0.10 or 0.11 (I've forgotten the exact version D:) which is still in ckpt V1

  # model configuration
  hidden_dim = 400  # hidden units of rnn
  emb_dim = 400  # the dimension of embedding vector (both input states and destination states)
  num_layers = 1  # how many layers the rnn has, which means you can have a deep rnn for the rnn layer
  rnn = 'lstm' #'rnn', 'gru', 'lstm'   p.s. 'lstm' is the best
  model_type = 'CSSRNN' # 'RNN', 'CSSRNN', 'SPECIFIED_CSSRNN', 'LPIRNN'
                        # 'SPECIFIED_CSSRNN' means you can specify something (e.g., different speed boosting strategy, etc.)
                        # For more details, please refer `TrajModel.build_RNNLM_model()`
                        # And in most time there is no need to set it to 'SPECIFIED_CSSRNN'
                        # So you can just think that you have only three choices, i.e., RNN, CSSRNN and LPIRNN.
  use_bidir_rnn = False # whether to use bidirectional structure in rnn layer
  eval_mode = False # set it to `True` to skip training process on training set and directly go into evaluation process.
                    # You may switch on it if you only want to check the performance in valid/test set w.r.t. a given ckpt
                    # Note that its priority is higher than `save_ckpt`,
                    # which means `save_ckpt` will be compulsively set to `False`
  pretrained_input_emb_path = '' # the file of the pretrained embedding vectors (such as word2vec) of input states
                                 # Each line contains the embedding vector of a state, with the delimiter as ','
                                 # It is recommended to save the file through `np.savetxt()`
                                 # w.r.t. the ndarray having the shape of (state_size, emb_size)
                                 # If you do not want to load pretrained embeddings, just leave it the blank string.

  # Do not manually set the following 5 settings if you do not know what you are doing
  use_constrained_softmax_in_train = True # have effects only when the model_type is `SPECIFIED_CSSRNN`
  build_unconstrained_in_train = False # have effects only when the model_type is `SPECIFIED_CSSRNN`
  use_constrained_softmax_in_test = True # have effects only when the model_type is `SPECIFIED_CSSRNN`
  build_unconstrained_in_test = False # have effects only when the model_type is `SPECIFIED_CSSRNN`
  constrained_softmax_strategy = 'adjmat_adjmask' # suggested  # 'sparse_tensor' or 'adjmat_adjmask'

  input_dest = True # if `True`, append the destination feature on the input feature
  dest_emb = True # if `True`, use the distributed representation to represent the destination states
                  # otherwise, use geo coordinate of the end point of the destination edge as the additonal feature

  # params for LPIRNN
  lpi_dim = 200
  individual_task_regularizer = 0.0001 # L2 regularizer on weight matrix of individual task layer,
                               # set this value to the one smaller than or equal to 0.0 to avoid using L2 regularization.
  individual_task_keep_prob = 0.9 # Dropout on weight matrix of individual task layer,
                                  # set this value to the one larger than or equal to 1.0 to avoid using dropout.

  # params for training
  batch_size = 50  # 100 is faster on simple model but need more memory
  lr = 0.0001
  lr_decay = 0.9 # parameter for RMSProp optimizer
  keep_prob = 0.9  # for dropout in rnn layer and embedding, set a value > 1 for avoid using dropout
  max_grad_norm = 1.0  # for grad clipping
  init_scale = 0.03 # initialize the parameter uniformly from [-init_scale, init_scale]
  fix_seq_len = False  # if True, make each batch with the size [batch_size, config.max_seq_len] (False is faster)
  use_seq_len_in_rnn = False  # whether to use seq_len_ when unrolling rnn. Useless if `fix_seq_len` is `True` (False is faster)
  max_seq_len = 80 # the maximum length of a trajectory, ones larger than it will be omitted in the dataset
  opt = 'rmsprop' #the optimizer you want to use, support 'sgd', 'rmsprop' and 'adam'

  # for epoch
  epoch_count = 1000
  samples_per_epoch_in_train = -1 # you can manually set this to control how many samples an epoch will train
                                  # leave it default which will be computed by `dataset_ratio[0]*data_size`
  samples_for_benchmark = 1000 # how many samples you want to run for speed benchmark
  run_options = None # useless
  run_metadata = None # useless
  trace_filename = "timeline.json" # useless
  time_trace = False # useless

  # miscellaneous
  eval_ngram_model = False # set `True` to evaluate ngram model before train our neural trajectory model

  # debug
  trace_hid_layer = False # set `True` to enable tracing lpi
  trace_input_id = None # if you want to trace the lpi w.r.t. a specific state, just set this.

  def __init__(self, config_path = None):
    if config_path is not None:
      self.load(config_path)
    if self.time_trace:
      self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      self.run_metadata = tf.RunMetadata()
    # set workspace
    self.workspace = os.path.join(self.workspace, self.dataset_name)
    self.dataset_path = os.path.join(self.workspace, self.file_name)
    self.map_path = os.path.join(self.workspace, "map/")
    self.__set_save_path()
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
          "\tmodel_type = {model_type}\n" \
          "\temb_dim = {emb_dim}\n" \
          "\thid_dim = {hid_dim}\n" \
          "\tdeep = {deep}\n" \
          "\tuse_bidir_rnn = {use_bidir_rnn}\n" \
          "\tinput_dest = {input_dest}\n" \
          "\tdest_emb = {dest_emb}" \
      .format(model_type=self.model_type, emb_dim=self.emb_dim, hid_dim=self.hidden_dim, deep=self.num_layers,
              input_dest=self.input_dest, dest_emb=self.dest_emb, use_bidir_rnn=self.use_bidir_rnn))
    if self.model_type == "LPIRNN":
      print("\tlpi_dim = {lpi_dim}\n" \
            "\tindividual_task_regularizer = {individual_task_regularizer}\n" \
            "\tindividual_task_keep_prob = {individual_task_keep_prob}" \
      .format(lpi_dim=self.lpi_dim, individual_task_regularizer = self.individual_task_regularizer,
              individual_task_keep_prob = self.individual_task_keep_prob))

    print("\ntraining params:\n" \
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
    :param routes: the whole dataset, list of lists, each entry is the road id
    :return: Nothing
    """
    self.data_size = len(routes)
    max_edge_id = max([max(route) for route in routes])
    min_edge_id = min([max(route) for route in routes])
    print("min_edge_id = %d, max_edge_id = %d" % (min_edge_id, max_edge_id))
    max_route_len = max([len(route) for route in routes])
    print("max seq_len = %d" % max_route_len)
    self.EOS_ID = max_edge_id + 1
    self.state_size = max_edge_id + 2

    if self.samples_per_epoch_in_train < 0:
      self.samples_per_epoch_in_train = int(self.dataset_ratio[0] * len(routes))

    # Note that we should pad target by the adjacent state of state `PAD_ID`.
    # Since if we also pad the target by `PAD_ID`, when computing using constrained softmax,
    # the logit of the target state `PAD_ID` will result in 0 since it is impossible to
    # transfer from `PAD_ID` to `PAD_ID` and if we use `-log(logit(PAD_ID))` to compute the cross-entropy,
    # `nan` will emerge and unfortunately it is useless to apply a mask to the result (`0 * nan` will not result to 0)
    # Here I use `0` to be the `PAD_ID`, and the road 0 is also a valid road in the road network
    # which means it has an adjacent state (i.e., roadnet.edges[config.PAD_ID].adjList_ids[0]),
    # And since there is no historical trajectory that passes road 0, I decide to leverage it.
    self.TARGET_PAD_ID = roadnet.edges[self.PAD_ID].adjList_ids[0]

  def load(self, config_path):
    """
    Load config file
    Format: standard format supported by `ConfigParser`
    The parameters which do not appear in the config file will be set as the default values.
    :param config_path: the file path of the config file
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
    self.__reformat()
    return

  def __set_save_path(self):
    """
    generate a string represents the capacity and settings of the model
    and use this string to set `self.save_path` and `self.load_path`
    :return:
    """
    ckpt_home = os.path.join(self.workspace, "ckpt")
    model_capacity_str = "emb_{emb}_hid_{hid}_deep_{deep}".format(emb=self.emb_dim,
                                                                  hid=self.hidden_dim,
                                                                  deep=self.num_layers)
    if self.use_bidir_rnn:
      model_capacity_str += "_bidir"
    if self.input_dest:
      if self.dest_emb:
        model_capacity_str = "dest_emb/" + model_capacity_str
      else:
        model_capacity_str = "dest_coord/" + model_capacity_str
    else:
      model_capacity_str = "without_dest/" + model_capacity_str
    if self.model_type == 'LPIRNN':
      model_capacity_str += ("_lpi_%d" % self.lpi_dim)

    # e.g., "workspace/porto_6k/ckpt/LPIRNN/dest_emb/emb_400_hid_400_deep_1_lpi_200/"
    self.save_path = os.path.join(ckpt_home, self.model_type + "/" + model_capacity_str)
    self.load_path = os.path.join(ckpt_home, self.model_type + "/" + model_capacity_str)

  def __reformat(self):
    """
    reformat the attributes from string to the correct type
    used after `load()`
    :return:
    """
    self.data_size = int(self.data_size)
    self.PAD_ID = int(self.PAD_ID)

    self.max_ckpt_to_keep = int(self.max_ckpt_to_keep)
    if isinstance(self.load_ckpt, str):
      self.load_ckpt = bool(du.strtobool(self.load_ckpt))
    if isinstance(self.save_ckpt, str):
      self.save_ckpt = bool(du.strtobool(self.save_ckpt))
    if isinstance(self.compute_ppl, str):
      self.compute_ppl = bool(du.strtobool(self.compute_ppl))
    if isinstance(self.direct_stdout_to_file, str):
      self.direct_stdout_to_file = bool(du.strtobool(self.direct_stdout_to_file))
    self.samples_per_epoch_in_train = int(self.samples_per_epoch_in_train)
    if isinstance(self.use_v2_saver, str):
      self.use_v2_saver = bool(du.strtobool(self.use_v2_saver))

    self.hidden_dim = int(self.hidden_dim)
    self.emb_dim = int(self.emb_dim)
    self.num_layers = int(self.num_layers)
    if isinstance(self.use_bidir_rnn, str):
      self.use_bidir_rnn = bool(du.strtobool(self.use_bidir_rnn))
    if isinstance(self.eval_mode, str):
      self.eval_mode = bool(du.strtobool(self.eval_mode))

    if isinstance(self.use_constrained_softmax_in_train, str):
      self.use_constrained_softmax_in_train = bool(du.strtobool(self.use_constrained_softmax_in_train))
    if isinstance(self.build_unconstrained_in_train, str):
      self.build_unconstrained_in_train = bool(du.strtobool(self.build_unconstrained_in_train))
    if isinstance(self.use_constrained_softmax_in_test, str):
      self.use_constrained_softmax_in_test = bool(du.strtobool(self.use_constrained_softmax_in_test))
    if isinstance(self.build_unconstrained_in_test, str):
      self.build_unconstrained_in_test = bool(du.strtobool(self.build_unconstrained_in_test))
    if isinstance(self.input_dest, str):
      self.input_dest = bool(du.strtobool(self.input_dest))
    if isinstance(self.dest_emb, str):
      self.dest_emb = bool(du.strtobool(self.dest_emb))

    self.lpi_dim = int(self.lpi_dim)
    self.individual_task_regularizer = float(self.individual_task_regularizer)
    self.individual_task_keep_prob = float(self.individual_task_keep_prob)

    self.batch_size = int(self.batch_size)
    self.lr = float(self.lr)
    self.lr_decay = float(self.lr_decay)
    self.keep_prob = float(self.keep_prob)
    self.max_grad_norm = float(self.max_grad_norm)
    self.init_scale = float(self.init_scale)
    if isinstance(self.fix_seq_len, str):
      self.fix_seq_len =bool(du.strtobool(self.fix_seq_len))
    if isinstance(self.use_seq_len_in_rnn, str):
      self.use_seq_len_in_rnn = bool(du.strtobool(self.use_seq_len_in_rnn))
    self.max_seq_len = int(self.max_seq_len)

    self.epoch_count = int(self.epoch_count)
    self.samples_for_benchmark = int(self.samples_for_benchmark)

    if isinstance(self.eval_ngram_model, str):
      self.eval_ngram_model = bool(du.strtobool(self.eval_ngram_model))

    if isinstance(self.trace_hid_layer, str):
      self.trace_hid_layer = bool(du.strtobool(self.trace_hid_layer))
    #self.trace_input_id = int(self.trace_input_id)

class MapInfo(object):
  def __init__(self, map, config):
    self.map = map
    self.config = config
    self.adj_mat, self.adj_mask = self.__get_adjmat_and_mask(config.PAD_ID)
    self.dest_coord = self.__get_dest_coord()
    return

  def __get_adjmat_and_mask(self, pad_id):
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

  def __get_dest_coord(self, do_normalization = True):
    """
    Extract the coordinates of each road (end node)
    :return: ndarray with shape of (state_size, 2)
    """
    map = self.map
    dests = []
    for edge in map.edges:
      node = map.nodes[edge.endNodeId]
      dests.append([node.lat, node.lon])
    # normalize
    np_dests = np.array(dests)
    if do_normalization:
      minlat, maxlat = min(np_dests[:, 0]), max(np_dests[:, 0])
      minlon, maxlon = min(np_dests[:, 1]), max(np_dests[:, 1])
      for row in range(np_dests.shape[0]):
        np_dests[row][0] = (np_dests[row][0] - minlat) / (maxlat - minlat)
        np_dests[row][1] = (np_dests[row][1] - minlon) / (maxlon - minlon)
    return np_dests

def read_data(file_path, max_count=-1, max_seq_len = None, ratio=[0.8, 0.1, 0.1]):
  """
  Read in the route data
  data format: one traj one line with delimiter as ','
  :param file_path: path of the file to load
  :param max_count: maximum count of routes to be loaded, default is -1 which loads all routes.
  :param max_seq_len: samples longer than `max_seq_len` will be skipped.
  :param ratio: ratio[train, valid, test] for split the dataset, automatically normalized if sum(ratio) is not 1.
  :return: three lists of lists, in the order: train, valid, test.
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


if __name__ == "__main__":
  config = Config("config")
  # set log file
  timestr = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time())) # use for naming the log file
  if config.direct_stdout_to_file:
    model_str = config.model_type
    config.log_filename = "log_" + timestr + "_" + config.dataset_name + "_" + model_str + ".txt"
    config.log_file = open(config.log_filename, 'w')
    sys.stdout = config.log_file

  # process data
  routes, train, valid, test = read_data(config.dataset_path, config.data_size, config.max_seq_len, config.dataset_ratio)
  print("successfully read %d routes" % sum([len(train), len(valid), len(test)]))
  max_edge_id = max([max(route) for route in routes])
  min_edge_id = min([max(route) for route in routes])
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
  GeoPoint.AREA_LAT = 41.15 # the latitude of the testing area. In fact, any value is ok in this problem.
  roadnet = Map()
  roadnet.open(config.map_path)

  # set config
  config.set_config(routes, roadnet)
  config.printf()

  # extract map info
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
    # markov_model.train_and_eval_given_dest(train, valid, 3, 60) # 40k
    # markov_model.train_and_eval_given_dest(train, valid, 2, 10, use_fast=True) # 40k
    # markov_model.train_and_eval_given_dest(train, valid, 4, 300) # 40k
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
      if ckpt_path and tf.gfile.Exists(ckpt_path):
        print("restoring model trainable params from %s" % ckpt_path)
        model.saver.restore(sess, ckpt_path)
      else:
        if config.load_ckpt:
          print("restore model params failed")
        print("initialize all variables...")
        sess.run(tf.initialize_all_variables())

      # benchmark for testing speed
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
  input()