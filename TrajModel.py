from __future__ import print_function
import numpy as np
import math
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from geo import Map, GeoPoint
from ngram_model import N_gram_model
import time
import os
import distutils.util as du
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.client import timeline
try: # python2
    import ConfigParser as configparser
except ImportError: # python3
    import configparser
import sys


class Batch(object):
  def __init__(self, inputs = None, targets = None, masks = None,
               dests = None, seq_lens = None, adj_indices = None,
               dir_distris = None, sub_onehot_target = None):
    self.inputs = inputs # (batch, time)
    self.targets = targets # (batch, time)
    self.masks = masks # (batch, time)
    self.dests = dests # (batch)
    self.seq_lens = seq_lens # (batch)
    self.adj_indices = adj_indices # (#indices, 2)
    self.dir_distris = dir_distris # (batch, dir_granularity)
    self.sub_onehot_target = sub_onehot_target # (batch, time, max_adj_num)


def constrained_softmax_cross_entropy_loss2(outputs_, input_, target_, w_t_, b_, adj_mat_, adj_mask_,
                                           fix_seq_len, max_seq_len, config):
  """
  constrained_softmax
  :param outputs_: [batch*t, hid_dim]
  :param input_: [batch, t]
  :param target_: [batch, t]
  :param w_t_: [state_size, hid_dim]
  :param b_: [state_size]
  :param adj_mat_: [state_size, max_adj_num], adj_mat_[i][j] is the j-th adjacent state of i (include padding)
  :param adj_mask_: [state_size, max_adj_num],
         adj_mask_[i][j] represents whether adj_mat_[i][j] is an adjacent state ( = 1.0) of i or padding ( = 0.0)
  :param fix_seq_len: bool
  :param max_seq_len: int

  :return: the loss with shape [batch*t]
  """
  input_flat_ = tf.reshape(input_, [-1])  # [batch*t]
  target_flat_ = tf.reshape(target_, [-1, 1])  # [batch*t, 1]
  #with tf.device("/gpu:0"):
  sub_adj_mat_ = tf.nn.embedding_lookup(adj_mat_, input_flat_)  # [batch*t, max_adj_num]
  sub_adj_mask_ = tf.nn.embedding_lookup(adj_mask_, input_flat_)  # [batch*t, max_adj_num]

  # first column is b_
  # b_and_w_t_ = tf.concat(1, [tf.reshape(b_, [-1, 1]), w_t_])  # [state_size, hid_dim+1]
  # first column is target_
  target_and_sub_adj_mat_ = tf.concat(1, [target_flat_, sub_adj_mat_])  # [batch*t, max_adj_num+1]

  outputs_3d_ = tf.expand_dims(outputs_, 2) # [batch*t, hid_dim] -> [batch*t, hid_dim, 1]
  def step(feed_info_per_step):
    """
    (outputs_3d[i], [target[i]]+adj_states_of_i)

    :param output_adj_mat_:  ([hid_dim, 1], [max_adj_num+1])
    :return: [max_adj_num+1]
    """
    target_and_adjs_ = feed_info_per_step[1]
    output_this_step_ = feed_info_per_step[0] # [hid_dim, 1]
    sub_w_t_ = tf.nn.embedding_lookup(w_t_, target_and_adjs_) # [max_adj_num, hid_dim]
    sub_b_ = tf.nn.embedding_lookup(b_, target_and_adjs_) # [max_adj_num]
    return tf.reshape(tf.matmul(sub_w_t_, output_this_step_), [-1]) + sub_b_ # [max_adj_num]

    '''
    sub_b_and_w_t_ = tf.nn.embedding_lookup(b_and_w_t_, output_adj_mat_[1])  # [max_adj_num+1, hid_dim+1]
    # output_2d = tf.reshape(output_adj_mat_[0], [-1, 1])  # [hid_dim, 1]
    sub_w_t_ = sub_b_and_w_t_[:, 1:]  # [max_adj_num+1, hid_dim]
    sub_b_ = sub_b_and_w_t_[:, 0:1]  # [max_adj_num+1, 1]
    return tf.reshape(tf.matmul(sub_w_t_, output_adj_mat_[0]) + sub_b_, [-1])  # [max_adj_num+1]

  if fix_seq_len:
    proj_targets_and_sub_proj_outputs_ = []
    for i in range(outputs_3d_.get_shape()[0]): # loop `batch*t` times
      proj_targets_and_sub_proj_outputs_.append(step((outputs_3d_[i], target_and_sub_adj_mat_[i])))
    proj_targets_and_sub_proj_outputs_ = tf.pack(proj_targets_and_sub_proj_outputs_) # [batch*t, max_adj_num+1]
  else:
  '''
  #with tf.device("/gpu:0"):
  proj_targets_and_sub_proj_outputs_ = tf.map_fn(step, (outputs_3d_, target_and_sub_adj_mat_),
                                                   dtype=tf.float32)  # [batch*t, max_adj_num+1]

  # for numerical stability
  scales_ = tf.reduce_max(proj_targets_and_sub_proj_outputs_, 1) # [batch*t]
  scaled_proj_targets_and_sub_proj_outputs_ = tf.transpose(
    tf.sub(tf.transpose(proj_targets_and_sub_proj_outputs_), scales_)) # transpose for broadcasting [batch*t, max_adj_num+1]

  scaled_sub_proj_outputs_ = scaled_proj_targets_and_sub_proj_outputs_[:, 1:]  # [batch*t, max_adj_num]
  exp_scaled_sub_proj_outputs_ = tf.exp(scaled_sub_proj_outputs_)  # [batch*t, max_adj_num]
  fenmu_ = tf.reduce_sum(tf.mul(exp_scaled_sub_proj_outputs_, sub_adj_mask_), 1)
  log_fenmu_ = tf.log(fenmu_)  # [batch*t]
  log_fenzi_ = tf.reshape(scaled_proj_targets_and_sub_proj_outputs_[:, 0:1], [-1])  # [batch*t]
  loss_ = tf.sub(log_fenmu_, log_fenzi_)  # [batch*t] since loss is -sum(log(softmax))
  """
  exp_proj_targets_and_sub_proj_outputs_ = tf.exp(proj_targets_and_sub_proj_outputs_) # [batch*t, max_adj_num+1]
  exp_sub_proj_outputs_ = exp_proj_targets_and_sub_proj_outputs_[:,1:] # [batch*t, max_adj_num]
  fenzi_2d = exp_proj_targets_and_sub_proj_outputs_[:,0:1] # [batch*t, 1]
  fenzi_ = tf.reshape(fenzi_2d, [-1]) # [batch*t]
  fenmu_ = tf.reduce_sum(tf.mul(exp_sub_proj_outputs_, sub_adj_mask_), [1]) # [batch*t]
  softmax_ = tf.div(fenzi_, fenmu_) # [batch*t]
  loss_ = -tf.log(softmax_) # [batch*t]
  """
  print("build constrained done 2")
  return loss_

def constrained_softmax_cross_entropy_loss3(outputs_, input_, target_, w_t_, b_, adj_mat_, adj_mask_,
                                           fix_seq_len, max_seq_len, config):
  """
  constrained_softmax, fastest!
  :param outputs_: [batch*t, hid_dim], float
  :param input_: [batch, t], int
  :param target_: [batch, t], int
  :param w_t_: [state_size, hid_dim], float
  :param b_: [state_size], float
  :param adj_mat_: [state_size, max_adj_num], float, adj_mat_[i][j] is the j-th adjacent state of i (include padding)
  :param adj_mask_: [state_size, max_adj_num], float
         adj_mask_[i][j] represents whether adj_mat_[i][j] is an adjacent state ( = 1.0) of i or padding ( = 0.0)
  :param fix_seq_len: bool
  :param max_seq_len: int

  :return: the loss with shape [batch*t]
  """
  input_flat_ = tf.reshape(input_, [-1])  # [batch*t]
  target_flat_ = tf.reshape(target_, [-1, 1])  # [batch*t, 1]
  #with tf.device("/gpu:0"):
  sub_adj_mat_ = tf.nn.embedding_lookup(adj_mat_, input_flat_)  # [batch*t, max_adj_num]
  sub_adj_mask_ = tf.nn.embedding_lookup(adj_mask_, input_flat_)  # [batch*t, max_adj_num]

  # first column is b_
  # b_and_w_t_ = tf.concat(1, [tf.reshape(b_, [-1, 1]), w_t_])  # [state_size, hid_dim+1]
  # first column is target_
  target_and_sub_adj_mat_ = tf.concat(1, [target_flat_, sub_adj_mat_])  # [batch*t, max_adj_num+1]

  outputs_3d_ = tf.expand_dims(outputs_, 1) # [batch*t, hid_dim] -> [batch*t, 1, hid_dim]

  '''
  if fix_seq_len:
    proj_targets_and_sub_proj_outputs_ = []
    for i in range(outputs_3d_.get_shape()[0]): # loop `batch*t` times
      proj_targets_and_sub_proj_outputs_.append(step((outputs_3d_[i], target_and_sub_adj_mat_[i])))
    proj_targets_and_sub_proj_outputs_ = tf.pack(proj_targets_and_sub_proj_outputs_) # [batch*t, max_adj_num+1]
  else:
  '''

  sub_w_ = tf.nn.embedding_lookup(w_t_, target_and_sub_adj_mat_) # [batch*t, max_adj_num+1, hid_dim]
  sub_b_ = tf.nn.embedding_lookup(b_, target_and_sub_adj_mat_) # [batch*t, max_adj_num+1]
  sub_w_flat_ = tf.reshape(sub_w_, [-1, int(sub_w_.get_shape()[2])]) # [batch*t*max_adj_num+1, hid_dim]
  outputs_tiled_ = tf.tile(outputs_3d_, [1, tf.shape(adj_mat_)[1]+1, 1]) # [batch*t, max+adj_num+1, hid_dim]
  outputs_tiled_ = tf.reshape(outputs_tiled_, [-1, int(outputs_tiled_.get_shape()[2])]) # [batch*t*max_adj_num+1, hid_dim]
  target_logit_and_sub_logits_ = tf.reshape(tf.reduce_sum(tf.mul(sub_w_flat_, outputs_tiled_), 1),
                                                  [-1, tf.shape(adj_mat_)[1]+1]) # [batch*t, max_adj_num+1]

  # for numerical stability
  scales_ = tf.reduce_max(target_logit_and_sub_logits_, 1) # [batch*t]
  scaled_target_logit_and_sub_logits_ = tf.transpose(
    tf.sub(tf.transpose(target_logit_and_sub_logits_), scales_)) # transpose for broadcasting [batch*t, max_adj_num+1]

  scaled_sub_logits_ = scaled_target_logit_and_sub_logits_[:, 1:]  # [batch*t, max_adj_num]
  exp_scaled_sub_logits_ = tf.exp(scaled_sub_logits_)  # [batch*t, max_adj_num]
  fenmu_ = tf.reduce_sum(tf.mul(exp_scaled_sub_logits_, sub_adj_mask_), 1) # [batch*t]
  log_fenmu_ = tf.log(fenmu_)  # [batch*t]
  log_fenzi_ = tf.reshape(scaled_target_logit_and_sub_logits_[:, 0:1], [-1])  # [batch*t]
  loss_ = tf.sub(log_fenmu_, log_fenzi_)  # [batch*t] since loss is -sum(log(softmax))
  print("build constrained done 3")

  # tf.tile(tf.expand_dims(fenmu_, 1), [1, int(adj_mat_.get_shape()[1])]) # [batch*t, max_adj_num]
  max_prediction_ = tf.one_hot(tf.argmax(exp_scaled_sub_logits_ * sub_adj_mask_, 1),
                               int(adj_mat_.get_shape()[1]), dtype=config.float_type) # [batch*t, max_adj_num]
  return loss_, max_prediction_


def constrained_softmax_cross_entropy_loss(outputs_, input_, target_, w_t_, b_, adj_mat_, adj_mask_,
                                           fix_seq_len, max_seq_len, config):
  """
  constrained_softmax
  :param outputs_: [batch*t, hid_dim]
  :param input_: [batch, t]
  :param target_: [batch, t]
  :param w_t_: [state_size, hid_dim]
  :param b_: [state_size]
  :param adj_mat_: [state_size, max_adj_num], adj_mat_[i][j] is the j-th adjacent state of i (include padding)
  :param adj_mask_: [state_size, max_adj_num],
         adj_mask_[i][j] represents whether adj_mat_[i][j] is an adjacent state ( = 1.0) of i or padding ( = 0.0)
  :param fix_seq_len: bool
  :param max_seq_len: int

  :return: the loss with shape [batch*t]
  """
  input_flat_ = tf.reshape(input_, [-1])  # [batch*t]
  target_flat_ = tf.reshape(target_, [-1])  # [batch*t]
  sub_adj_mat_ = tf.nn.embedding_lookup(adj_mat_, input_flat_)  # [batch*t, max_adj_num]
  sub_adj_mask_ = tf.nn.embedding_lookup(adj_mask_, input_flat_)  # [batch*t, max_adj_num]

  # first column is b_
  b_and_w_t_ = tf.concat(1, [tf.reshape(b_, [-1, 1]), w_t_])  # [state_size, hid_dim+1]
  # first column is target_
  target_and_sub_adj_mat_ = tf.concat(1, [tf.reshape(target_flat_, [-1, 1]), sub_adj_mat_])  # [batch*t, max_adj_num+1]

  outputs_3d_ = tf.expand_dims(outputs_, 2) # [batch*t, hid_dim] -> [batch*t, hid_dim, 1]
  def step(output_adj_mat_):
    """
    (outputs_3d[i], [target[i]]+adj_states_of_i)

    :param output_adj_mat_:  ([hid_dim, 1], [max_adj_num+1])
    :return: [max_adj_num+1]
    """
    sub_b_and_w_t_ = tf.nn.embedding_lookup(b_and_w_t_, output_adj_mat_[1])  # [max_adj_num+1, hid_dim+1]
    # output_2d = tf.reshape(output_adj_mat_[0], [-1, 1])  # [hid_dim, 1]
    sub_w_t_ = sub_b_and_w_t_[:, 1:]  # [max_adj_num+1, hid_dim]
    sub_b_ = sub_b_and_w_t_[:, 0:1]  # [max_adj_num+1, 1]
    return tf.reshape(tf.matmul(sub_w_t_, output_adj_mat_[0]) + sub_b_, [-1])  # [max_adj_num+1]

  if fix_seq_len:
    proj_targets_and_sub_proj_outputs_ = []
    for i in range(outputs_3d_.get_shape()[0]): # loop `batch*t` times
      proj_targets_and_sub_proj_outputs_.append(step((outputs_3d_[i], target_and_sub_adj_mat_[i])))
    proj_targets_and_sub_proj_outputs_ = tf.pack(proj_targets_and_sub_proj_outputs_) # [batch*t, max_adj_num+1]
  else:
    proj_targets_and_sub_proj_outputs_ = tf.map_fn(step, (outputs_3d_, target_and_sub_adj_mat_),
                                                   dtype=tf.float32)  # [batch*t, max_adj_num+1]

  # for numerical stability
  scales_ = tf.reduce_max(proj_targets_and_sub_proj_outputs_, 1)  # [batch*t]
  scaled_proj_targets_and_sub_proj_outputs_ = tf.transpose(
    tf.sub(tf.transpose(proj_targets_and_sub_proj_outputs_), scales_)) # transpose for broadcasting [batch*t, max_adj_num+1]

  scaled_sub_proj_outputs_ = scaled_proj_targets_and_sub_proj_outputs_[:, 1:]  # [batch*t, max_adj_num]
  exp_scaled_sub_proj_outputs_ = tf.exp(scaled_sub_proj_outputs_)  # [batch*t, max_adj_num]
  log_fenmu_ = tf.log(tf.reduce_sum(tf.mul(exp_scaled_sub_proj_outputs_, sub_adj_mask_), [1]))  # [batch*t]
  log_fenzi_ = tf.reshape(scaled_proj_targets_and_sub_proj_outputs_[:, 0:1], [-1])  # [batch*t]
  loss_ = tf.sub(log_fenmu_, log_fenzi_)  # [batch*t] since loss is -sum(log(softmax))
  """
  exp_proj_targets_and_sub_proj_outputs_ = tf.exp(proj_targets_and_sub_proj_outputs_) # [batch*t, max_adj_num+1]
  exp_sub_proj_outputs_ = exp_proj_targets_and_sub_proj_outputs_[:,1:] # [batch*t, max_adj_num]
  fenzi_2d = exp_proj_targets_and_sub_proj_outputs_[:,0:1] # [batch*t, 1]
  fenzi_ = tf.reshape(fenzi_2d, [-1]) # [batch*t]
  fenmu_ = tf.reduce_sum(tf.mul(exp_sub_proj_outputs_, sub_adj_mask_), [1]) # [batch*t]
  softmax_ = tf.div(fenzi_, fenmu_) # [batch*t]
  loss_ = -tf.log(softmax_) # [batch*t]
  """
  print("build constrained done")
  return loss_


class TrajModel(object):
  def __init__(self, train_phase, config, data, model_scope = None, map = None, mapInfo = None): #dir_distris = None, adj_mat_dense = None, adj_mat=None, adj_mask=None):
    self.debug_tensors = {} # set this to any tensor you want to evaluate for debugging and just run `speed_benchmark()`
    self.model_scope = model_scope
    self.loss_dict = {}
    self.data = data
    self.train_phase = train_phase
    self.config = config
    self.map = map
    self.mapInfo = mapInfo
    self.lr_ = tf.placeholder(config.float_type, name="lr")
    self.dir_distris = mapInfo.dir_distris
    self.adj_mat = mapInfo.adj_mat
    self.adj_mask = mapInfo.adj_mask
    self.dest_coord = mapInfo.dest_coord
    self.logits_mask__ = None
    self.sub_onehot_targets_ = None
    self.trace_dict = {}
    self.trace_items = {} # k = target id, v = list of layer output

    self.dest_coord_ = tf.constant(self.dest_coord, dtype=config.float_type, name="dest_coord") # [state_size, 2]
    self.dests_label_ = tf.placeholder(config.int_type, shape=[config.batch_size], name="dests_label")
    self.seq_len_ = tf.placeholder(config.int_type, shape=[config.batch_size], name="seq_len")
    if config.fix_seq_len:
      self.inputs_ = tf.placeholder(config.int_type, shape=[config.batch_size, config.max_seq_len], name="inputs")
      self.mask_ = tf.placeholder(config.float_type, shape=[config.batch_size, config.max_seq_len], name="mask")
      self.max_t_ = config.max_seq_len
      self.targets_ = tf.placeholder(config.int_type, shape=[config.batch_size, config.max_seq_len], name="targets")
    else:
      self.inputs_ = tf.placeholder(config.int_type, shape=[config.batch_size, None], name="inputs")
      self.mask_ = tf.placeholder(config.float_type, shape=[config.batch_size, None], name="mask")
      self.max_t_ = tf.shape(self.inputs_)[1]
      self.targets_ = tf.placeholder(config.int_type, shape=[config.batch_size, None], name="targets")
    self.sub_onehot_targets_ = tf.placeholder(config.float_type, (config.batch_size, None, self.adj_mat.shape[1]))  # [batch, t, max_adj_num]

    if config.constrained_softmax_strategy == 'adjmat_adjmask':
      self.adj_mat_ = tf.constant(self.adj_mat, config.int_type, name="adj_mat")
      self.adj_mask_ = tf.constant(self.adj_mask, config.float_type, name="adj_mask")

    if config.predict_dir:
      # build direction model
      if config.encoder_decoder == 'end2end':
        self.build_end2end_encoder_decoder(train_phase, False)
      elif config.encoder_decoder == 'end2end_multitask':
        self.build_end2end_encoder_decoder(train_phase, True)
      elif config.encoder_decoder == 'decoder':
        self.build_encoder_decoder(train_phase)
      elif config.encoder_decoder == 'encoder':
        self.build_encoder(train_phase, False, self.dir_distris)
      return
    else:
      # build traditional RNN+xent model
      self.build_normal(train_phase)
  def build_max_predict_loss(self, max_prediction_flat_, targets_flat_, input_onehot):
    if not input_onehot:
      correct_count_ = tf.reduce_sum(tf.cast(tf.equal(max_prediction_flat_, targets_flat_), self.config.float_type)\
                       * tf.reshape(self.mask_, [-1]))
    else:
      correct_count_ = tf.reduce_sum(tf.reduce_sum(max_prediction_flat_ * targets_flat_, 1) *
                                     tf.reshape(self.mask_, [-1]))
    tot_count_ = tf.cast(tf.reduce_sum(self.seq_len_), self.config.float_type)
    max_prediction_acc_ = correct_count_ / tot_count_

    """
    # for debug
    self.debug_tensors["sub_adj_mask"] = sub_adj_mask_
    self.debug_tensors["log_probs"] = log_probs_
    self.debug_tensors["probs"] = tf.exp(log_probs_)
    self.debug_tensors["sub_onehot_targets_flat"] = sub_onehot_targets_flat
    self.debug_tensors["max_prediction"] = max_prediction_
    self.debug_tensors["seq_mask"] = tf.reshape(seq_mask_, [-1])
    """
    self.loss_dict["max_prediction_acc"] = max_prediction_acc_  # add to loss_dict for show

  def build_input(self, train_phase, input_label, input_dest=False, dest_label_=None, dest_emb=False, var_scope="input"):
    """

    :param train_phase: bool
    :param input_label: [batch, t] int
    :param input_dest: bool, whether to use destination information
    :param dest_label_: [batch] int
    :param dest_emb: bool, whether to use embedding for destination
    :param var_scope: string
    :return: inputs_: [batch, t, emb], if `input_dest` is False
    """
    config = self.config
    with tf.variable_scope(var_scope):
      # construct embeddings
      if config.pretrained_input_emb_path != '': # load pretrained embeddings (such as word2vec)
        pretrained_emb = np.loadtxt(config.pretrained_input_emb_path, delimiter=',')
        pretrained_emb_ = tf.constant(pretrained_emb, config.float_type)
        emb_ = tf.get_variable("embedding", dtype=config.float_type, initializer=pretrained_emb_)
        print("init emb by pretraining.")
        #print(pretrained_emb)
        #raw_input()
      else:
        emb_ = tf.get_variable("embedding", [config.state_size, config.emb_dim], dtype=config.float_type)
      emb_inputs_ = tf.nn.embedding_lookup(emb_, input_label, name="emb_inputs")  # batch_size x time_steps x emb_dim
      if train_phase and config.keep_prob < 1:
        emb_inputs_ = tf.nn.dropout(emb_inputs_, keep_prob=config.keep_prob, name="dropout_emb_inputs")

      # with destination input information
      if input_dest:
        if dest_emb:
          self.dest_emb_ = tf.get_variable("dest_emb", [config.state_size, config.emb_dim], dtype=config.float_type)
        else:
          self.dest_emb_ = self.dest_coord_
        dest_inputs_ = tf.tile(tf.expand_dims(tf.nn.embedding_lookup(self.dest_emb_, dest_label_), 1), [1, self.max_t_, 1])  # [batch, t, dest_emb]

        inputs_ = tf.concat(2, [emb_inputs_, dest_inputs_], "input_with_dest")
      else:
        inputs_ = emb_inputs_
      return inputs_

  def build_encoder_decoder(self, train_phase):
    self.encoder_outputs_, err_dir_ = self.build_encoder(train_phase=False, encoder_phase=True, dir_distris=self.dir_distris)
    self.build_decoder(train_phase, decoder_phase=False)

  def build_trainer(self, object_loss_, params_to_train):
    # compute grads and update params
    # params = tf.trainable_variables()
    config = self.config
    if config.opt == 'sgd':
      opt = tf.train.GradientDescentOptimizer(self.lr_)
    elif config.opt == 'rmsprop':
      opt = tf.train.RMSPropOptimizer(self.lr_, config.lr_decay)
    elif config.opt == 'adam':
      opt = tf.train.AdamOptimizer(self.lr_)
    else:
      raise Exception("config.opt should be correctly defined.")
    grads = tf.gradients(object_loss_, params_to_train)
    clipped_grads, norm = tf.clip_by_global_norm(grads, config.max_grad_norm)
    self.update_op = opt.apply_gradients(zip(clipped_grads, params_to_train))

  def step_pretrained_encoder_and_decoder(self, sess, batch, eval_op=None):
    overall_feed_dict = self.feed(batch)
    encoder_fetch_dict = self.fetch_encoder()
    # run sess
    encoder_vals = sess.run(encoder_fetch_dict, overall_feed_dict)
    decoder_feed_dict = self.feed_decoder(overall_feed_dict, encoder_vals["encoder_outputs"])
    decoder_fetch_dict = self.fetch(eval_op)
    vals = sess.run(decoder_fetch_dict, decoder_feed_dict)
    return vals

  def build_encoder(self, train_phase, encoder_phase, dir_distris, var_scope ="encoder"):
    config = self.config
    with tf.variable_scope(var_scope):
      self.deg_buckets = tf.constant(np.linspace(0.0, 360.0, config.dir_granularity+1, dtype=np.float32)[0:-1],
                                     dtype=config.float_type)
      # construct embeddings
      emb_inputs_ = self.build_input(train_phase, self.inputs_, config.input_dest, self.dests_label_, config.dest_emb)

      # construct rnn
      rnn_outputs_ = self.build_rnn(emb_inputs_, train_phase) # [batch, time, hid_dim]
      outputs_flat_ = tf.reshape(rnn_outputs_, [-1, int(rnn_outputs_.get_shape()[2])])  # [batch*t, hid_dim]

      # construct targets
      dir_distris_ = tf.constant(dir_distris, dtype=config.float_type, name="dir_distris")
      dir_targets_ = tf.nn.embedding_lookup(dir_distris_, self.targets_)
      targets_flat_ = tf.reshape(dir_targets_, [-1, int(dir_targets_.get_shape()[2])]) # [batch*t, dir]

      # hidden to output
      #w_dir_ = tf.get_variable("wdir", [config.hidden_dim, config.dir_granularity], dtype=config.float_type)
      w_dir_ = tf.get_variable("wdir", [int(outputs_flat_.get_shape()[1]), int(targets_flat_.get_shape()[1])],
                               dtype=config.float_type) # [hid_dim, dir]
      b_dir_ = tf.get_variable("bdir", [int(targets_flat_.get_shape()[1])], dtype=config.float_type) # [dir]
      logits_ = tf.matmul(outputs_flat_, w_dir_) + b_dir_ # [batch*t, dir]
      softmax_ = tf.nn.softmax(logits_) # [batch*t, dir]

      fwd_params = [v for v in tf.all_variables() if v.name.startswith(self.model_scope + "/" + var_scope)] #TODO
      if config.use_v2_saver:
        self.encoder_forward_saver = tf.train.Saver(fwd_params, max_to_keep=config.max_ckpt_to_keep,
                                                    write_version=saver_pb2.SaverDef.V2)
      else:
        self.encoder_forward_saver = tf.train.Saver(fwd_params, max_to_keep=config.max_ckpt_to_keep,
                                                    write_version=saver_pb2.SaverDef.V1)

      # compute loss
      mask_flat_ = tf.reshape(self.mask_, [-1]) # [batch*t]
      loss_vec_ = tf.mul(tf.nn.softmax_cross_entropy_with_logits(logits_, targets_flat_), mask_flat_) # [batch*t]
      loss_ = tf.reduce_sum(loss_vec_) / config.batch_size
      expec_dir_ = tf.reduce_sum(tf.mul(softmax_, self.deg_buckets), 1) # [batch*t]
      true_dir_ = tf.reduce_sum(tf.mul(targets_flat_, self.deg_buckets), 1) # [batch*t]
      err_dir_vec = tf.abs(tf.mul(expec_dir_ - true_dir_, mask_flat_))
      err_dir_ = tf.reduce_sum(err_dir_vec) / tf.reduce_sum(tf.cast(self.seq_len_,config.float_type))
      if encoder_phase:
        # return softmax_, err_dir_
        return logits_, err_dir_
        # return tf.nn.relu(logits_), err_dir_ # return logits_ will be better than normalized logits_ (softmax_)
        # return outputs_flat_, err_dir_

      self.loss_dict = {}
      self.loss_dict["loss"] = loss_
      self.loss_dict["err_dir"] = err_dir_

      #self.debug_tensors["e_dir"] = tf.reshape(expec_dir_, [config.batch_size, -1])[0]
      #self.debug_tensors["t_dir"] = tf.reshape(true_dir_, [config.batch_size, -1])[0]
      #self.debug_tensors["e_dir_distri"] = tf.nn.softmax(logits_) # [batch*t, dir]
      #self.debug_tensors["t_dir_distri"] = targets_flat_  # [batch*t, dir]
      #self.debug_tensors["seq_mask"] = self.mask_[0]
      if train_phase:
        params = [v for v in tf.all_variables()
                  if v.name.startswith(self.model_scope + "/" + var_scope)
                  or v.name.startswith("Train/" + self.model_scope + "/" + var_scope)]  # TODO
        self.build_trainer(self.loss_dict["loss"], params)
        params = [v for v in tf.all_variables()
                  if v.name.startswith(self.model_scope + "/" + var_scope)
                  or v.name.startswith("Train/" + self.model_scope + "/" + var_scope)]   # TODO
        if config.use_v2_saver:
          self.saver = tf.train.Saver(params, max_to_keep=config.max_ckpt_to_keep,
                                      write_version=saver_pb2.SaverDef.V2)
        else:
          self.saver = tf.train.Saver(params, max_to_keep=config.max_ckpt_to_keep,
                                      write_version=saver_pb2.SaverDef.V1)

  def build_decoder(self, train_phase, decoder_phase, decoder_inputs_ = None, var_scope ="decoder"):
    config = self.config
    if decoder_inputs_ is not None: # if none, input is from feed dict which is obtained from the outputs of encoder
      self.decoder_inputs_ = decoder_inputs_
    else:
      self.decoder_inputs_ = tf.placeholder(config.float_type, [None, config.dir_granularity], name="decoder_inputs") # [batch*t, dir]

    with tf.variable_scope(var_scope):
      self.w_dir_dec_ = tf.get_variable("w_dir_dec", [config.state_size, self.adj_mat_.get_shape()[1], decoder_inputs_.get_shape()[1]],
                                        dtype=config.float_type)
      self.b_dir_dec_ = tf.get_variable("b_dir_dec", [config.state_size, self.adj_mat_.get_shape()[1]])
      # dropout
      if train_phase and config.keep_prob < 1:
        w_dir_dec_ = tf.nn.dropout(self.w_dir_dec_, keep_prob=config.keep_prob, name="dropout_wdir_dec")
      else:
        w_dir_dec_ = self.w_dir_dec_

      b_dir_dec_ = self.b_dir_dec_

      #compute loss
      def constrained_softmax_cross_entropy_loss_with_individual_weights(inputs_, encoder_outputs_ ,
                                                                         sub_onehot_targets_, seq_mask_,
                                                                         w_dec_, b_dec_, adj_mask_):
        """
        :param inputs_: [batch, t], int
        :param decoder_inputs_: [batch*t, dir], float
        :param sub_onehot_targets_: [batch, t, max_adj_num], float
        :param seq_mask_: [batch, t], float
        :param w_dec_: [state, max_adj_num, dir], float
        :param b_dec_: [state, max_adj_num], float
        :param adj_mask_: [state_size, max_adj_num], ,float
               adj_mask_[i][j] represents whether adj_mat_[i][j] is an adjacent state ( = 1.0) of i or padding ( = 0.0)

        :return: the xent loss with shape [batch*t]
        """

        inputs_flat_ = tf.reshape(inputs_, [-1])  # [batch*t]
        sub_adj_mask_ = tf.nn.embedding_lookup(adj_mask_, inputs_flat_)  # [batch*t, max_adj_num]

        encoders_3d_ = tf.expand_dims(encoder_outputs_, 1)  # [batch*t, dir] -> [batch*t, 1, dir]

        sub_w_ = tf.nn.embedding_lookup(w_dec_, inputs_flat_)  # [batch*t, max_adj_num, dir]
        sub_b_ = tf.nn.embedding_lookup(b_dec_, inputs_flat_)  # [batch*t, max_adj_num]

        encoders_tiled_ = tf.tile(encoders_3d_, [1, int(adj_mask_.get_shape()[1]), 1])  # [batch*t, max_adj_num, dir] TODO, maybe fixed
        logits_ = tf.reduce_sum(tf.mul(sub_w_, encoders_tiled_), 2) + sub_b_ # [batch*t, max_adj_num]

        # for numerical stability
        scales_ = tf.reduce_max(logits_, 1)  # [batch*t]
        scaled_logits = tf.transpose(tf.sub(tf.transpose(logits_), scales_))  # transpose for broadcasting [batch*t, max_adj_num]
        exp_scaled_logits_ = tf.exp(scaled_logits)  # [batch*t, max_adj_num]
        normalizations_ = tf.reduce_sum(tf.mul(exp_scaled_logits_, sub_adj_mask_), 1)  # [batch*t]
        log_probs_ = tf.transpose(tf.transpose(scaled_logits) - tf.log(normalizations_)) # [batch*t, max_adj_num]

        sub_onehot_targets_flat = tf.reshape(sub_onehot_targets_, [-1, int(sub_onehot_targets_.get_shape()[2])]) # [batch*t, max_adj_num]

        xent_ = - tf.reduce_sum(sub_onehot_targets_flat * log_probs_, 1) * tf.reshape(seq_mask_, [-1]) # [batch*t]

        # max prediction acc computation
        max_prediction_ = tf.one_hot(tf.argmax(tf.exp(log_probs_) * sub_adj_mask_, dimension=1), int(adj_mask_.get_shape()[1]),
                                      dtype=config.float_type)  # [batch*t, max_adj_num]


        return xent_, max_prediction_

      xent_, max_prediction_ = constrained_softmax_cross_entropy_loss_with_individual_weights(self.inputs_, self.decoder_inputs_,
                                                                             self.sub_onehot_targets_, self.mask_,
                                                                             w_dir_dec_, b_dir_dec_, self.adj_mask_)
      self.build_max_predict_loss(max_prediction_,
                                  tf.reshape(self.sub_onehot_targets_, [-1, int(self.sub_onehot_targets_.get_shape()[2])]),
                                  input_onehot=True)
      loss_p_ = tf.reduce_sum(xent_) / config.batch_size
      if config.decoder_regularizer > 0 and train_phase:
        if config.decoder_keep_prob < 1.0:
          print("Warning: you'd better only choose one between dropout and L2-regularizer")
        print("Use L2 regularizer on w in decoder")
        loss_ = loss_p_ + config.decoder_regularizer * tf.nn.l2_loss(w_dir_dec_)
      else:
        loss_ = loss_p_

      if decoder_phase:
        return loss_, loss_p_
      self.loss_dict["loss"] = loss_
      self.loss_dict["loss_p"] = loss_p_

      #build saver and trainer
      params = [v for v in tf.all_variables() if v.name.startswith(self.model_scope + "/" + var_scope)]  # TODO

      if train_phase:
        self.build_trainer(self.loss_dict["loss"], params)
        params = [v for v in tf.all_variables()
                  if v.name.startswith(self.model_scope + "/" + var_scope)
                  or v.name.startswith("Train/" + self.model_scope + "/" + var_scope)]  # TODO
        if config.use_v2_saver:
          self.saver = tf.train.Saver(params, max_to_keep=config.max_ckpt_to_keep,
                                      write_version=saver_pb2.SaverDef.V2)
        else:
          self.saver = tf.train.Saver(params, max_to_keep=config.max_ckpt_to_keep,
                                      write_version=saver_pb2.SaverDef.V1)

  def build_end2end_encoder_decoder(self, train_phase, multitask):
    config = self.config
    self.encoder_outputs_, err_dir_ = self.build_encoder(train_phase=train_phase,
                                               encoder_phase=True, dir_distris=self.dir_distris)
    loss_regularized_, loss_p_ = self.build_decoder(train_phase=train_phase,
                                    decoder_phase=True, decoder_inputs_=self.encoder_outputs_)
    if config.trace_hid_layer:
      self.trace_dict["encoder_outputs_"+str(config.trace_input_id)] = self.encoder_outputs_
    if multitask:
      loss_ = 1.0 * loss_regularized_ + err_dir_
    else:
      loss_ = loss_regularized_
    self.loss_dict["loss"] = loss_
    self.loss_dict["loss_p"] = loss_p_
    self.loss_dict["err_dir"] = err_dir_
    # compute grads and update params
    self.build_trainer(self.loss_dict["loss"], tf.trainable_variables())
    if config.use_v2_saver:
      self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=config.max_ckpt_to_keep,
                                  write_version=saver_pb2.SaverDef.V2)
    else:
      self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=config.max_ckpt_to_keep,
                                  write_version=saver_pb2.SaverDef.V1)
  def build_normal(self, train_phase, var_scope = "normal"):
    config = self.config
    # construct embeddings
    emb_inputs_ = self.build_input(train_phase, self.inputs_, config.input_dest, self.dests_label_, config.dest_emb)

    # construct rnn
    rnn_outputs_ = self.build_rnn(emb_inputs_, train_phase)  # [batch, time, hid_dim]
    outputs_flat_ = tf.reshape(rnn_outputs_, [-1, int(rnn_outputs_.get_shape()[2])])  # [batch*t, hid_dim]

    # construct output losses
    if train_phase:
      self.build_rnn_to_xent_loss(outputs_flat_, config.build_multitask_in_train,
                                                   config.use_constrained_softmax_in_train,
                                                   config.constrained_softmax_strategy,
                                                   config.build_unconstrained_in_train)
    else:
      self.build_rnn_to_xent_loss(outputs_flat_, config.build_multitask_in_test,
                                                   config.use_constrained_softmax_in_test,
                                                   config.constrained_softmax_strategy,
                                                   config.build_unconstrained_in_test)

    # compute grads and update params
    self.build_trainer(self.loss_dict["loss"], tf.trainable_variables())
    if config.use_v2_saver:
      self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=config.max_ckpt_to_keep,
                                  write_version=saver_pb2.SaverDef.V2)
    else:
      self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=config.max_ckpt_to_keep,
                                  write_version=saver_pb2.SaverDef.V1)

  def build_rnn_to_xent_loss(self, outputs_flat_, build_multitask,
                             use_constrained_softmax, constrained_softmax_strategy,
                             build_unconstrained):
    """
    prediction model  P(r_{t+1}|r_{1:t})
    destination model P(d_{t+1}|r_{1:t})
    :return: A dict with some losses you may care.
    """
    config = self.config
    # projection to output space
    wp_ = tf.get_variable("wp", [int(outputs_flat_.get_shape()[1]), config.state_size],
                          dtype=config.float_type)  # [hid_dim, state_size]
    bp_ = tf.get_variable("bp", [config.state_size], dtype=config.float_type)  # [state_size]

    wd_ = tf.get_variable("wd", [int(outputs_flat_.get_shape()[1]), config.state_size],
                          dtype=config.float_type)  # [hid_dim, state_size]
    bd_ = tf.get_variable("bd", [config.state_size], dtype=config.float_type)  # [state_size]

    def build_loss_d_standard(outputs_flat_):
      if config.fix_seq_len:
        targets_d_ = tf.tile(tf.expand_dims(self.dests_label_, 1), [1, config.max_seq_len])  # [batch] -> [batch, t]
      else:
        targets_d_ = tf.tile(tf.expand_dims(self.dests_label_, 1), [1, self.max_t_])  # [batch] -> [batch, t]
      targets_d_flat_ = tf.reshape(targets_d_, [-1])  # [batch, t] -> [batch*t]

      # hidden to output
      logits_d_ = tf.matmul(outputs_flat_, wd_) + bd_  # [batch*t, state_size]
      loss_d_vec_ = tf.nn.sparse_softmax_cross_entropy_with_logits(logits_d_, targets_d_flat_)  # [batch*t]
      masked_loss_d_ = tf.mul(loss_d_vec_, tf.reshape(self.mask_, [-1]))  # [batch*t]
      unmasked_loss_d_ = tf.reduce_sum(loss_d_vec_) / config.batch_size
      loss_d_ = tf.reduce_sum(masked_loss_d_) / config.batch_size

      # loss given s
      logits_d_given_s_ = tf.reshape(logits_d_, [config.batch_size, -1, config.state_size])[:, 0,
                          :]  # [batch, state_size]
      targets_d_given_s_ = targets_d_[:, 0]  # [batch]
      loss_d_given_s_vec_ = tf.nn.sparse_softmax_cross_entropy_with_logits(logits_d_given_s_,
                                                                           targets_d_given_s_)  # [batch]

      return loss_d_, unmasked_loss_d_, loss_d_given_s_vec_

    def build_loss_p_standard(outputs_flat_):
      targets_p_flat_ = tf.reshape(self.targets_, [-1])  # [batch*t]

      # hidden to output
      logits_p_ = tf.matmul(outputs_flat_, wp_) + bp_  # [batch*t, state_size]
      loss_p_vec_ = tf.nn.sparse_softmax_cross_entropy_with_logits(logits_p_, targets_p_flat_)  # [batch*t]

      masked_loss_p_ = tf.mul(loss_p_vec_, tf.reshape(self.mask_, [-1]))  # [batch*t]
      unmasked_loss_p_ = tf.reduce_sum(loss_p_vec_) / config.batch_size
      loss_p_ = tf.reduce_sum(masked_loss_p_) / config.batch_size

      #onehot_targets_ = tf.one_hot(self.targets_, config.state_size,
      #                             dtype=config.float_type)  # [batch, t, state_size]
      #max_prediction_ = tf.one_hot(tf.argmax(logits_p_, dimension=1), config.state_size,
      #                             dtype=config.float_type)  # [batch*t, state_size]
      max_prediction_label_ = tf.cast(tf.argmax(logits_p_, dimension=1), config.int_type) # [batch*t], int64->int32

      self.build_max_predict_loss(max_prediction_label_, targets_p_flat_, False)

      return loss_p_, unmasked_loss_p_

    def build_loss_p_constrained_use_sparse_tensor(outputs_flat_):
      """
      Build the constrained cross-entropy loss for `outputs_flat_`.

      The strategy is to use a sparse tensor `self.logits_mask__` to directly represent the logits mask.
      This approach is about 3 times faster than `build_loss_p_constrained_use_adjmat_adjmask()`.
      Instead of using tf.nn.sparse_softmax_cross_entropy_with_logits(),
      it generates an one-hot target tensor with shape [batch*t, state_size] and computes the cross-entropy
      by sparse tensor ops.

      :param outputs_flat_: [batch*t, hid_dim], flattened outputs of rnn
      :return: loss_p_: 1-D tensor
               unmasked_loss_p_: 1-D tensor, denotes the loss if sequence mask is not applied.
      """
      # hidden to output
      logits_p_ = tf.matmul(outputs_flat_, wp_) + bp_  # [batch*t, state_size]
      self.logits_mask__ = tf.sparse_placeholder(tf.float32, name="logits_mask")  # "__" to indicate a sparse tensor
      logits_p_constrained__ = tf.sparse_softmax(self.logits_mask__ * logits_p_)  # [batch*t, state_size], sparse

      # construct one-hot targets
      targets_p_flat_ = tf.reshape(self.targets_, [-1])  # [batch*t]
      onehot_targets_ = tf.one_hot(targets_p_flat_, config.state_size,
                                   dtype=config.float_type)  # [batch*t, state_size]

      xent__ = logits_p_constrained__ * onehot_targets_  # [batch*t, state_size], sparse
      loss_p_vec_ = -tf.log(tf.sparse_reduce_sum(xent__, 1))  # [batch*t]
      masked_loss_p_ = tf.mul(loss_p_vec_, tf.reshape(self.mask_, [-1]))  # [batch*t]
      unmasked_loss_p_ = tf.reduce_sum(loss_p_vec_) / config.batch_size
      loss_p_ = tf.reduce_sum(masked_loss_p_) / config.batch_size

      return loss_p_, unmasked_loss_p_

    def build_loss_p_constrained_use_adjmat_dense(outputs_flat_):
      """

      Build the constrained cross-entropy loss for `outputs_flat_`.
      The strategy is to use `self.adj_mat_dense` to represent the transition mask of each state
      and use `tf.nn.embedding_lookup(inputs_flat, self.adj_mat_dense)` to get the logits mask.
      Note that this approach will consume about 1.5GB memory if `state_size` is about 20,000.
      Remember to add `self.adj_mat_dense` in the `feed_dict` if choose this strategy of constrained softmax.

      [TODO] This function has not been tested yet.

      :param outputs_flat_: [batch*t, hid_dim], flattened outputs of rnn
      :return: loss_p_: 1-D tensor
               unmasked_loss_p_: 1-D tensor, denotes the loss if sequence mask is not applied.
      """
      adj_mat_dense = None # TODO
      self.adj_mat_dense = tf.constant(adj_mat_dense, config.float_type, name="adj_mat_dense")
      targets_p_flat_ = tf.reshape(self.targets_, [-1])  # [batch*t]
      inputs_flat = tf.reshape(self.inputs_, [-1])  # [batch*t]
      # hidden to output
      logits_p_ = tf.matmul(outputs_flat_, wp_) + bp_  # [batch*t, state_size]
      logits_p_mask = tf.nn.embedding_lookup(inputs_flat, self.adj_mat_dense)  # [batch*t, state_size]
      constrained_logits_p_ = tf.mul(logits_p_, logits_p_mask)
      loss_p_vec_ = tf.nn.sparse_softmax_cross_entropy_with_logits(constrained_logits_p_,
                                                                   targets_p_flat_)  # [batch*t]

      masked_loss_p_ = tf.mul(loss_p_vec_, tf.reshape(self.mask_, [-1]))  # [batch*t]
      unmasked_loss_p_ = tf.reduce_sum(loss_p_vec_) / config.batch_size
      loss_p_ = tf.reduce_sum(masked_loss_p_) / config.batch_size

      return loss_p_, unmasked_loss_p_

    def build_loss_p_constrained_use_adjmat_adjmask(outputs_flat_):
      """
      Build the constrained cross-entropy loss for `outputs_flat_`.

      The strategy is to use adjmat to record all adjacent states of each state
      and pad them to the one with the maximum #adjacents.
      According to the adjacent states of current input, it will pick up corresponding `w`s and `b`s
      which are constructed as 'sub_w' and 'sub_b' in a 2-D tensor with the shape (max_adj_num, hid_dim) and (hid_dim).
      Then the logits will be computed for each state (use tf.map() function)
      which may be the reason why this strategy is much slower than `build_loss_p_constrained_use_sparse_tensor()`

      TODO: documents should be updated!
      :param outputs_flat_: [batch*t, hid_dim], flattened outputs of rnn
      :return: loss_p_: 1-D tensor
               unmasked_loss_p_: 1-D tensor, denotes the loss if sequence mask is not applied.
      """
      wp_t_ = tf.transpose(wp_)  # [state_size, hid_dim]
      loss_p_vec_, max_prediction_ = constrained_softmax_cross_entropy_loss3(outputs_flat_, self.inputs_,
                                                            self.targets_,
                                                            wp_t_, bp_,
                                                            self.adj_mat_, self.adj_mask_,
                                                            config.fix_seq_len,
                                                            config.max_seq_len, self.config)  # [batch*t]
      self.build_max_predict_loss(max_prediction_,
                                  tf.reshape(self.sub_onehot_targets_,
                                             [-1, int(self.sub_onehot_targets_.get_shape()[2])]),
                                  input_onehot=True)
      masked_loss_p_ = tf.mul(loss_p_vec_, tf.reshape(self.mask_, [-1]))  # [batch*t]
      loss_p_ = tf.reduce_sum(masked_loss_p_) / config.batch_size
      unmasked_loss_p_ = tf.reduce_sum(loss_p_vec_) / config.batch_size
      return loss_p_, unmasked_loss_p_

    def build_sampled_softmax(outputs_flat_):
      wp_t_ = tf.transpose(wp_)  # [state_size, hid_dim]
      targets_flat_ = tf.reshape(self.targets_, [-1, 1])  # [batch*t, 1]

      if config.candidate_sampler == 'default':
        sampled_values_ = None  # set None to use default sampler by tensorflow (uniform actually)
      else:
        targets_flat_ = tf.cast(targets_flat_, tf.int64)  # if use sampler, targets should be tf.int64
        if config.candidate_sampler == 'uniform':
          sampled_values_ = tf.nn.uniform_candidate_sampler(targets_flat_, 1,
                                                            config.sample_count_for_sampled_softmax,
                                                            True, config.state_size)
        elif config.candidate_sampler == 'log_uniform':
          sampled_values_ = tf.nn.log_uniform_candidate_sampler(targets_flat_, 1,
                                                                config.sample_count_for_sampled_softmax,
                                                                True, config.state_size)
        elif config.candidate_sampler == 'learned_unigram':
          sampled_values_ = tf.nn.learned_unigram_candidate_sampler(targets_flat_, 1,
                                                                    config.sample_count_for_sampled_softmax,
                                                                    True, config.state_size)
        elif config.candidate_sample == 'fixed_unigram':
          sampled_values_ = tf.nn.fixed_unigram_candidate_sampler(targets_flat_, 1,
                                                                  config.sample_count_for_sampled_softmax,
                                                                  True, config.state_size)
        else:
          raise Exception("`config.candidate_sample` should be correctly defined.")

      if config.sampled_softmax_alg == 'sampled_softmax':
        loss_p_vec_ = tf.nn.sampled_softmax_loss(wp_t_, bp_, outputs_flat_, targets_flat_,
                                                 config.sample_count_for_sampled_softmax, config.state_size,
                                                 sampled_values=sampled_values_)
      elif config.sampled_softmax_alg == 'nce':
        loss_p_vec_ = tf.nn.nce_loss(wp_t_, bp_, outputs_flat_, targets_flat_,
                                     config.sample_count_for_sampled_softmax,
                                     config.state_size, sampled_values=sampled_values_)
      else:
        raise Exception("`config.sampled_softmax_alg` should be correctly defined.")
      masked_loss_p_ = tf.mul(loss_p_vec_, tf.reshape(self.mask_, [-1]))  # [batch*t]
      loss_p_ = tf.reduce_sum(masked_loss_p_) / config.batch_size
      unmasked_loss_p_ = tf.reduce_sum(loss_p_vec_) / config.batch_size
      loss_ = loss_p_
      return {"loss": loss_, "loss_p": loss_p_, "unmasked_loss_p": unmasked_loss_p_}

    # loss p
    if use_constrained_softmax:
      if constrained_softmax_strategy == 'sparse_tensor':
        loss_p_, unmasked_loss_p_ = build_loss_p_constrained_use_sparse_tensor(outputs_flat_)
      elif constrained_softmax_strategy == 'adjmat_dense':
        loss_p_, unmasked_loss_p_ = build_loss_p_constrained_use_adjmat_dense(outputs_flat_)
      elif constrained_softmax_strategy == 'adjmat_adjmask':
        loss_p_, unmasked_loss_p_ = build_loss_p_constrained_use_adjmat_adjmask(outputs_flat_)
      else:
        raise Exception('`config.constrained_softmax_strategy` should be correctly defined')
      if build_unconstrained:
        loss_p_std_, unmasked_loss_p_std_ = build_loss_p_standard(outputs_flat_)
    else:
      loss_p_, unmasked_loss_p_ = build_loss_p_standard(outputs_flat_)

    # loss d & total loss
    if build_multitask:
      loss_d_, unmasked_loss_d_, loss_d_given_s_vec_ = build_loss_d_standard(outputs_flat_)
      loss_ = loss_p_ + loss_d_
      # compute P(d|s)
      loss_cond_ = tf.sub(loss_p_, tf.reduce_sum(loss_d_given_s_vec_) / config.batch_size)
    else:
      loss_ = loss_p_

    # construct loss_dict
    self.loss_dict["loss"] = loss_
    self.loss_dict["loss_p"] = loss_p_
    self.loss_dict["unmasked_loss_p"] = unmasked_loss_p_
    if build_multitask:
      self.loss_dict["loss_d"] = loss_d_
      self.loss_dict["unmasked_loss_d"] = unmasked_loss_d_
      self.loss_dict["loss_cond"] = loss_cond_
    if use_constrained_softmax and build_unconstrained:
      self.loss_dict["loss_p_std"] = loss_p_std_
      self.loss_dict["unmasked_loss_p_std"] = unmasked_loss_p_std_
    return

  def build_rnn(self, inputs_, train_phase):
    config = self.config

    def unrolled_rnn(cell, emb_inputs_, initial_state_, seq_len_):
      if not config.fix_seq_len:
        raise Exception("`config.fix_seq_len` should be set to `True` if using unrolled_rnn()")
      outputs = []
      state = initial_state_
      with tf.variable_scope("unrolled_rnn"):
        for t in range(config.max_seq_len):
          if t > 0:
            tf.get_variable_scope().reuse_variables()
          output, state = cell(emb_inputs_[:, t], state)  # [batch, hid_dim]
          outputs.append(output)
        rnn_outputs_ = tf.pack(outputs, axis=1)  # [batch, t, hid_dim]
      return rnn_outputs_
    def dynamic_rnn(cell, emb_inputs_, initial_state_, seq_len_):
      rnn_outputs_, last_states_ = tf.nn.dynamic_rnn(cell, emb_inputs_, initial_state=initial_state_,
                                                     sequence_length=seq_len_,
                                                     dtype=config.float_type)  # you should define dtype if initial_state is not provided
      return rnn_outputs_
    def bidirectional_rnn(cell, emb_inputs_, initial_state_, seq_len_):
      rnn_outputs_, output_states = tf.nn.bidirectional_dynamic_rnn(cell, cell, emb_inputs_, seq_len_,
                                                                    initial_state_, initial_state_, config.float_type)
      return tf.concat(2, rnn_outputs_)
    def rnn(cell, emb_inputs_, initial_state_, seq_len_):
      if not config.fix_seq_len:
        raise Exception("`config.fix_seq_len` should be set to `True` if using rnn()")
      inputs_ = tf.unpack(emb_inputs_, axis=1)
      outputs_, states_ = tf.nn.rnn(cell, inputs_, initial_state_, dtype=config.float_type, sequence_length=seq_len_)
      return outputs_

    if config.rnn == 'rnn':
      cell = tf.nn.rnn_cell.BasicRNNCell(config.hidden_dim)
    elif config.rnn == 'lstm':
      cell = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_dim)
    elif config.rnn == 'gru':
      cell = tf.nn.rnn_cell.GRUCell(config.hidden_dim)
    else:
      raise Exception("`config.rnn` should be correctly defined.")

    if train_phase and config.keep_prob < 1:
      cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
    if config.num_layers is not None and config.num_layers > 1:
      cell = tf.nn.rnn_cell.MultiRNNCell([cell] * config.num_layers)

    initial_state_ = cell.zero_state(config.batch_size, dtype=config.float_type)
    if config.use_seq_len_in_rnn:
      seq_len_ = self.seq_len_
    else:
      seq_len_ = None
    rnn_outputs_ = dynamic_rnn(cell, inputs_, initial_state_, seq_len_)  # [batch, time, hid_dim]
    return rnn_outputs_

  def get_batch(self, data, batch_size, max_len, append_EOS=False):
    """
    Get one mini-batch in data.
    :param data: list of lists, one line for one sample (do not add `EOS` in `data`, this func will do it instead)
    :param batch_size: int
    :param max_len: int, samples (+EOS) longer than `max_len` will be skipped.
    :param append_EOS: bool, if `True`, `EOS` will be automatically appended for each sample
                       if `False`, `sample[0:-1]` will be used as input and `sample[1:]` will be used as target.
                       Note that if `False`, samples with length 1 will also be skipped.
    :return: a Batch() object, containing:
             batch_in, batch_target, batch_mask: (batch_size, max_seq_len)
             batch_dest, batch_seq_lens: (batch_size)
             batch_adj_indices: (#indices, 2)
             where #indices is #adjacent_states of all input states of all samples (include padding state)
    """
    config = self.config
    batch_in, batch_target, batch_dest, batch_seq_lens, batch_adj_indices = [], [], [], [], []
    for _ in range(batch_size):
      while True:  # generate sample whose length is smaller than `max_seq_len`
        sample = random.choice(data)
        if len(sample) < max_len and (append_EOS or len(sample) > 1):
          break
      if append_EOS:
        batch_in.append(list(sample))
        batch_target.append(sample[1:] + [config.EOS_ID])
        batch_dest.append(sample[-1])
        batch_seq_lens.append(len(sample))
      else:
        batch_in.append(sample[:-1])
        batch_target.append(sample[1:])
        batch_dest.append(sample[-1])
        batch_seq_lens.append(len(sample) - 1)

    if config.fix_seq_len:
      max_seq_len = config.max_seq_len
    else:
      max_seq_len = max(batch_seq_lens)

    # padding and get adj indices for sparse tensor used in constrained_softmax
    sample_id = 0
    for sample_in, sample_target in zip(batch_in, batch_target):
      sample_in.extend((max_seq_len - len(sample_in)) * [config.PAD_ID])
      sample_target.extend((max_seq_len - len(sample_target)) * [config.TARGET_PAD_ID])
      # adj indices
      if config.constrained_softmax_strategy == 'sparse_tensor':
        for edge_id in sample_in:
          adjList_ids = self.map.edges[edge_id].adjList_ids
          for adj_id in adjList_ids:
            batch_adj_indices.append([sample_id, adj_id])
          sample_id += 1

    # masking
    batch_mask = np.zeros((batch_size, max_seq_len), dtype=float)
    for row in range(batch_size):
      for col in range(batch_seq_lens[row]):
        batch_mask[row][col] = 1.0

    # sub_one_hot_target
    batch_sub_onehot_target = np.zeros([config.batch_size, max_seq_len, self.adj_mask.shape[1]], dtype=np.float32)  # TODO, implement automatic inference, maybe fixed
    for x0 in range(batch_sub_onehot_target.shape[0]):
      for x1 in range(batch_sub_onehot_target.shape[1]):
        for x2 in range(batch_sub_onehot_target.shape[2]):
          if batch_target[x0][x1] == self.adj_mat[batch_in[x0][x1]][x2]:
            batch_sub_onehot_target[x0][x1][x2] = 1.0
            break

    return Batch(inputs=np.array(batch_in), targets=np.array(batch_target), masks=batch_mask,
           dests=np.array(batch_dest), seq_lens=np.array(batch_seq_lens), adj_indices=np.array(batch_adj_indices),
           sub_onehot_target=batch_sub_onehot_target)

  def get_batch_for_test(self, data, batch_size, max_len, from_id, append_EOS=False):
    """
    Get one mini-batch in data.
    The batch is returned in the order of `data` to ensure each sample is evaluated.

    :param data: list of lists, one line for one sample (do not add `EOS` in `data`, this func will do it instead)
    :param batch_size: int
    :param max_len: int, samples longer than `max_len` will be skipped.
    :param from_id: int, get the batch start exactly from `data[from_id]`
    :param append_EOS: bool, if `True`, `EOS` will be automatically appended for each sample
                       if `False`, `sample[0:-1]` will be used as input and `sample[1:]` will be used as target
                       Note that if `False`, samples with length 1 will also be skipped.
    :return: a Batch() object, containing:
             batch_in, batch_target, batch_mask: (batch_size, max_seq_len)
             batch_dest, batch_seq_lens: (batch_size)
             batch_adj_indices: (#indices, 2)
             where #indices is #adjacent_states of all input states of all samples (include padding state)

             and a pointer: the next position to fetch the batch (useful for iteration)
    """
    if len(data) - from_id < batch_size:
      return None, -1

    config = self.config
    batch_in, batch_target, batch_dest, batch_seq_lens, batch_adj_indices = [], [], [], [], []
    pointer = from_id
    indices_counter = 0
    for _ in range(batch_size):
      while pointer < len(data):  # generate sample whose length is smaller than `max_seq_len`
        sample = data[pointer]
        if len(sample) < max_len and (append_EOS or len(sample) > 1):
          break
        else:
          pointer += 1
      if pointer >= len(data):
        return None, -1

      if append_EOS:
        batch_in.append(list(sample))
        batch_target.append(sample[1:] + [config.EOS_ID])
        batch_dest.append(sample[-1])
        batch_seq_lens.append(len(sample))
      else:
        batch_in.append(sample[:-1])
        batch_target.append(sample[1:])
        batch_dest.append(sample[-1])
        batch_seq_lens.append(len(sample) - 1)

      pointer += 1
    if config.fix_seq_len:
      max_seq_len = config.max_seq_len
    else:
      max_seq_len = max(batch_seq_lens)

    # padding and get adj indices for sparse tensor used in constrained_softmax
    sample_id = 0
    for sample_in, sample_target in zip(batch_in, batch_target):
      sample_in.extend((max_seq_len - len(sample_in)) * [config.PAD_ID])
      sample_target.extend((max_seq_len - len(sample_target)) * [config.TARGET_PAD_ID])
      # adj indices
      if config.constrained_softmax_strategy == 'sparse_tensor':
        for edge_id in sample_in:
          adjList_ids = self.map.edges[edge_id].adjList_ids
          for adj_id in adjList_ids:
            batch_adj_indices.append([sample_id, adj_id])
          sample_id += 1

    # masking
    batch_mask = np.zeros((batch_size, max_seq_len), dtype=float)
    for row in range(batch_size):
      for col in range(batch_seq_lens[row]):
        batch_mask[row][col] = 1.0

    # sub_one_hot_target
    batch_sub_onehot_target = np.zeros([config.batch_size, max_seq_len, self.adj_mat.shape[1]],
                                       dtype=np.float32)  # TODO, implement automatic inference, maybe done
    for x0 in range(batch_sub_onehot_target.shape[0]):
      for x1 in range(batch_sub_onehot_target.shape[1]):
        for x2 in range(batch_sub_onehot_target.shape[2]):
          if batch_target[x0][x1] == self.adj_mat[batch_in[x0][x1]][x2]:
            batch_sub_onehot_target[x0][x1][x2] = 1.0
            break
    return Batch(inputs=np.array(batch_in), targets=np.array(batch_target), masks=batch_mask,
                 dests=np.array(batch_dest), seq_lens=np.array(batch_seq_lens), adj_indices=batch_adj_indices,
                 sub_onehot_target=batch_sub_onehot_target), pointer

  def fetch(self, eval_op):
    fetch_dict = {}
    for k in self.loss_dict.keys():
      fetch_dict[k] = self.loss_dict[k]
    if eval_op is not None:
      fetch_dict["_eval_op"] = eval_op
    for k in self.debug_tensors.keys():
      # use '_' in the front of the name to identify the tensors not to be included in the `print` time
      fetch_dict["_"+k] = self.debug_tensors[k]
    for k in self.trace_dict.keys():
      # use '~' in the front of the name to identify the tensors not to be included in the `print` time
      # and to identify these tensors need to be traced.
      fetch_dict["~"+k] = self.trace_dict[k]
    return fetch_dict
  def fetch_encoder(self):
    fetch_dict = {}
    fetch_dict["encoder_outputs"] = self.encoder_outputs_
    return fetch_dict

  def feed(self, batch):
    """
    feed one batch to placeholders
    :param batch: a Batch object
    :return: feed dict of inputs
    """
    input_feed = {}
    input_feed[self.inputs_.name] = batch.inputs
    input_feed[self.targets_.name] = batch.targets
    input_feed[self.mask_.name] = batch.masks
    input_feed[self.dests_label_.name] = batch.dests
    input_feed[self.seq_len_.name] = batch.seq_lens
    if self.logits_mask__ is not None:
      values = np.ones(len(batch.adj_indices), np.float32)
      shape = np.array([np.size(batch.inputs), self.config.state_size], dtype=np.int32)
      input_feed[self.logits_mask__] = tf.SparseTensorValue(batch.adj_indices, values, shape)
    input_feed[self.lr_] = self.config.lr
    if self.sub_onehot_targets_ is not None:
      input_feed[self.sub_onehot_targets_] = batch.sub_onehot_target
    return input_feed
  def feed_decoder(self, overall_feed_dict, encoder_outputs):
    overall_feed_dict[self.decoder_inputs_.name] = encoder_outputs
    return overall_feed_dict

  def step(self, sess, batch, eval_op=None):
    if self.config.predict_dir and self.config.encoder_decoder == 'decoder':
      vals = self.step_pretrained_encoder_and_decoder(sess, batch, eval_op)
    else:
      feed_dict = self.feed(batch)
      fetch_dict = self.fetch(eval_op)
      # run sess
      vals = sess.run(fetch_dict, feed_dict, options=self.config.run_options, run_metadata=self.config.run_metadata)
      # trace time consumption
      # very slow and requires large memory
      if self.config.time_trace:
        tl = timeline.Timeline(self.config.run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open(self.config.trace_filename, 'w') as f:
          f.write(ctf)
          print("time tracing output to " + self.config.trace_filename)
    return vals

  def speed_benchmark(self, sess, samples_for_test=500):
    data = self.data
    steps_for_test = samples_for_test // self.config.batch_size + 1
    loss = 0.0
    t1 = time.time()
    for _ in range(steps_for_test):
      batch = self.get_batch(data, self.config.batch_size, self.config.max_seq_len)
      if self.train_phase:
        eval_op = self.update_op
      else:
        eval_op = None
      loss_dict = self.step(sess, batch, eval_op)  # average by batch
      loss += loss_dict[self.config.loss_for_filename]

      def plot_dir():
        map = self.map
        e_dir = loss_dict["_e_dir_distri"]
        t_dir = loss_dict["_t_dir_distri"]
        targets = batch.targets.flatten()
        buckets = np.linspace(0.0, 360.0, self.config.dir_granularity + 1)[0:-1]
        for i in range(e_dir.shape[0]): # batch*t
          def get_adj_distri(edge_id):
            edge = map.edges[edge_id]
            print("pick up edge " + str(edge.id))
            def calc_deg(p1, p2):
              cos_val = (p2[0] - p1[0]) / math.sqrt(
                (p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1]))
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
              for bucket_deg in np.linspace(0.0, 360.0, self.config.dir_granularity + 1)[0:-1]:
                if abs(bucket_deg - deg) < 360.0 - abs(bucket_deg - deg):
                  delta = abs(bucket_deg - deg)
                else:
                  delta = 360.0 - abs(bucket_deg - deg)
                probs.append(math.exp(-delta * delta / (2 * self.config.dir_sigma * self.config.dir_sigma)))
              # normalization
              summation = sum(probs)
              for i in range(len(probs)):
                probs[i] /= summation
              adj_dists.append(probs)
            return adj_dists
          adj_distri = get_adj_distri(targets[i])
          for distri in adj_distri:
            plt.plot(buckets, distri, 'g-')
          plt.plot(buckets, e_dir[i], 'b-')
          plt.plot(buckets, t_dir[i], 'r-')
          plt.title(str(targets[i]))
          plt.show()
      # plot_dir()

      if len(self.debug_tensors) > 0:
        print("[Debug mode]")

        t = loss_dict["_log_probs"].shape[0]
        for i in range(t):
          if (loss_dict["_seq_mask"][i] == 0.0):
            continue
          print(loss_dict["_sub_adj_mask"][i])
          print(loss_dict["_log_probs"][i])
          print(loss_dict["_probs"][i])
          print(loss_dict["_sub_onehot_targets_flat"][i])
          print(loss_dict["_max_prediction"][i])
          print(loss_dict["_seq_mask"][i])
          raw_input()

        for k in self.debug_tensors.keys():
          debug_val = loss_dict["_"+k]
          print(k +".shape = " + str(debug_val.shape))
          print(debug_val)
          input()

    t2 = time.time()
    samples_per_sec = steps_for_test * self.config.batch_size / float(t2 - t1)
    ms_per_sample = float(t2 - t1) * 1000.0 / (steps_for_test * self.config.batch_size)
    print("%d samples per sec, %.4f ms per sample, batch_size = %d" % (
    samples_per_sec, ms_per_sample, self.config.batch_size))
    print("benchmark loss = %.5f" % (loss / steps_for_test))

  def train_epoch(self, sess, data):
    config = self.config
    cumulative_losses = {}
    general_step_count = 0
    batch_counter = 0

    steps_per_epoch_in_train = self.config.samples_per_epoch_in_train // self.config.batch_size
    for _ in range(steps_per_epoch_in_train):
      batch_counter += 1
      general_step_count += 1
      batch = self.get_batch(data, self.config.batch_size, self.config.max_seq_len)
      fetch_vals = self.step(sess, batch, eval_op=self.update_op)  # average by batch

      for k in fetch_vals.keys():
        if k[0] == '_' or k[0] == '~': # keys having "_" or "~" in the front is the tensors we do not want to output here
          continue
        if cumulative_losses.get(k) is None:
          cumulative_losses[k] = fetch_vals[k]
        else:
          cumulative_losses[k] += fetch_vals[k]

      # collect trace information
      for k in self.trace_dict.keys():
        if self.trace_items.get(k) is None:
          # self.trace_items[k] = {}
          self.trace_items[k] = [[], []] # label, value
        trace_val = fetch_vals['~' + k]
        inputs = batch.inputs.flatten()
        targets = batch.targets.flatten()
        if trace_val.shape[0] != inputs.shape[0]:
          raise Exception("The first dimension of `trace_val` and `inputs` should match (batch*t), %d vs %d" %
                          (trace_val.shape[0], inputs.shape[0]))
        for input, target, hid_layer_val in zip(inputs, targets, trace_val):
          if input == config.trace_input_id:
            self.trace_items[k][0].append(target)
            self.trace_items[k][1].append(hid_layer_val)
            """
            if self.trace_items[k].get(target) is None:
              self.trace_items[k][target] = []
            self.trace_items[k][target].append(hid_layer_val)
            """

      # reach every 10%
      # print losses and reinitialize counters and losses
      if general_step_count % (steps_per_epoch_in_train // 10) == 0:
        print("\t%d%%:" % (general_step_count // (steps_per_epoch_in_train // 10) * 10), end='')
        if self.config.compute_ppl:
          print("| ppl = %.3f" % math.exp(float(cumulative_losses[self.config.loss_for_filename] / batch_counter)),end='')
        for (k,v) in cumulative_losses.items():
          print("| %s = %.4f" % (k, v / batch_counter), end='')
          cumulative_losses[k] = 0.0
        batch_counter = 0
        print("")

        if self.config.trace_hid_layer:
          self.dump_trace_item()

        # flush IO buffer
        if self.config.direct_stdout_to_file:
          self.config.log_file.close()
          self.config.log_file = open(self.config.log_filename, "a+")
          sys.stdout = self.config.log_file
    print("")

  def dump_trace_item(self):
    config = self.config
    for k in self.trace_items.keys():
      np.savetxt(os.path.join(config.save_path, 'label_' + k), self.trace_items[k][0], '%d')
      np.savetxt(os.path.join(config.save_path, 'feature_' + k), self.trace_items[k][1])
      print("trace items have been output to " + config.save_path)

  def eval(self, sess, data, is_valid_set, save_ckpt, model_train = None):
    cumulative_losses = {}
    batch_counter = 0
    from_id = 0
    while True:
      batch, from_id = self.get_batch_for_test(data, self.config.batch_size, self.config.max_seq_len, from_id)
      if batch is None:
        break
      fetch_vals = self.step(sess, batch)  # average by batch
      for k in fetch_vals.keys():
        if k[0] == '_' or k[0] == '~':
          continue
        if cumulative_losses.get(k) is None:
          cumulative_losses[k] = fetch_vals[k]
        else:
          cumulative_losses[k] += fetch_vals[k]
      batch_counter += 1

    # print valid loss
    if is_valid_set:
      name = 'valid'
    else:
      name = 'test'
    print(name +" set:", end='')
    if self.config.compute_ppl:
      print("| ppl = %.3f" % math.exp(float(cumulative_losses[self.config.loss_for_filename] / batch_counter)),
            end='')
    for (k, v) in cumulative_losses.items():
      print("| %s = %.4f" % (k, v / batch_counter), end='')
    print("")

    if save_ckpt and self.config.save_ckpt:
      # save ckpt
      filename = name + "_%s_%.3f" % (self.config.loss_for_filename,
                                    cumulative_losses[self.config.loss_for_filename] / batch_counter)
      if self.config.compute_ppl:
        ppl_str = "_ppl_%.3f" % math.exp(float(cumulative_losses[self.config.loss_for_filename] / batch_counter))
        filename += ppl_str
      if not os.path.exists(self.config.save_path):
        os.makedirs(self.config.save_path)

      # save encoder forward part only (for decoder inputs)
      if self.config.predict_dir and self.config.encoder_decoder == 'encoder':
        forward_ckpt_path = os.path.join(self.config.encoder_save_path_forward_part, filename)
        if not os.path.exists(self.config.encoder_save_path_forward_part):
          os.makedirs(self.config.encoder_save_path_forward_part)
        print("saving forward params to: " + forward_ckpt_path)
        model_train.encoder_forward_saver.save(sess, forward_ckpt_path)  # params should be saved by model_train

      # save training ckpt file
      ckpt_path = os.path.join(self.config.save_path, filename)
      print("saving training ckpt to: " + ckpt_path)
      model_train.saver.save(sess,
                             ckpt_path)  # params should be saved by model_train as model_test do not have params of RMSProp
      print("done")

      # flush IO buffer
      if self.config.direct_stdout_to_file:
        self.config.log_file.close()
        self.config.log_file = open(self.config.log_filename, "a+")
        sys.stdout = self.config.log_file
    print("")
