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
    self.sub_onehot_target = sub_onehot_target # (batch, time, max_adj_num)


class TrajModel(object):
  def __init__(self, train_phase, config, data, model_scope = None, map = None, mapInfo = None):
    """
    :param train_phase: bool, indicate whether the instance of the model is in train phase or valid/test phase
    :param config: Config
    :param data: ndarray
    :param model_scope: string
    :param map: Map
    :param mapInfo: MapInfo
    """
    self.debug_tensors = {} # add any tensor you want to evaluate for debugging and just run `speed_benchmark()`
    self.model_scope = model_scope
    self.loss_dict = {}
    self.data = data
    self.train_phase = train_phase
    self.config = config
    self.map = map
    self.mapInfo = mapInfo
    self.lr_ = tf.placeholder(config.float_type, name="lr") # learning rate
    self.adj_mat = mapInfo.adj_mat
    self.adj_mask = mapInfo.adj_mask
    self.dest_coord = mapInfo.dest_coord
    self.logits_mask__ = None
    self.sub_onehot_targets_ = None
    self.trace_dict = {}
    self.trace_items = {} # k = target id, v = list of layer output
    self.loss_logs = {}

    # construct tensors
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

    # build whole computation graph of the model
    if config.model_type == "CSSRNN" or config.model_type == "SPECIFIED_CSSRNN":
      self.adj_mat_ = tf.constant(self.adj_mat, config.int_type, name="adj_mat")
      self.adj_mask_ = tf.constant(self.adj_mask, config.float_type, name="adj_mask")
      self.build_RNNLM_model(train_phase)
    elif config.model_type == "LPIRNN":
      self.adj_mat_ = tf.constant(self.adj_mat, config.int_type, name="adj_mat")
      self.adj_mask_ = tf.constant(self.adj_mask, config.float_type, name="adj_mask")
      self.build_LPIRNN_model(train_phase)
    else: # build traditional RNN language model
      self.build_RNNLM_model(train_phase)

  def build_input_layer(self, train_phase, input_label, use_dest=False, dest_label_=None, use_emb_for_dest=True, var_scope="input"):
    """
    Build input tensor of the model (transform to distributed representation)
    :param train_phase: bool
    :param input_label: [batch, t] int
    :param use_dest: bool, whether to use destination information
    :param dest_label_: [batch] int
    :param use_emb_for_dest: bool, whether to use distributed representation for destination,
                             if False, the coordinate (2 dimension) of the dest will be appended to input
    :param var_scope: string
    :return: inputs_: [batch, t, emb], emb (use_dest=True,use_emb_for_dest=True) will be twice of emb (use_dest=False)

    Note that we can set `config.pretrained_input_emb_path` to load the pretrained embeddings of input states (not support for dest states).
    """
    config = self.config
    with tf.variable_scope(var_scope):
      # construct embeddings
      if config.pretrained_input_emb_path != '': # load pretrained embeddings (such as word2vec)
        pretrained_emb = np.loadtxt(config.pretrained_input_emb_path, delimiter=',')
        pretrained_emb_ = tf.constant(pretrained_emb, config.float_type)
        emb_ = tf.get_variable("embedding", dtype=config.float_type, initializer=pretrained_emb_)
        print("init emb by pretraining.")
      else:
        emb_ = tf.get_variable("embedding", [config.state_size, config.emb_dim], dtype=config.float_type)
      emb_inputs_ = tf.nn.embedding_lookup(emb_, input_label, name="emb_inputs")  # batch_size x time_steps x emb_dim
      if train_phase and config.keep_prob < 1:
        emb_inputs_ = tf.nn.dropout(emb_inputs_, keep_prob=config.keep_prob, name="dropout_emb_inputs")

      # with destination input information
      if use_dest:
        if use_emb_for_dest:
          self.dest_emb_ = tf.get_variable("dest_emb", [config.state_size, config.emb_dim], dtype=config.float_type)
        else:
          self.dest_emb_ = self.dest_coord_
        dest_inputs_ = tf.tile(tf.expand_dims(tf.nn.embedding_lookup(self.dest_emb_, dest_label_), 1), [1, self.max_t_, 1])  # [batch, t, dest_emb]
        inputs_ = tf.concat([emb_inputs_, dest_inputs_], 2, "input_with_dest")
        #inputs_ = tf.concat(2, [emb_inputs_, dest_inputs_], "input_with_dest")
      else:
        inputs_ = emb_inputs_
      return inputs_

  def build_rnn_layer(self, inputs_, train_phase):
    """
    Build the computation graph from inputs to outputs of the RNN layer.
    :param inputs_: [batch, t, emb], float
    :param train_phase: bool
    :return: rnn_outputs_: [batch, t, hid_dim], float
    """
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

  def build_xent_loss_layer(self, outputs_flat_, use_constrained_softmax=False,
                            constrained_softmax_strategy='adjmat_adjmask', build_unconstrained=False):
    """
    Build the computation process from the output of rnn layer to the x-ent loss layer.
    :param outputs_flat_: [batch*t, hid_dim], float. The output of RNN layer of all time steps
    :param use_constrained_softmax: bool. Set True to use CSSRNN or False to use traditional RNN (RNN-based LM)
    :param constrained_softmax_strategy: string, 'adjmat_adjmask' = the method proposed in the paper.
                                         'sparse_tensor' = an alternative, but slower than 'adjmat_adjmask'
    :param build_unconstrained: bool, Set True to see the loss if we do not use the mask on the final layer when testing.
    :return: This function will finally build some tensors you may care and add them to `self.loss_dict`
    """
    config = self.config
    # projection to output space
    wp_ = tf.get_variable("wp", [int(outputs_flat_.get_shape()[1]), config.state_size],
                          dtype=config.float_type)  # [hid_dim, state_size]
    bp_ = tf.get_variable("bp", [config.state_size], dtype=config.float_type)  # [state_size]

    def build_loss_p_standard(outputs_flat_):
      """
      The traditional cross-ent loss.
      :param outputs_flat_: [batch*t, hid_dim], float
      :return: loss_p_: 1-D tensor, the x-ent loss averaged by batch_size
      """
      targets_p_flat_ = tf.reshape(self.targets_, [-1])  # [batch*t]

      # hidden to output
      logits_p_ = tf.matmul(outputs_flat_, wp_) + bp_  # [batch*t, state_size]
      loss_p_vec_ = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_p_, labels=targets_p_flat_)  # [batch*t]

      masked_loss_p_ = tf.multiply(loss_p_vec_, tf.reshape(self.mask_, [-1]))  # [batch*t]
      loss_p_ = tf.reduce_sum(masked_loss_p_) / config.batch_size

      max_prediction_label_ = tf.cast(tf.argmax(logits_p_, dimension=1), config.int_type) # [batch*t], int64->int32

      self.build_max_predict_loss_layer(max_prediction_label_, targets_p_flat_, False)

      return loss_p_

    def build_loss_p_constrained_use_sparse_tensor(outputs_flat_):
      """
      Build the constrained cross-entropy loss for `outputs_flat_`.

      The strategy is to use a sparse tensor `self.logits_mask__` to directly represent the logits mask.
      Instead of using tf.nn.sparse_softmax_cross_entropy_with_logits(),
      it generates an one-hot target tensor with shape [batch*t, state_size] and computes the cross-entropy
      by sparse tensor ops.

      :param outputs_flat_: [batch*t, hid_dim], flattened outputs of rnn
      :return: loss_p_: 1-D tensor, the x-ent loss averaged by batch_size
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
      masked_loss_p_ = tf.multiply(loss_p_vec_, tf.reshape(self.mask_, [-1]))  # [batch*t]
      loss_p_ = tf.reduce_sum(masked_loss_p_) / config.batch_size

      return loss_p_

    def build_loss_p_constrained_use_adjmat_adjmask(outputs_flat_):
      """
      Build the constrained cross-entropy loss for `outputs_flat_`.

      This strategy is the one introduced in the paper. Please refer the paper for more details.
      :param outputs_flat_: [batch*t, hid_dim], flattened outputs of rnn
      :return: loss_p_: 1-D tensor, the x-ent loss averaged by batch_size
      """

      def constrained_softmax_cross_entropy_loss(outputs_, input_, target_, w_t_, b_, adj_mat_, adj_mask_):
        """
        constrained_softmax, fastest version!
        :param outputs_: [batch*t, hid_dim], float
        :param input_: [batch, t], int
        :param target_: [batch, t], int
        :param w_t_: [state_size, hid_dim], float
        :param b_: [state_size], float
        :param adj_mat_: [state_size, max_adj_num], int, the 'legal transition matrix' in the paper,
                         adj_mat_[i][j] is the j-th adjacent state of i (include padding)
        :param adj_mask_: [state_size, max_adj_num], float, the 'legal transition mask' in the paper,
                          adj_mask_[i][j] represents whether adj_mat_[i][j] is an adjacent state ( = 1.0) of i or padding ( = 0.0)

        :return: the loss with shape, [batch*t], float
                 the prediction of next state w.r.t. each state by returning the one having maximum prob, [batch*t, max_adj_num], one-hot
        """
        input_flat_ = tf.reshape(input_, [-1])  # [batch*t]
        target_flat_ = tf.reshape(target_, [-1, 1])  # [batch*t, 1]
        sub_adj_mat_ = tf.nn.embedding_lookup(adj_mat_, input_flat_)  # [batch*t, max_adj_num]
        sub_adj_mask_ = tf.nn.embedding_lookup(adj_mask_, input_flat_)  # [batch*t, max_adj_num]

        # first column is target_
        target_and_sub_adj_mat_ = tf.concat([target_flat_, sub_adj_mat_], 1)  # [batch*t, max_adj_num+1]

        outputs_3d_ = tf.expand_dims(outputs_, 1)  # [batch*t, hid_dim] -> [batch*t, 1, hid_dim]

        sub_w_ = tf.nn.embedding_lookup(w_t_, target_and_sub_adj_mat_)  # [batch*t, max_adj_num+1, hid_dim]
        sub_b_ = tf.nn.embedding_lookup(b_,
                                        target_and_sub_adj_mat_)  # [batch*t, max_adj_num+1] #TODO: I find that I forgot to add the bias :(
        sub_w_flat_ = tf.reshape(sub_w_, [-1, int(sub_w_.get_shape()[2])])  # [batch*t*max_adj_num+1, hid_dim]
        outputs_tiled_ = tf.tile(outputs_3d_, [1, tf.shape(adj_mat_)[1] + 1, 1])  # [batch*t, max+adj_num+1, hid_dim]
        outputs_tiled_ = tf.reshape(outputs_tiled_,
                                    [-1, int(outputs_tiled_.get_shape()[2])])  # [batch*t*max_adj_num+1, hid_dim]
        target_logit_and_sub_logits_ = tf.reshape(tf.reduce_sum(tf.multiply(sub_w_flat_, outputs_tiled_), 1),
                                                  [-1, tf.shape(adj_mat_)[1] + 1])  # [batch*t, max_adj_num+1]

        # for numerical stability
        scales_ = tf.reduce_max(target_logit_and_sub_logits_, 1)  # [batch*t]
        scaled_target_logit_and_sub_logits_ = tf.transpose(
          tf.subtract(tf.transpose(target_logit_and_sub_logits_),
                 scales_))  # transpose for broadcasting [batch*t, max_adj_num+1]

        scaled_sub_logits_ = scaled_target_logit_and_sub_logits_[:, 1:]  # [batch*t, max_adj_num]
        exp_scaled_sub_logits_ = tf.exp(scaled_sub_logits_)  # [batch*t, max_adj_num]
        deno_ = tf.reduce_sum(tf.multiply(exp_scaled_sub_logits_, sub_adj_mask_), 1)  # [batch*t]
        log_deno_ = tf.log(deno_)  # [batch*t]
        log_nume_ = tf.reshape(scaled_target_logit_and_sub_logits_[:, 0:1], [-1])  # [batch*t]
        loss_ = tf.subtract(log_deno_, log_nume_)  # [batch*t] since loss is -sum(log(softmax))

        max_prediction_ = tf.one_hot(tf.argmax(exp_scaled_sub_logits_ * sub_adj_mask_, 1),
                                     int(adj_mat_.get_shape()[1]), dtype=config.float_type)  # [batch*t, max_adj_num]
        return loss_, max_prediction_

      wp_t_ = tf.transpose(wp_)  # [state_size, hid_dim]
      loss_p_vec_, max_prediction_ = constrained_softmax_cross_entropy_loss(outputs_flat_, self.inputs_,
                                                            self.targets_, wp_t_, bp_,
                                                            self.adj_mat_, self.adj_mask_)  # [batch*t]
      self.build_max_predict_loss_layer(max_prediction_,
                                        tf.reshape(self.sub_onehot_targets_,
                                             [-1, int(self.sub_onehot_targets_.get_shape()[2])]),
                                        use_onehot=True)
      masked_loss_p_ = tf.multiply(loss_p_vec_, tf.reshape(self.mask_, [-1]))  # [batch*t]
      loss_p_ = tf.reduce_sum(masked_loss_p_) / config.batch_size
      return loss_p_

    # loss p (i.e., the negative log likelihood loss / x-ent loss)
    if use_constrained_softmax: # CSSRNN
      if constrained_softmax_strategy == 'sparse_tensor':
        loss_p_ = build_loss_p_constrained_use_sparse_tensor(outputs_flat_)
      elif constrained_softmax_strategy == 'adjmat_adjmask': # this one is better
        loss_p_ = build_loss_p_constrained_use_adjmat_adjmask(outputs_flat_)
      else:
        raise Exception('`config.constrained_softmax_strategy` should be correctly defined')
      if build_unconstrained: # if you want to see the loss without the topological mask
        loss_no_topo_mask = build_loss_p_standard(outputs_flat_)
    else: # traditional RNN
      loss_p_ = build_loss_p_standard(outputs_flat_)

    loss_ = loss_p_

    # construct loss_dict
    # Note that here "loss" refers to the training loss (including L2-regularizer if it has, e.g., in LPIRNN model)
    # and "loss_p" refers to the x-ent loss of the sequence. They are not the same
    # (although they may refer to the same tensor like here)
    self.loss_dict["loss"] = loss_
    self.loss_dict["loss_p"] = loss_p_
    if use_constrained_softmax and build_unconstrained:
      self.loss_dict["loss_no_topo_mask"] = loss_no_topo_mask
    return

  def build_max_predict_loss_layer(self, max_prediction_flat_, targets_flat_, use_onehot):
    """
    Build the accuracy of predicting the next state by the one with maximum prob.
    "_flat_" means the first dimension should be `batch*t` rather than `batch`
    :param max_prediction_flat_: [batch*t, max_adj_num], one-hot, or [batch*t], int
    :param targets_flat_: [batch*t, max_adj_num], one-hot, or [batch*t], int
    :param use_onehot: bool, whether the representation is by one-hot or by label
    :return: float, the prediction accuracy
    """
    if not use_onehot:
      correct_count_ = tf.reduce_sum(tf.cast(tf.equal(max_prediction_flat_, targets_flat_), self.config.float_type) \
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

  def build_sharedTask_part(self, train_phase, var_scope="shared_task"):
    """
    Build input->RNN->FC_layer->latent prediction information.
    :param train_phase: bool
    :param var_scope: string
    :return: [batch*t, lpi_dim], the latent prediction information of all time steps
    """
    config = self.config
    with tf.variable_scope(var_scope):
      # construct embeddings
      emb_inputs_ = self.build_input_layer(train_phase, self.inputs_, config.input_dest, self.dests_label_, config.dest_emb)

      # construct rnn
      rnn_outputs_ = self.build_rnn_layer(emb_inputs_, train_phase)  # [batch, time, hid_dim]
      outputs_flat_ = tf.reshape(rnn_outputs_, [-1, int(rnn_outputs_.get_shape()[2])])  # [batch*t, hid_dim]

      # hidden to output
      w_fc_ = tf.get_variable("w_fc", [int(outputs_flat_.get_shape()[1]), config.lpi_dim],
                               dtype=config.float_type)  # [hid_dim, lpi_dim]
      b_fc_ = tf.get_variable("b_fc", [config.lpi_dim], dtype=config.float_type)  # [lpi_dim]
      lpi_ = tf.matmul(outputs_flat_, w_fc_) + b_fc_  # [batch*t, lpi_dim]
      return lpi_

  def build_individualTask_part(self, train_phase, lpi_, var_scope="individual_task"):
    """
    Build LPI->individual_task-> x-ent loss & max-prediction
    :param train_phase: bool
    :param lpi_: [batch*t, lpi_dim], the latent prediction information obtained from `build_sharedTask_layer()`
    :param var_scope: string
    :return: loss_: 1-D tensor, the training loss (including L2-regularizer if it has one)
             loss_p_: 1-D tensor, the x-ent averaged by `batch_size`
    """
    config = self.config
    with tf.variable_scope(var_scope):
      self.all_w_task_ = tf.get_variable("all_w_task", [config.state_size, self.adj_mat_.get_shape()[1],
                                                        lpi_.get_shape()[1]],
                                         dtype=config.float_type) # [state, max_adj_num, batch*t]
      self.all_b_task_ = tf.get_variable("all_b_task", [config.state_size, self.adj_mat_.get_shape()[1]]) # [state, max_adj_num]
      # dropout
      if train_phase and config.individual_task_keep_prob < 1:
        all_w_task_ = tf.nn.dropout(self.all_w_task_, keep_prob=config.individual_task_keep_prob, name="dropout_all_w_task")
      else:
        all_w_task_ = self.all_w_task_

      all_b_task_ = self.all_b_task_

      # compute loss
      def constrained_softmax_cross_entropy_loss_with_individual_weights(inputs_, lpi_,
                                                                         sub_onehot_targets_, seq_mask_,
                                                                         all_w_task_, all_b_task_, adj_mask_):
        """
        A fast strategy to compute the x-ent loss for LPIRNN model. The main idea is similar to the speed-up strategy
        in CSSRNN introduced in the paper. Thus we do not include this stuff in the paper. For more details, just read
        the code :)
        :param inputs_: [batch, t], int, just the original inputs of the trajectory
        :param lpi_: [batch*t, lpi_dim], float, the latent prediction information
        :param sub_onehot_targets_: [batch, t, max_adj_num], float, the one-hot target matrix constructed according to adjacent states
        :param seq_mask_: [batch, t], float, sequence mask, for masking different length of sequences in a batch
        :param all_w_task_: [state_size, max_adj_num, dir], float,
                            all_w_task_[i] is the weight matrix w.r.t. task i (i.e., current state is i)
        :param all_b_task_: [state_size, max_adj_num], float
        :param adj_mask_: [state_size, max_adj_num], ,float
               adj_mask_[i][j] represents whether adj_mat_[i][j] is an adjacent state ( = 1.0) of i or padding ( = 0.0)

        :return: xent_: the flattened x-ent loss with shape [batch*t]
                 max_prediction_: the flattened one-hot representation by max-prediction with shape [batch*t, max_adj_num]
        """

        inputs_flat_ = tf.reshape(inputs_, [-1])  # [batch*t]
        sub_adj_mask_ = tf.nn.embedding_lookup(adj_mask_, inputs_flat_)  # [batch*t, max_adj_num]

        encoders_3d_ = tf.expand_dims(lpi_, 1)  # [batch*t, dir] -> [batch*t, 1, dir]

        sub_w_ = tf.nn.embedding_lookup(all_w_task_, inputs_flat_)  # [batch*t, max_adj_num, dir]
        sub_b_ = tf.nn.embedding_lookup(all_b_task_, inputs_flat_)  # [batch*t, max_adj_num]

        encoders_tiled_ = tf.tile(encoders_3d_, [1, int(adj_mask_.get_shape()[1]), 1])  # [batch*t, max_adj_num, dir]
        logits_ = tf.reduce_sum(tf.multiply(sub_w_, encoders_tiled_), 2) + sub_b_  # [batch*t, max_adj_num]

        # for numerical stability
        scales_ = tf.reduce_max(logits_, 1)  # [batch*t]
        scaled_logits = tf.transpose(
          tf.subtract(tf.transpose(logits_), scales_))  # transpose for broadcasting [batch*t, max_adj_num]
        exp_scaled_logits_ = tf.exp(scaled_logits)  # [batch*t, max_adj_num]
        normalizations_ = tf.reduce_sum(tf.multiply(exp_scaled_logits_, sub_adj_mask_), 1)  # [batch*t]
        log_probs_ = tf.transpose(tf.transpose(scaled_logits) - tf.log(normalizations_))  # [batch*t, max_adj_num]

        sub_onehot_targets_flat = tf.reshape(sub_onehot_targets_,
                                             [-1, int(sub_onehot_targets_.get_shape()[2])])  # [batch*t, max_adj_num]

        xent_ = - tf.reduce_sum(sub_onehot_targets_flat * log_probs_, 1) * tf.reshape(seq_mask_, [-1])  # [batch*t]

        # max prediction acc computation
        max_prediction_ = tf.one_hot(tf.argmax(tf.exp(log_probs_) * sub_adj_mask_, dimension=1),
                                     int(adj_mask_.get_shape()[1]),
                                     dtype=config.float_type)  # [batch*t, max_adj_num]

        return xent_, max_prediction_

      xent_, max_prediction_ = constrained_softmax_cross_entropy_loss_with_individual_weights(self.inputs_,
                                                                                              lpi_,
                                                                                              self.sub_onehot_targets_,
                                                                                              self.mask_,
                                                                                              all_w_task_, all_b_task_,
                                                                                              self.adj_mask_)
      self.build_max_predict_loss_layer(max_prediction_,
                                        tf.reshape(self.sub_onehot_targets_,
                                             [-1, int(self.sub_onehot_targets_.get_shape()[2])]),
                                        use_onehot=True)
      loss_p_ = tf.reduce_sum(xent_) / config.batch_size
      if config.individual_task_regularizer > 0 and train_phase:
        if config.individual_task_keep_prob < 1.0:
          print("Warning: you'd better only choose one between dropout and L2-regularizer")
        print("Use L2 regularizer on w in individual task layer")
        loss_ = loss_p_ + config.individual_task_regularizer * tf.nn.l2_loss(all_w_task_)
      else:
        loss_ = loss_p_
      return loss_, loss_p_

  def build_LPIRNN_model(self, train_phase):
    config = self.config
    self.lpi_ = self.build_sharedTask_part(train_phase)
    loss_, loss_p_ = self.build_individualTask_part(train_phase, self.lpi_)
    if config.trace_hid_layer:
      self.trace_dict["lpi_"+str(config.trace_input_id)] = self.lpi_ # here you can collect the lpi w.r.t. a given state id
    self.loss_dict["loss"] = loss_
    self.loss_dict["loss_p"] = loss_p_
    # compute grads and update params
    self.build_trainer(self.loss_dict["loss"], tf.trainable_variables())
    if config.use_v2_saver:
      self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=config.max_ckpt_to_keep,
                                  write_version=saver_pb2.SaverDef.V2)
    else:
      self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=config.max_ckpt_to_keep,
                                  write_version=saver_pb2.SaverDef.V1)

  def build_RNNLM_model(self, train_phase):
    """
    Build the RNN-based language model for trajectory modeling task. Note that CSSRNN is also an RNNLM.
    Here you can build the model by setting `config.model_type`.
    If it is set to "SPECIFIED_CSSRNN", you can specify to use which constrained softmax strategy of whether to build
    unconstrained loss in train/test phase. These settings can be find in config.
    :param train_phase: bool
    :return:
    """
    config = self.config
    # construct embeddings
    emb_inputs_ = self.build_input_layer(train_phase, self.inputs_, config.input_dest, self.dests_label_, config.dest_emb)

    # construct rnn
    rnn_outputs_ = self.build_rnn_layer(emb_inputs_, train_phase)  # [batch, time, hid_dim]
    outputs_flat_ = tf.reshape(rnn_outputs_, [-1, int(rnn_outputs_.get_shape()[2])])  # [batch*t, hid_dim]

    if config.model_type == "CSSRNN":
      self.build_xent_loss_layer(outputs_flat_, True, 'adjmat_adjmask', False) # the default settings of CSSRNN
    elif config.model_type == "RNN":
      self.build_xent_loss_layer(outputs_flat_, use_constrained_softmax=False)
    elif config.model_type == "SPECIFIED_CSSRNN":
      # construct specified losses
      if train_phase:
        self.build_xent_loss_layer(outputs_flat_, config.use_constrained_softmax_in_train,
                                   config.constrained_softmax_strategy,
                                   config.build_unconstrained_in_train)
      else:
        self.build_xent_loss_layer(outputs_flat_, config.use_constrained_softmax_in_test,
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

  def get_batch(self, data, batch_size, max_len, append_EOS=False):
    """
    Get one mini-batch in data which is drawn randomly from the data.
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
    The batch is returned according to the order in `data` to ensure each sample is evaluated.

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
    """
    Construct the fetch dict as the input for `sess.run()`
    :param eval_op: an operation you want to execute in `sess.run()`, e.g., `self.update_op`
    :return: fetch_dict: a dict containing the tensors in `loss_dict`, `debug_tensors` and `trace_dict`
    """
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
      # for example, in this code, we want to trace the latent prediction information w.r.t. a given state id
      # which has already been added in `build_LPIRNN_model()`
      fetch_dict["~"+k] = self.trace_dict[k]
    return fetch_dict

  def feed(self, batch):
    """
    feed one batch to placeholders by constructing the feed dict
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

  def step(self, sess, batch, eval_op=None):
    """
    One step for a batch
    Either sgd training by setting `eval_op` to `self.update_op` or only evaluate the loss by leaving it to be `None`
    :param sess: a tensorflow session
    :param batch: a Batch object
    :param eval_op: an operator in tensorflow
    :return: vals: dict containing the values evaluated by `sess.run()`
    """
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

  def speed_benchmark(self, sess, samples_for_test=1000):
    """
    Train some samples for testing the speed.
    if `self.debug_tensors` is not empty, its the time for outputting these debug tensors.
    Free to modify anything in the block of `if len(self.debug_tensors) > 0:` for debugging intermediate tensors.
    :param sess: tensorflow session
    :param samples_for_test: how many samples you want the benchmarker to test
    """
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
      loss += loss_dict[self.config.loss_for_filename] # specify the loss by what you really care

      # def plot_dir():
      #   map = self.map
      #   e_dir = loss_dict["_e_dir_distri"]
      #   t_dir = loss_dict["_t_dir_distri"]
      #   targets = batch.targets.flatten()
      #   buckets = np.linspace(0.0, 360.0, self.config.dir_granularity + 1)[0:-1]
      #   for i in range(e_dir.shape[0]): # batch*t
      #     def get_adj_distri(edge_id):
      #       edge = map.edges[edge_id]
      #       print("pick up edge " + str(edge.id))
      #       def calc_deg(p1, p2):
      #         cos_val = (p2[0] - p1[0]) / math.sqrt(
      #           (p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1]))
      #         rad = math.acos(cos_val)
      #         if p2[1] < p1[1]:  # [pi, 2pi]
      #           rad = 2 * math.pi - rad
      #         return rad / math.pi * 180.0
      #       adj_dists = []
      #       for adj_edge in edge.adjList:
      #         p1 = map.nodes[adj_edge.startNodeId]
      #         p2 = map.nodes[adj_edge.endNodeId]
      #         deg = calc_deg(p1, p2)
      #         print("adj_edge %d deg = %.3f" % (adj_edge.id, deg))
      #         probs = []
      #         for bucket_deg in np.linspace(0.0, 360.0, self.config.dir_granularity + 1)[0:-1]:
      #           if abs(bucket_deg - deg) < 360.0 - abs(bucket_deg - deg):
      #             delta = abs(bucket_deg - deg)
      #           else:
      #             delta = 360.0 - abs(bucket_deg - deg)
      #           probs.append(math.exp(-delta * delta / (2 * self.config.dir_sigma * self.config.dir_sigma)))
      #         # normalization
      #         summation = sum(probs)
      #         for i in range(len(probs)):
      #           probs[i] /= summation
      #         adj_dists.append(probs)
      #       return adj_dists
      #     adj_distri = get_adj_distri(targets[i])
      #     for distri in adj_distri:
      #       plt.plot(buckets, distri, 'g-')
      #     plt.plot(buckets, e_dir[i], 'b-')
      #     plt.plot(buckets, t_dir[i], 'r-')
      #     plt.title(str(targets[i]))
      #     plt.show()
      # plot_dir()

      # output debug tensors
      if len(self.debug_tensors) > 0:
        print("[Debug mode]")
        # personal codes, here is just an example
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
          # raw_input()

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

  def train_epoch(self, sess, data, epoch):
    config = self.config
    cumulative_losses = {}
    general_step_count = 0
    batch_counter = 0

    steps_per_epoch_in_train = self.config.samples_per_epoch_in_train // self.config.batch_size
    for step_i in range(steps_per_epoch_in_train):
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
      self.loss_logs[(epoch, step_i)] = cumulative_losses

      # collect trace information (here we are collecting the latent prediction information w.r.t state `config.trace_input_id`)
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

      # if it reaches every 10% of training data
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
    # dump LPI
    config = self.config
    for k in self.trace_items.keys():
      np.savetxt(os.path.join(config.save_path, 'label_' + k), self.trace_items[k][0], '%d')
      np.savetxt(os.path.join(config.save_path, 'feature_' + k), self.trace_items[k][1])
      print("trace items have been output to " + config.save_path)

  def eval(self, sess, data, is_valid_set, save_ckpt, model_train = None):
    """
    evaluate all samples in `data`
    :param sess: tensorflow session
    :param data: data for evaluation
    :param is_valid_set: bool, `True` if `data` is the validation set, `False` for the test set
    :param save_ckpt: bool
    :param model_train: TrajModel object (which is for training process), to save the parameters after evaluation
    :return:
    """
    if self.train_phase:
      raise Exception("Evaluation should not be invoked when training (i.e., `self.train_phase` is `True`)")
    if save_ckpt and model_train is None:
      raise Exception("You should not leave `model_train` as `None` if you want to save the checkpoint")
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

      # save training ckpt file
      ckpt_path = os.path.join(self.config.save_path, filename)
      print("saving training ckpt to: " + ckpt_path)
      # params should be saved by `model_train` as model_test (i.e., `self`) do not contain intermediate params of
      # some optimizers such as RMSProp
      model_train.saver.save(sess, ckpt_path)
      print("done")

      # flush IO buffer
      if self.config.direct_stdout_to_file:
        self.config.log_file.close()
        self.config.log_file = open(self.config.log_filename, "a+")
        sys.stdout = self.config.log_file
    print("")
