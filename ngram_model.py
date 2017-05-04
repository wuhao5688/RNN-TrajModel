from __future__ import print_function
import math
import copy
import gc
try: # python2
    import Queue as queue
except ImportError: # python3
    import queue


class Trie(object):
  def __init__(self, count):
    """
    :param count: int, number of states
    """
    self.__root = [{} for i in range(count)]
    return

  def __setitem__(self, key, value):
    """
    `key` should go til the leaf and it will set the leaf value to `value`
    No any checking.
    :param key: list
    :param value: int or float
    :return:
    """
    if len(key) < 2:
      raise Exception("len(key) should at least be 2")
    node = self.__root[key[0]]
    for i in range(1, len(key) - 1):
      if node.get(key[i]) is None:
        node[key[i]] = {}
      node = node[key[i]]
    node[key[-1]] = value

  def __getitem__(self, key):
    """
    :param key: list
    :param value: int or float
    :return: if `key` go til the leaf, return the leaf value
             else return the intermediate node (which is a dict).
             When going down the trie, if any node is `None`, return `None`
    """
    #if len(key) < 2:
    #  raise Exception("len(key) should at least be 2")
    node = self.__root[key[0]]
    for i in range(1, len(key) - 1):
      if node.get(key[i]) is None:
        return None
      node = node[key[i]]
    if len(key) >= 2:
      return node[key[-1]]
    else:
      return node

  def normalize_leaf(self):
    for node in self.__root:
      if len(node) > 0: # some states may not connect to any state, just pass it
        self.__normalize_subtree(node)
    return

  def __normalize_subtree(self, node):
    sample = None # to check whether current node is a node connecting to the leaves
    for v in node.values(): # get the first sample in the childs of `node`
      sample = v
      break
    if sample is None:
      raise Exception("sample is None")

    if not isinstance(sample, dict):  # if current node connects to the leaf nodes, do normalization
      summation = sum([v for v in node.values()])
      for k in node.keys():
        node[k] /= summation
    else:
      for k in node.keys(): # else, recursively normalizes the subtree until meets the leaf nodes
        self.__normalize_subtree(node[k])
    return


class N_gram_model(object):
  def __init__(self, roadnet, config):
    self.roadnet = roadnet
    self.config = config
    return

  def train_bi_gram_fast(self, data, given_dest = False):
    """
    faster ver, but can only compute bi-gram
    :param data: list of lists
    :param given_dest: bool, True to train given destination
    :return: trans: list of dicts if `use_dest` is False
                    otherwise trans will be [edge_count, edge_count] dicts which the first dim indicates the destination.
    """
    # initialization
    print("start initialization")
    trans = []
    roadnet = self.roadnet
    psudo_count = 1.0
    error_count = 0
    for edge in roadnet.edges:
      adjs = {}
      for adj_edge_id in edge.adjList_ids:
        adjs[adj_edge_id] = psudo_count
      trans.append(adjs)
    if given_dest:
      trans_d = [copy.deepcopy(trans) for _ in range(len(roadnet.edges))]

    # do stats
    print("start stats")
    for route in data:
      dest = route[-1]
      for i in range(len(route) - 1):
        if given_dest:
          trans_d[dest][route[i]][route[i + 1]] += 1.0
        else:
          trans[route[i]][route[i + 1]] += 1.0

    # normalization
    print("start normalization")
    if given_dest:
      for dest in range(len(roadnet.edges)):
        for edge_id in range(len(roadnet.edges)):
          summation = sum([v for v in trans_d[dest][edge_id].values()])
          for adj_edge_id in roadnet.edges[edge_id].adjList_ids:
            trans_d[dest][edge_id][adj_edge_id] /= summation
      return trans_d
    else:
      for edge_id in range(len(roadnet.edges)):
        summation = sum([v for v in trans[edge_id].values()])
        for adj_edge_id in roadnet.edges[edge_id].adjList_ids:
          trans[edge_id][adj_edge_id] /= summation
      return trans

  def loss_bi_gram_fast(self, trans, data, max_seq_len, given_dest = False):
    """
    designed for `train_bi_gram_fast()`, fast ver. to compute bi-gram loss
    :param trans: the returned value get from `train_bi_gram()`
    :param data: list of lists, data to compute the loss
    :param max_seq_len: int, sample longer than `max_seq_len` will not be evaluated
    :param given_dest: bool, `True` to train given destination
    :return: loss: float
    """
    print("start evaluating...")
    loss = 0.0
    count = 0
    for route in data:
      if len(route) > max_seq_len or len(route) < 2:
        continue
      count += 1
      dest = route[-1]
      for i in range(len(route) - 1):
        if given_dest:
          loss += math.log(trans[dest][route[i]][route[i + 1]])
        else:
          loss += math.log(trans[route[i]][route[i + 1]])
    loss = -loss / count #len(data)
    return loss

  def train_bi_gram_fast_given_dest_subroutine(self, train_data, interval, edge_flag, verbose = False):
    # initialization
    if verbose:
      print("start initialization")
    trans = []
    roadnet = self.roadnet
    psudo_count = 1.0
    for edge in roadnet.edges:
      adjs = {}
      for adj_edge_id in edge.adjList_ids:
        adjs[adj_edge_id] = psudo_count
      trans.append(adjs)
    trans_d = []
    for _ in range(interval[1] - interval[0]):
      if edge_flag[interval[0] + _]:
        trans_d.append(copy.deepcopy(trans))
      else:
        trans_d.append(None)
      if _ % 100 == 0 and verbose:
        print(_)

    # do stats
    if verbose:
      print("start stats")
    for route in train_data:
      dest = route[-1]
      if dest < interval[0] or dest >= interval[1]:
        continue
      if edge_flag[dest] is False:
        continue
      for i in range(len(route) - 1):
         trans_d[dest-interval[0]][route[i]][route[i + 1]] += 1.0

    # normalization
    if verbose:
      print("start normalization")
    for trans in trans_d:
      if trans is None:
        continue
      else:
        for edge_id in range(len(roadnet.edges)):
          summation = sum([v for v in trans[edge_id].values()])
          for adj_edge_id in roadnet.edges[edge_id].adjList_ids:
            trans[edge_id][adj_edge_id] /= summation
    return trans_d

  def train_n_gram_given_dest_subroutine(self, train_data, n, interval, edge_flag, verbose = False):
    if verbose:
      print("start initialization")
    roadnet = self.roadnet
    trans = Trie(self.config.state_size)
    psudo_count = 1.0
    for edge_id in range(len(roadnet.edges)):
      paths = self.__bfs(roadnet, edge_id, n - 1)
      for path in paths:
        if len(path) != n:
          print(path)
          input()  # TODO
        trans[path] = psudo_count
    #trans_d = [copy.deepcopy(trans) for _ in range(interval[1]-interval[0])]
    trans_d = []
    for _ in range(interval[1] - interval[0]):
      if edge_flag[interval[0] + _]:
        trans_d.append(copy.deepcopy(trans))
      else:
        trans_d.append(None)
      if _ % 100 == 0 and verbose:
        print(_)

    if verbose:
      print("start stats")
    for route in train_data:
      dest = route[-1]
      if dest < interval[0] or dest >= interval[1]:
        continue
      if edge_flag[dest] is False:
        continue
      for i in range(n - 1, len(route)):
        path = route[i - n + 1: i + 1]
        trans_d[dest-interval[0]][path] = trans_d[dest-interval[0]][path] + 1.0

    if verbose:
      print("start normalization")
    # normalization
    for trans in trans_d:
      if trans is not None:
        trans.normalize_leaf()
    return trans_d

  def loss(self, trans, data, n, max_seq_len, use_fast = True, given_dest = False, interval = None):
    """
        designed for `train_n_gram()`, slow but more general (support any n >= 2)
        :param trans: the returned value get from `train_bi_gram()`
        :param data: list of lists, data to compute the loss
        :param n: n for n-gram
        :return: loss: float, averaged by `count`
                 count: int, # of routes evaluated in this function
    """
    loss = 0.0
    predict_correct_count = 0.0
    predict_tot_count = 0.0
    count = 0
    eps = 1e-8
    for route in data:
      if len(route) > max_seq_len or len(route) < n: # skip too long or too short
        continue
      dest = route[-1]
      if given_dest and (dest < interval[0] or dest >= interval[1]): # not in interval, pass
        continue
      count += 1

      if use_fast and n == 2:
        for i in range(len(route) - 1):
          if given_dest:
            adj_dict = trans[dest-interval[0]][route[i]]
          else:
            adj_dict = trans[route[i]]
          prob = adj_dict[route[i + 1]]
          if (abs(max(adj_dict.values()) - prob) < eps):
            same_count = 0
            for v in adj_dict.values():
              if (abs(v - prob) < eps):
                same_count += 1
            predict_correct_count += 1.0 / same_count
          predict_tot_count += 1.0
          loss += math.log(prob)
      else:
        for i in range(n - 1, len(route)):
          path = route[i - n + 1: i + 1]
          """
          if given_dest:
            loss += math.log(trans[dest-interval[0]][path])
          else:
            loss += math.log(trans[path])
          """
          if given_dest:
            adj_dict = trans[dest-interval[0]][path[:-1]]
          else:
            adj_dict = trans[path[:-1]]
          prob = adj_dict[path[-1]]
          if (abs(max(adj_dict.values()) - prob) < eps):
            same_count = 0
            for v in adj_dict.values():
              if (abs(v - prob) < eps):
                same_count += 1
            predict_correct_count += 1.0 / same_count
          predict_tot_count += 1.0
          loss += math.log(prob)

    if count != 0:
      loss = -loss / count
    return loss, count, predict_correct_count, predict_tot_count

  def __bfs(self, roadnet, start_edge_id, depth):
    """
    return all paths start from `start_edge_id` with length = `depth` + 1 (including `start_edge_id` itself)
    which means if depth is 0, the `start_edge_id` it self will be returned
    :param roadnet: Map object
    :param start_edge_id: int
    :param depth: int, >= 0
    :return: list of lists,
      e.g. if `depth` = 2, `start_edge_id` = 100, the result may be [[100,102,104], [100, 156, 134], ... ]
    """
    # depth = 0 do nothing
    # start_node_id = start_edge.endNodeId
    if depth == 0:
      return [[start_edge_id]]
    elif depth == 1:
      return [[start_edge_id, adjEdge_id] for adjEdge_id in roadnet.edges[start_edge_id].adjList_ids]
    elif depth < 0:
      raise Exception("depth should not be smaller than 0")
    result = []
    q = queue.Queue()
    q.put((start_edge_id, 0, [start_edge_id]))  # depth is 0 now
    while not q.empty():
      item = q.get()
      start_edge_id, level, path = item[0], item[1], item[2]
      if level <= depth:
        new_level = level + 1
      else:
        break
      if level == depth:
        result.append(path)
      for adjEdge_id in roadnet.edges[start_edge_id].adjList_ids:
        new_path = copy.copy(path)
        new_path.append(adjEdge_id)
        q.put((adjEdge_id, new_level, new_path))
    return result

  def train_n_gram(self, train_data, n, use_fast = True):
    if n == 2 and use_fast:
      return self.train_bi_gram_fast(train_data, False)

    print("start initialization")
    roadnet = self.roadnet
    trans = Trie(self.config.state_size)
    psudo_count = 1.0
    for edge_id in range(len(roadnet.edges)):
      paths = self.__bfs(roadnet, edge_id, n - 1)
      for path in paths:
        if len(path) != n:
          print(path)
          input()  # TODO
        trans[path] = psudo_count

    print("start stats")
    for route in train_data:
      for i in range(n - 1, len(route)):
        path = route[i - n + 1: i + 1]
        trans[path] = trans[path] + 1.0

    print("start normalization")
    # normalization
    trans.normalize_leaf()
    return trans

  def train_and_eval_given_dest(self, train_data, test_data, n, slice_count, use_fast = True, train_continue = False, verbose = False):
    print("start training and evaluation for %d gram model..." % n)
    tot_loss = 0.0
    tot_count = 0
    tot_predict_correct = 0.0
    tot_predict_count = 0.0
    edge_count = len(self.roadnet.edges)

    if train_continue:
      # restore statistics
      last_dest = int(input("input the right bound of the last successfully computed interval [left, right): "))
      avg_loss = float(input("input the newest loss: "))
      tot_count = int(input("input the newest count: "))
      predict_acc = float(input("input the newest predict acc: "))
      tot_predict_count = int(input("input the newest tot_predict_count: "))
      tot_predict_correct = predict_acc * tot_predict_count
      tot_loss = avg_loss * tot_count
    # figure out the dests which will be used in the test
    edge_flag = [False for _ in range(edge_count)]
    for route in test_data:
      edge_flag[route[-1]] = True

    # construct intervals
    step = edge_count // slice_count + 1
    linspace = range(0, edge_count, step)
    for i in range(len(linspace) - 1):
      interval = [linspace[i], linspace[i + 1]]
      if train_continue and interval[0] < last_dest:
        continue
      print("start processing destination [%d, %d)" % (interval[0], interval[1]))
      # train to get `trans`
      if n==2 and use_fast:
        trans = self.train_bi_gram_fast_given_dest_subroutine(train_data, interval, edge_flag, verbose)
      else:
        trans = self.train_n_gram_given_dest_subroutine(train_data, n, interval, edge_flag, verbose)

      # compute subroutine avg loss
      loss, count, predict_correct, predict_count = \
        self.loss(trans, test_data, n, self.config.max_seq_len, use_fast, given_dest=True, interval=interval)

      tot_loss += (loss * count)
      tot_count += count
      tot_predict_correct += predict_correct
      tot_predict_count += predict_count

      print("current loss for %d-gram model at [%d, %d)= %.8f, ppl = %f, tot_count = %d" %
            (n, interval[0], interval[1], tot_loss / tot_count, math.exp(float(tot_loss / tot_count)), tot_count))
      print("current prediction acc = %.6f, tot_predict_count = %d" %
            (float(tot_predict_correct) / tot_predict_count, tot_predict_count))
      # release the memory
      del trans
      gc.collect()
    # last bucket
    last_interval = [linspace[-1], edge_count]
    print("start processing destination [%d, %d)" % (last_interval[0], last_interval[1]))
    if n == 2 and use_fast:
      trans = self.train_bi_gram_fast_given_dest_subroutine(train_data, last_interval, edge_flag, verbose)
    else:
      trans = self.train_n_gram_given_dest_subroutine(train_data, n, last_interval, edge_flag, verbose)
    loss, count, predict_correct, predict_count = \
      self.loss(trans, test_data, n, self.config.max_seq_len, use_fast, given_dest=True, interval=last_interval)
    tot_loss += (loss * count)
    tot_count += count
    tot_predict_correct += predict_correct
    tot_predict_count += predict_count
    # release the memory
    del trans
    gc.collect()

    #final loss
    loss = tot_loss / tot_count
    print("loss for %d-gram model = %f, ppl = %f, tot_test_count = %d" % (n, loss, math.exp(float(loss)), tot_count))
    print("prediction acc = %.6f, tot_predict_count = %d" % (float(tot_predict_correct) / tot_predict_count, tot_predict_count))
  def loss_n_gram_with_head(self, trans, data, n):
    """
        designed for `train_n_gram()`, slow but more general (support any n >= 2)
        :param trans: the returned value get from `train_bi_gram()`
        :param data: list of lists, data to compute the loss
        :param n: n for n-gram
        :return: loss: float, edge_level_loss: float
    """
    loss = 0.0
    edge_count = 0
    predict_correct_count = 0.0
    predict_tot_count = 0.0
    eps = 1e-8
    for route in data:
      edge_count += len(route)-1
      # n-gram loss
      for i in range(n - 1, len(route)):
        path = route[i - n + 1: i + 1]
        prob = trans[n - 2][path]
        loss += math.log(prob)
        if (abs(max(trans[n-2][path[:-1]].values()) - prob) < eps):
          predict_correct_count += 1.0
        predict_tot_count += 1.0
    print("%d gram done" % n)
    for route in data:
      # 2 ~ n-1 gram loss at the beginning
      for sub_gram in range(2, n):
        if len(route) < sub_gram:
          continue
        path = route[0:sub_gram]
        prob = trans[sub_gram-2][path]
        loss += math.log(prob)
        if (abs(max(trans[sub_gram-2][path[:-1]].values()) - prob) < eps):
          predict_correct_count += 1.0
        predict_tot_count += 1.0
    edge_level_loss = -loss / edge_count
    loss = -loss / len(data)
    acc =predict_correct_count / predict_tot_count
    return loss, edge_level_loss, acc

  def train_and_eval(self, train_data, test_data, n, max_seq_len, use_fast=True, compute_all_gram=True):
    if compute_all_gram:
      trans = []
      for i in range(2, n + 1):
        print("training %d-gram..." % i)
        trans.append(self.train_n_gram(train_data, i, use_fast=False)) # must be False here
        loss, edge_level_loss, acc = self.loss_n_gram_with_head(trans, test_data, i)
        print("loss for %d-gram model = %f, ppl = %f" % (i, loss, math.exp(float(loss))))
        print(
          "edge_level loss for %d-gram model = %f, ppl = %f" % (i, edge_level_loss, math.exp(float(edge_level_loss))))
        print("accuracy for max guess = %f" % acc)
    else:
      trans = self.train_n_gram(train_data, n, use_fast)
      loss, count, predict_correct, predict_tot = self.loss(trans, test_data, n, self.config.max_seq_len, use_fast)
      print("loss for %d-gram model = %f, ppl = %f" % (n, loss, math.exp(float(loss))))
      print("accuracy for max guess = %f" % (float(predict_correct) / predict_tot))


