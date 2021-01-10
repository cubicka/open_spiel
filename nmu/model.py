import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from utils.lru_cache import LRUCache
import tensorflow.compat.v1 as tfv1
tfv1.logging.set_verbosity(tfv1.logging.ERROR)

def cascade(x, fns):
  for fn in fns:
    x = fn(x)
  return x

def to_one_hot(x,n):
  ret = np.zeros([n])
  if x >= 0:
    ret[x] = 1.0
  return ret

def bstack(bb):
  ret = [[x] for x in bb[0]]
  for i in range(1, len(bb)):
    for j in range(len(bb[i])):
      ret[j].append(bb[i][j])
  return [np.array(x) for x in ret]

def reformat_batch(batch, a_dim, remove_policy=False):
  X,Y = [], []
  for o,a,outs in batch:
    x = [o] + [to_one_hot(x, a_dim) for x in a]
    y = []
    for ll in [list(x) for x in outs]:
      y += ll
    X.append(x)
    Y.append(y)
  X = bstack(X)
  Y = bstack(Y)
  if remove_policy:
    nY = [Y[0]]
    for i in range(3, len(Y), 3):
      nY.append(Y[i])
      nY.append(Y[i+1])
    Y = nY
  else:
    Y = [Y[0]] + Y[2:]
  return X,Y

class MuModel():
  LAYER_COUNT = 4
  LAYER_DIM = 512
  BN = False

  def __init__(self, o_dim, a_dim, s_dim=8, K=5, lr=0.0001):
    self.o_dim = o_dim
    self.a_dim = a_dim
    self.K = K
    # self.losses = []
    self.policy_targets = []
    self.value_targets = []
    # self.with_policy = with_policy
    self.f_cache = LRUCache(2**16)
    self.g_cache = LRUCache(2**16)
    self.h_cache = LRUCache(2**16)

    # h: representation function
    # s_0 = h(o_1...o_t)
    # x = o_0 = Input(o_dim)
    # for i in range(self.LAYER_COUNT):
    #   x = Dense(self.LAYER_DIM, activation='elu')(x)
    #   if i != self.LAYER_COUNT-1 and self.BN:
    #     x = BatchNormalization()(x)
    # s_0 = Dense(s_dim, name='s_0')(x)
    # self.h = Model(o_0, s_0, name="h")

    # # g: dynamics function (recurrent in state?) old_state+action -> state+reward
    # # r_k, s_k = g(s_k-1, a_k)
    # s_km1 = Input(s_dim)
    # a_k = Input(self.a_dim)
    # x = Concatenate()([s_km1, a_k])
    # for i in range(self.LAYER_COUNT):
    #   x = Dense(self.LAYER_DIM, activation='elu')(x)
    #   if i != self.LAYER_COUNT-1 and self.BN:
    #     x = BatchNormalization()(x)
    # s_k = Dense(s_dim, name='s_k')(x)
    # r_k = Dense(1, name='r_k')(x)
    # self.g = Model([s_km1, a_k], [r_k, s_k], name="g")

    # # f: prediction function -- state -> policy+value
    # # p_k, v_k = f(s_k)
    # x = s_k = Input(s_dim)
    # for i in range(self.LAYER_COUNT):
    #   x = Dense(self.LAYER_DIM, activation='elu')(x)
    # #   if i != self.LAYER_COUNT-1 and self.BN:
    # #     x = BatchNormalization()(x)
    # v_k = Dense(1, name='v_k')(x)
    # p_k = Dense(self.a_dim, name='p_k')(x)
    # self.f = Model(s_k, [p_k, v_k], name="f")

    tfkl = tfv1.keras.layers
    g = tfv1.Graph()
    with g.as_default():
      # Representation model
      h_input = tfv1.placeholder(tfv1.float32, [None, o_dim], name="h_input")
      h_torso = h_input
      for i in range(self.LAYER_COUNT):
        h_torso = cascade(h_torso, [
            tfkl.Dense(self.LAYER_DIM, name=f"h_torso_{i}_dense"),
            tfkl.Activation("relu"),
        ])
      h_state = cascade(h_torso, [
          tfkl.Dense(s_dim, name="h_state"),
      ])
      
      # Dynamic model
      g_input = tfv1.placeholder(tfv1.float32, [None, s_dim + a_dim], name="g_input")
      g_torso = g_input
      for i in range(self.LAYER_COUNT):
        g_torso = cascade(g_torso, [
            tfkl.Dense(self.LAYER_DIM, name=f"g_torso_{i}_dense"),
            tfkl.Activation("relu"),
        ])
      g_state = cascade(g_torso, [
          tfkl.Dense(self.LAYER_DIM, name="g_state_dense"),
          tfkl.Activation("relu"),
          tfkl.Dense(s_dim, name="g_state"),
      ])
      g_reward = cascade(g_torso, [
        tfkl.Dense(self.LAYER_DIM, name="g_reward_dense"),
        tfkl.Activation("relu"),
        tfkl.Dense(1, name="g_reward"),
      ])

      # Prediction model
      f_input = tfv1.placeholder(tfv1.float32, [None, s_dim], name="f_input")
      f_torso = f_input  # Ignore the input shape, treat it as a flat array.
      for i in range(self.LAYER_COUNT):
        f_torso = cascade(f_torso, [
            tfkl.Dense(self.LAYER_DIM, name=f"f_torso_{i}_dense"),
            tfkl.Activation("relu"),
        ])

      f_policy = cascade(f_torso, [
          tfkl.Dense(self.LAYER_DIM, name="f_policy_dense"),
          tfkl.Activation("relu"),
          tfkl.Dense(a_dim, name="f_policy"),
      ])
    #   f_softmax = tfkl.Softmax()(f_logits)
      f_value = cascade(f_torso, [
        tfkl.Dense(self.LAYER_DIM, name="f_value_dense"),
        tfkl.Activation("relu"),
        tfkl.Dense(1, name="f_value"),
        tfkl.Activation("tanh"),
      ])

    #   policy_targets = []
    #   value_targets = []
    #   losses = []
    #   for i in range(K):
    #     self.policy_targets.append(tfv1.placeholder(
    #         shape=[None, a_dim], dtype=tfv1.float32, name=f"policy_targets_{i}"))
    #     self.value_targets.append(tfv1.placeholder(
    #         shape=[None, 1], dtype=tfv1.float32, name=f"value_targets_{i}"))
    #     losses.append(tfv1.reduce_mean(
    #         tfv1.nn.softmax_cross_entropy_with_logits_v2(
    #             logits=policy_logits, labels=policy_targets),
    #         name="policy_loss")

    #   optimizer = tfv1.train.AdamOptimizer(lr)
    #   train = optimizer.minimize(total_loss, name="train")

      graph_init = tfv1.variables_initializer(tfv1.global_variables(), name="init_all_vars_op")
      with tfv1.device("/cpu:0"):  # Saver only works on CPU.
        self.saver = tfv1.train.Saver(
            max_to_keep=10000, sharded=False, name="saver")


    #   policy_targets = tfv1.placeholder(
    #         shape=[None, a_dim], dtype=tfv1.float32, name="policy_targets")
    #   policy_loss = tfv1.reduce_mean(
    #         tfv1.nn.softmax_cross_entropy_with_logits_v2(
    #             logits=f_policy, labels=policy_targets),
    #         name="policy_loss")
    #   optimizer = tfv1.train.AdamOptimizer(lr)
    #   train = optimizer.minimize(policy_loss, name="train")


    #   with tf.device("/cpu:0"):  # Saver only works on CPU.
    #     saver = tf.train.Saver(
    #         max_to_keep=10000, sharded=False, name="saver")
    session = tfv1.Session(graph=g)
    session.__enter__()
    session.run(graph_init)
    self.session = session

    def get_var(name):
      return self.session.graph.get_tensor_by_name(name + ":0")

    self.h_input = h_input
    self.h_state = h_state
    self.g_input = g_input
    self.g_state = g_state
    self.g_reward = g_reward
    self.f_input = f_input # get_var("f_input")
    self.f_policy = f_policy # get_var("f_policy")
    self.f_value = f_value # get_var("f_value")
    # self._train = self.session.graph.get_operation_by_name("train")

    # combine them all
    # self.create_mu(K, lr)

  def clear_cache(self):
    self.f_cache.clear()
    self.g_cache.clear()
    self.h_cache.clear()

  def ht(self, o_0):
    inp = o_0.tobytes()
    hidden_state = self.h_cache.make(inp, lambda: self.h.predict(o_0[None]))[0]
    return self.h.predict(np.array(o_0)[None])[0]

  def gt(self, s_km1, a_k):
    a_hot = to_one_hot(a_k, self.a_dim)
    inp = s_km1.tobytes() + a_hot.tobytes()
    r_k, s_k = self.g_cache.make(inp, lambda: self.g.predict([s_km1[None], a_hot[None]]))
    return r_k[0][0], s_k[0]

  def ft(self, s_k):
    inp = s_k.tobytes()
    p_k, v_k = self.f_cache.make(inp, lambda: self.f.predict(s_k[None]))
    # p_k, v_k = self.f.predict(s_k[None])
    return np.exp(p_k[0]), v_k[0][0]

  def ht(self, o_0):
    hidden_state = self.session.run([self.h_state],feed_dict={self.h_input: o_0[None]})[0][0]
    max_hidden, min_hidden = hidden_state.max(), hidden_state.min()
    scale = max_hidden - min_hidden
    if scale < 1e-5: scale += 1e-5
    # print("ht")
    # print(hidden_state)
    # print((hidden_state - min_hidden) / scale)
    return (hidden_state - min_hidden) / scale

  def gt(self, s_km1, a_k):
    a_hot = to_one_hot(a_k, self.a_dim)
    # inp = s_km1.tobytes() + a_hot.tobytes()
    s_k, r_k = self.session.run([self.g_state, self.g_reward], 
        feed_dict={self.g_input: np.concatenate((s_km1, a_hot))[None]})
    hidden_state = s_k[0]
    max_hidden, min_hidden = hidden_state.max(), hidden_state.min()
    scale = max_hidden - min_hidden
    if scale < 1e-5: scale += 1e-5
    # print("gt")
    # print(hidden_state)
    # print((hidden_state - min_hidden) / scale)
    return (hidden_state - min_hidden) / scale, r_k[0][0]

  def ft(self, s_k):
    policy, value = self.session.run([self.f_policy, self.f_value],feed_dict={self.f_input: s_k[None]})
    return policy[0], value[0]

  def save_checkpoint(self, path):
    return self.saver.save(self.session, path)

  def load_checkpoint(self, path):
    return self.saver.restore(self.session, path)

  def train_on_batch(self, batch):
    X,Y = reformat_batch(batch, self.a_dim, not self.with_policy)
    l = self.mu.train_on_batch(X,Y)
    self.losses.append(l)
    return l

  def create_mu(self, K, lr):
    self.K = K
    # represent
    o_0 = Input(self.o_dim, name="o_0")
    s_km1 = self.h(o_0)

    a_all, mu_all, loss_all = [], [], []

    def softmax_ce_logits(y_true, y_pred):
      return tf.nn.softmax_cross_entropy_with_logits_v2(y_true, y_pred)

    # run f on the first state
    if self.with_policy:
      p_km1, v_km1 = self.f([s_km1])
      mu_all += [v_km1, p_km1]
      loss_all += ["mse", softmax_ce_logits]
    else:
      v_km1 = self.f([s_km1])
      mu_all += [v_km1]
      loss_all += ["mse"]

    for k in range(K):
      a_k = Input(self.a_dim, name="a_%d" % k)
      a_all.append(a_k)

      r_k, s_k = self.g([s_km1, a_k])

      # predict + store
      if self.with_policy:
        p_k, v_k = self.f([s_k])
        mu_all += [v_k, r_k, p_k]
        loss_all += ["mse", "mse", softmax_ce_logits]
      else:
        v_k = self.f([s_k])
        mu_all += [v_k, r_k]
        loss_all += ["mse", "mse"]
      
      # passback
      s_km1 = s_k

    mu = Model([o_0] + a_all, mu_all)
    mu.compile(Adam(lr), loss_all)
    self.mu = mu
