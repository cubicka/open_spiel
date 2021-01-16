import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from utils.lru_cache import LRUCache
import tensorflow.compat.v1 as tfv1
import collections
tfv1.logging.set_verbosity(tfv1.logging.ERROR)

class Losses(collections.namedtuple("Losses", "policy value l2")):
  """Losses from a training step."""

  @property
  def total(self):
    return self.policy + self.value + self.l2

  def __str__(self):
    return ("Losses(total: {:.3f}, policy: {:.3f}, value: {:.3f}, "
            "l2: {:.3f})").format(self.total, self.policy, self.value, self.l2)

  def __add__(self, other):
    return Losses(self.policy + other.policy,
                  self.value + other.value,
                  self.l2 + other.l2)

  def __truediv__(self, n):
    return Losses(self.policy / n, self.value / n, self.l2 / n)

def cascade(x, fns):
  for fn in fns:
    x = fn(x)
  return x

def build_f_graph(f_nn, f_nn_policy, f_nn_value, torso):
    training_f0_torso = torso
    for nn in f_nn:
        training_f0_torso = nn[1](nn[0](training_f0_torso))
    training_f0_policy = training_f0_torso
    for nn in f_nn_policy:
        training_f0_policy = nn(training_f0_policy)
    training_f0_value = training_f0_torso
    for nn in f_nn_value:
        training_f0_value = nn(training_f0_value)
    return training_f0_policy, training_f0_value

def build_g_graph(g_nn, g_nn_state, g_nn_reward, torso):
    g_torso = torso
    for nn in g_nn:
        g_torso = nn[1](nn[0](g_torso))

    g_state = g_torso
    for nn in g_nn_state:
        g_state = nn(g_state)

    g_reward = g_torso
    for nn in g_nn_reward:
        g_reward = nn(g_reward)
    return g_state, g_reward

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
  LAYER_COUNT = 3
  LAYER_DIM = 256
  BN = False

  def __init__(self, o_dim, a_dim, s_dim=8, K=5, lr=0.0002, weight_decay=1e-4):
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




    # # h: representation function
    # # s_0 = h(o_1...o_t)
    # x = o_0 = Input(o_dim)
    # for i in range(self.LAYER_COUNT):
    #   x = Dense(self.LAYER_DIM, activation='elu')(x)
    # s_0 = Dense(s_dim, name='s_0')(x)
    # self.h = Model(o_0, s_0, name="h")
    # # self.h.compile()

    # # g: dynamics function (recurrent in state?) old_state+action -> state+reward
    # # r_k, s_k = g(s_k-1, a_k)
    # s_km1 = Input(s_dim)
    # a_k = Input(self.a_dim)
    # x = Concatenate()([s_km1, a_k])
    # for i in range(self.LAYER_COUNT):
    #   x = Dense(self.LAYER_DIM, activation='elu')(x)
    # s_k = Dense(s_dim, name='s_k')(x)
    # r_k = Dense(1, name='r_k')(x)
    # self.g = Model([s_km1, a_k], [r_k, s_k], name="g")
    # # self.g.compile()

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
    # # self.f.compile()

    # # combine them all
    # self.create_mu(K, lr)




    tfkl = tfv1.keras.layers
    g = tfv1.Graph()
    with g.as_default():
      # Representation model
      h_input = tfv1.placeholder(tfv1.float32, [None, o_dim], name="h_input")
      h_torso = h_input

      h_nn = [[
        tfkl.Dense(self.LAYER_DIM, name=f"h_torso_{i}_dense"),
        tfkl.Activation("relu"),
        ] for i in range(self.LAYER_COUNT)]
      h_nn_final = tfkl.Dense(s_dim, name="h_state")

      for i in range(self.LAYER_COUNT):
        h_torso = h_nn[i][1](h_nn[i][0](h_torso))
        # h_torso = cascade(h_torso, [
        #     tfkl.Dense(self.LAYER_DIM, name=f"h_torso_{i}_dense"),
        #     tfkl.Activation("relu"),
        # ])
    #   h_state = cascade(h_torso, [
    #       tfkl.Dense(s_dim, name="h_state"),
    #   ])
      h_state = h_nn_final(h_torso)
      
      # Dynamic model
      g_input_state = tfv1.placeholder(tfv1.float32, [None, s_dim])
      g_input_action = tfv1.placeholder(tfv1.float32, [None, a_dim])
      g_input = tfv1.concat(axis=1, values=[g_input_state, g_input_action])

      g_nn = [[
        tfkl.Dense(self.LAYER_DIM, name=f"g_torso_{i}_dense"),
        tfkl.Activation("relu"),
        ] for i in range(self.LAYER_COUNT)]
      g_nn_state = [
        tfkl.Dense(self.LAYER_DIM, name="g_state_dense"),
        tfkl.Activation("relu"),
        tfkl.Dense(s_dim, name="g_state"),
      ]
      g_nn_reward = [
        tfkl.Dense(self.LAYER_DIM, name="g_reward_dense"),
        tfkl.Activation("relu"),
        tfkl.Dense(1, name="g_reward"),
      ]

      g_torso = g_input
      for i in range(self.LAYER_COUNT):
        g_torso = g_nn[i][1](g_nn[i][0](g_torso))

      g_state = g_torso
      for nn in g_nn_state:
        g_state = nn(g_state)

      g_reward = g_torso
      for nn in g_nn_reward:
        g_reward = nn(g_reward)

    #   for i in range(self.LAYER_COUNT):
    #     g_torso = cascade(g_torso, [
    #         tfkl.Dense(self.LAYER_DIM, name=f"g_torso_{i}_dense"),
    #         tfkl.Activation("relu"),
    #     ])
    #   g_state = cascade(g_torso, [
    #       tfkl.Dense(self.LAYER_DIM, name="g_state_dense"),
    #       tfkl.Activation("relu"),
    #       tfkl.Dense(s_dim, name="g_state"),
    #   ])
    #   g_reward = cascade(g_torso, [
    #     tfkl.Dense(self.LAYER_DIM, name="g_reward_dense"),
    #     tfkl.Activation("relu"),
    #     tfkl.Dense(1, name="g_reward"),
    #   ])

      # Prediction model
      f_input = tfv1.placeholder(tfv1.float32, [None, s_dim], name="f_input")
      f_torso = f_input  # Ignore the input shape, treat it as a flat array.

      f_nn = [[
        tfkl.Dense(self.LAYER_DIM, name=f"f_torso_{i}_dense"),
        tfkl.Activation("relu"),
        ] for i in range(self.LAYER_COUNT)]
      f_nn_policy = [
        tfkl.Dense(self.LAYER_DIM, name="f_policy_dense"),
        tfkl.Activation("relu"),
        tfkl.Dense(a_dim, name="f_policy"),
      ]
      f_nn_value = [
        tfkl.Dense(self.LAYER_DIM, name="f_value_dense"),
        tfkl.Activation("relu"),
        tfkl.Dense(1, name="f_value"),
        tfkl.Activation("tanh"),
      ]

      for i in range(self.LAYER_COUNT):
        f_torso = f_nn[i][1](f_nn[i][0](f_torso))
        # f_torso = cascade(f_torso, [
        #     tfkl.Dense(self.LAYER_DIM, name=f"f_torso_{i}_dense"),
        #     tfkl.Activation("relu"),
        # ])

      f_policy = f_torso
      for i in range(len(f_nn_policy)):
        f_policy = f_nn_policy[i](f_policy)

      f_value = f_torso
      for i in range(len(f_nn_value)):
        f_value = f_nn_value[i](f_value)
    #   f_policy = cascade(f_torso, [
    #       tfkl.Dense(self.LAYER_DIM, name="f_policy_dense"),
    #       tfkl.Activation("relu"),
    #       tfkl.Dense(a_dim, name="f_policy"),
    #   ])
    # #   f_softmax = tfkl.Softmax()(f_logits)
    #   f_value = cascade(f_torso, [
    #     tfkl.Dense(self.LAYER_DIM, name="f_value_dense"),
    #     tfkl.Activation("relu"),
    #     tfkl.Dense(1, name="f_value"),
    #     tfkl.Activation("tanh"),
    #   ])

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

      target_actions = [
        tfv1.placeholder(tfv1.float32, [None, a_dim], name=f"target_{i}_actions")
        for i in range(K-1)]
      target_policies = [
        tfv1.placeholder(tfv1.float32, [None, a_dim], name=f"target_{i}_policy")
        for i in range(K)]
      target_values = [
        tfv1.placeholder(tfv1.float32, [None, 1], name=f"target_{i}_value")
        for i in range(K)]

    #   training_f0_torso = h_state
    #   for nn in f_nn:
    #     training_f0_torso = nn[1](nn[0](training_f0_torso))
    #   training_f0_policy = training_f0_torso
    #   for nn in f_nn_policy:
    #     training_f0_policy = nn(training_f0_policy)
    #   training_f0_value = training_f0_torso
    #   for nn in f_nn_value:
    #     training_f0_value = nn(training_f0_value)

      training_policy, training_value = build_f_graph(f_nn, f_nn_policy, f_nn_value, h_state)
      loss_policy = tfv1.reduce_mean(
        tfv1.nn.softmax_cross_entropy_with_logits_v2(
        logits=training_policy, labels=target_policies[0]))
      loss_value = tfv1.losses.mean_squared_error(
        training_value, target_values[0])

      training_hs = h_state
      for i in range(K-1):
        training_g_input = tfv1.concat(axis=1, values=[training_hs, target_actions[i]])
        training_hs, _ = build_g_graph(g_nn, g_nn_state, g_nn_reward, training_g_input)
        training_policy, training_value = build_f_graph(f_nn, f_nn_policy, f_nn_value, training_hs)
        loss_policy += tfv1.reduce_mean(
          tfv1.nn.softmax_cross_entropy_with_logits_v2(
          logits=training_policy, labels=target_policies[i+1]))
        loss_value += tfv1.losses.mean_squared_error(
          training_value, target_values[i+1])

      loss_policy /= K
      loss_value /= K

    #   target_f0_policy = tfv1.placeholder(tfv1.float32, [None, a_dim], name="target_f0_policy")
    #   target_f0_value = tfv1.placeholder(tfv1.float32, [None, 1], name="target_f0_value")

    #   loss_f0_policy = tfv1.reduce_mean(
    #     tfv1.nn.softmax_cross_entropy_with_logits_v2(
    #     logits=training_f0_policy, labels=target_f0_policy))
    #   loss_f0_value = tfv1.losses.mean_squared_error(
    #     training_f0_value, target_f0_value)

      l2_reg_loss = tfv1.add_n([
        weight_decay * tfv1.nn.l2_loss(var)
        for var in tfv1.trainable_variables()
        if "/bias:" not in var.name
      ], name="l2_reg_loss")

      total_loss = loss_policy + loss_value + l2_reg_loss
      optimizer = tfv1.train.AdamOptimizer(lr)
      optimizer_train = optimizer.minimize(total_loss, name="train")

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
    self.g_input_state = g_input_state
    self.g_input_action = g_input_action
    self.g_state = g_state
    self.g_reward = g_reward
    self.f_input = f_input # get_var("f_input")
    self.f_policy = f_policy # get_var("f_policy")
    self.f_value = f_value # get_var("f_value")

    # self.target_f0_policy = target_f0_policy
    # self.target_f0_value = target_f0_value
    self.target_actions = target_actions
    self.target_policies = target_policies
    self.target_values = target_values
    self.loss_policy = loss_policy
    self.loss_value = loss_value
    self.l2_reg_loss = l2_reg_loss
    self.optimizer_train = optimizer_train
    # self._train = self.session.graph.get_operation_by_name("train")

  def clear_cache(self):
    self.f_cache.clear()
    self.g_cache.clear()
    self.h_cache.clear()

#   def ht(self, o_0):
#     inp = o_0.tobytes()
#     hidden_state = self.h_cache.make(inp, lambda: self.h.predict(o_0[None]))[0]
#     max_hidden, min_hidden = hidden_state.max(), hidden_state.min()
#     scale = max_hidden - min_hidden
#     if scale < 1e-5: scale += 1e-5
#     return (hidden_state - min_hidden) / scale

#   def gt(self, s_km1, a_k):
#     a_hot = to_one_hot(a_k, self.a_dim)
#     inp = s_km1.tobytes() + a_hot.tobytes()
#     r_k, s_k = self.g_cache.make(inp, lambda: self.g.predict([s_km1[None], a_hot[None]]))
#     hidden_state = s_k[0]
#     max_hidden, min_hidden = hidden_state.max(), hidden_state.min()
#     scale = max_hidden - min_hidden
#     if scale < 1e-5: scale += 1e-5
#     return (hidden_state - min_hidden) / scale, r_k[0][0]

#   def ft(self, s_k):
#     inp = s_k.tobytes()
#     p_k, v_k = self.f_cache.make(inp, lambda: self.f.predict(s_k[None]))
#     return p_k[0], v_k[0][0]

  def ht(self, o_0):
    hidden_state = self.session.run([self.h_state],feed_dict={self.h_input: o_0[None]})[0][0]
    return hidden_state
    # max_hidden, min_hidden = hidden_state.max(), hidden_state.min()
    # scale = max_hidden - min_hidden
    # if scale < 1e-5: scale += 1e-5
    # # print("ht")
    # # print(hidden_state)
    # # print((hidden_state - min_hidden) / scale)
    # return (hidden_state - min_hidden) / scale

  def gt(self, s_km1, a_k):
    a_hot = to_one_hot(a_k, self.a_dim)
    # inp = s_km1.tobytes() + a_hot.tobytes()
    s_k, r_k = self.session.run([self.g_state, self.g_reward], 
        feed_dict={self.g_input_state: s_km1[None], self.g_input_action: a_hot[None]})
    return s_k[0], r_k[0][0]
    # hidden_state = s_k[0]
    # max_hidden, min_hidden = hidden_state.max(), hidden_state.min()
    # scale = max_hidden - min_hidden
    # if scale < 1e-5: scale += 1e-5
    # # print("gt")
    # # print(hidden_state)
    # # print((hidden_state - min_hidden) / scale)
    # return (hidden_state - min_hidden) / scale, r_k[0][0]

  def ft(self, s_k):
    policy, value = self.session.run([self.f_policy, self.f_value],feed_dict={self.f_input: s_k[None]})
    return policy[0], value[0]

  def save_checkpoint(self, path):
    return self.saver.save(self.session, path)

  def load_checkpoint(self, path):
    return self.saver.restore(self.session, path)

  def train(self, obs, acts0, acts1, acts2, acts3, pols0, pols1, pols2, pols3, pols4, rets0, rets1, rets2, rets3, rets4):
    _, l1, l2, l3 = self.session.run([self.optimizer_train, self.loss_policy, self.loss_value, self.l2_reg_loss],
      feed_dict={self.h_input: np.array(obs),
      self.target_actions[0]: acts0,
      self.target_actions[1]: acts1,
      self.target_actions[2]: acts2,
      self.target_actions[3]: acts3,
      self.target_policies[0]: pols0,
      self.target_policies[1]: pols1,
      self.target_policies[2]: pols2,
      self.target_policies[3]: pols3,
      self.target_policies[4]: pols4,
      self.target_values[0]: rets0,
      self.target_values[1]: rets1,
      self.target_values[2]: rets2,
      self.target_values[3]: rets3,
      self.target_values[4]: rets4,
    #   self.target_actions[0]: target_actions[0],
    #   self.target_policies: np.array(target_policies),
    #   self.target_values: np.array(target_values)
      })

    return Losses(l1, l2, l3)

  def check_loss(self, obs, acts0, acts1, acts2, acts3, pols0, pols1, pols2, pols3, pols4, rets0, rets1, rets2, rets3, rets4):
    l1, l2, l3 = self.session.run([self.loss_policy, self.loss_value, self.l2_reg_loss],
      feed_dict={self.h_input: np.array(obs),
      self.target_actions[0]: acts0,
      self.target_actions[1]: acts1,
      self.target_actions[2]: acts2,
      self.target_actions[3]: acts3,
      self.target_policies[0]: pols0,
      self.target_policies[1]: pols1,
      self.target_policies[2]: pols2,
      self.target_policies[3]: pols3,
      self.target_policies[4]: pols4,
      self.target_values[0]: rets0,
      self.target_values[1]: rets1,
      self.target_values[2]: rets2,
      self.target_values[3]: rets3,
      self.target_values[4]: rets4,
    #   self.target_actions[0]: target_actions[0],
    #   self.target_policies: np.array(target_policies),
    #   self.target_values: np.array(target_values)
      })

    return Losses(l1, l2, l3)

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
    p_km1, v_km1 = self.f([s_km1])
    mu_all += [v_km1, p_km1]
    loss_all += ["mse", softmax_ce_logits]

    for k in range(K):
      a_k = Input(self.a_dim, name="a_%d" % k)
      a_all.append(a_k)

      r_k, s_k = self.g([s_km1, a_k])

      # predict + store
      p_k, v_k = self.f([s_k])
      mu_all += [v_k, r_k, p_k]
      loss_all += ["mse", "mse", softmax_ce_logits]
      
      # passback
      s_km1 = s_k

    mu = Model([o_0] + a_all, mu_all)
    # mu.compile(Adam(lr), loss_all)
    self.mu = mu

  @property
  def num_trainable_variables(self):
    return sum(np.prod(v.shape) for v in tfv1.trainable_variables())
