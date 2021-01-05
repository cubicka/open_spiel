import collections

class Config(collections.namedtuple(
    "Config", [
        "game",
        "cp_num",
        "path",
        "learning_rate",
        "weight_decay",
        "train_batch_size",
        "replay_buffer_size",
        "replay_buffer_reuse",
        "max_steps",
        "checkpoint_freq",
        "actors",
        "evaluators",
        "evaluation_window",
        "eval_levels",

        "uct_c",
        "max_simulations",
        "policy_alpha",
        "policy_epsilon",
        "temperature",
        "temperature_drop",

        "nn_model",
        "nn_width",
        "nn_depth",
        "observation_shape",
        "output_size",
        "value_size",

        "quiet",
    ])):
  """A config for the model/experiment."""
  pass

game_name='nimmt'
az_config = Config(
    game=game_name,
    cp_num=None,
    path='./ma/' + game_name,
    learning_rate=0.001,
    weight_decay=1e-4,
    train_batch_size=2**10,
    replay_buffer_size=2**14,
    replay_buffer_reuse=4,
    max_steps=0,
    checkpoint_freq=1000,

    actors=2,
    evaluators=1,
    uct_c=2,
    max_simulations=300,
    policy_alpha=1,
    policy_epsilon=0.25,
    temperature=1,
    temperature_drop=10,
    evaluation_window=50,
    eval_levels=7,

    nn_model="mlp",
    nn_width=256,
    nn_depth=10,
    observation_shape=None,
    output_size=None,
    value_size=None,

    quiet=True,
)
