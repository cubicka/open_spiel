from trajectory import Trajectory, TrajectoryState
from mcts_bot import mcts_search, expand_node
import numpy as np
from search_node import SearchNode

random_state = np.random.default_rng()

def nodes_of_state(az_evaluator, state, is_training=True):
    """Play one game from state, return the trajectory."""

    nodes = []
    policies = []
    actions = []
    trajectory = Trajectory()
    is_prev_state_simultaneous = False

    while not state.is_terminal():
        if is_prev_state_simultaneous:
            if len(root.children) == 0:
                expand_node(az_evaluator, state, root, history_cache, True)
        else:
            history_cache = {}
            root = SearchNode(None, state.current_player(), 1)
            mcts_search(az_evaluator, state, root, history_cache, is_training)

        # root.children.sort(key=SearchNode.sort_key)
        # root.children.reverse()
        nodes.append(root)
        # print("root", state)
        # print(state.legal_actions())

        policy = np.zeros(state.num_actions())
        for c in root.children:
            policy[c.action] = c.history.explore_count

        policy /= policy.sum()
        policies.append(policy)

        if is_training:
            best_action = random_state.choice(len(policy), p=policy)
        else:
            best_action = root.best_child().action
        actions.append(best_action)
        # print("action", state.current_player(), best_action)
        rcount, rreward = 0, 0
        for c in root.children:
            rreward += c.history.total_reward
            rcount += c.history.explore_count
        root.final_value = rreward / rcount

        trajectory.states.append(TrajectoryState(
            state.observation_tensor(), state.current_player(),
            state.legal_actions_mask(), best_action, policy,
            root.final_value))

        root = next((c for c in root.children if c.action == best_action))

        is_prev_state_simultaneous = state.is_simultaneous_node()
        state.apply_action(best_action)

    trajectory.returns = state.returns()
    return (nodes, policies, actions, trajectory)

def play_and_explore(az_evaluator, state):
    state.reset()
    _, _, _, trajectory = nodes_of_state(az_evaluator, state.clone())
    return trajectory

def play_and_explain(logger, az_evaluator, state):
    """Play one game, return the trajectory."""
    state.reset()
    actions = []

    # if is_shallow:
    #     nodes, policies, acts, trajectory = play_shallow_az(state.clone(), evaluators, prior_fns, game.num_distinct_actions(), is_stupid_enemy)
    # else:
    nodes, policies, acts, trajectory = nodes_of_state(az_evaluator, state.clone(), False)

    logger.print("Initial state:\n{}".format(state))
    for idx, node in enumerate(nodes):
        vi, pi = az_evaluator._inference(state.observation_tensor(), state.legal_actions_mask())
        logger.print("Root ({}):".format(vi))
        logger.print(node.to_str(state, True))
        # logger.print()
        logger.print("Children:")
        logger.print("\n" + node.children_str(state))

        # logger.print("Root ({:.3f}):".format(evaluators[state.current_player()](state, state.current_player())))
        # for c in node.children:
        #     cstate = state.clone()
        #     cstate.apply_action(c.action)
        #     logger.print("{}: ({:.3f})".format(state.action_to_string(c.action), evaluators[0](cstate, state.current_player())))

        # action = node.best_child().action
        action = acts[idx]
        action_str = state.action_to_string(state.current_player(), action)
        actions.append(action_str)

        logger.print("======= Sample {}: {} ({:.3f})".format(
            state.current_player(), action_str, policies[idx][action]))
        logger.print("\n\n\n")

        state.apply_action(action)
        logger.print("Next state:\n{}".format(state))

    logger.print("Returns: {}; Actions: {}".format(
        " ".join(map(str, trajectory.returns)), " ".join(actions)))
    logger.print("".center(60, '='))
    logger.print("\n\n")
    return

def play_shallow_az(state, evaluators, priors, action_len, is_stupid_enemy):
    nodes = []
    policies = []
    actions = []
    trajectory = Trajectory()

    is_prev_state_simultaneous = False
    while not state.is_terminal():
        policy = None
        if is_prev_state_simultaneous:
            root = chosen_child
            if len(root.children) == 0:
                state_key = np.expand_dims(state.observation_tensor(state.current_player()), 0).tobytes()
                cached_histories = history_cache[state_key]
                ps = priors[state.current_player()](state)
                root.children = create_children_nodes(state.current_player(), ps, cached_histories)
        else:
            history_cache = {}
            root = mcts_search(evaluators, priors, 2, state, history_cache)

        root.children.sort(key=SearchNode.sort_key)
        root.children.reverse()
        nodes.append(root)
        # print("root", root)

        if policy is None:
            policy = np.zeros(action_len)
            for c in root.children:
                policy[c.action] = c.history.explore_count

            policy /= policy.sum()
        policies.append(policy)

        if state.current_player() == 0:
            az_prior = priors[0](state)
            best_action_prior = max(az_prior, key=lambda probs: probs[1])
            best_action = best_action_prior[0]
        elif is_stupid_enemy:
            best_action = np.random.choice(root.children).action
        else:
            best_action = root.best_child().action

        actions.append(best_action)
        # chosen_child = filter(lambda c: c.action == best_action, root.children)
        chosen_child = next((c for c in root.children if c.action == best_action))

        trajectory.states.append(TrajectoryState(
            state.observation_tensor(state.current_player()), state.current_player(),
            state.legal_actions_mask(state.current_player()), best_action, policy,
            root.history.value(root.player)))

        is_prev_state_simultaneous = state.is_simultaneous_node()
        # print("apply action", state.current_player(), best_action)
        state.apply_action(best_action)

    trajectory.returns = state.returns()
    return (nodes, policies, actions, trajectory)
