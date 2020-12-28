from trajectory import Trajectory, TrajectoryState
from mcts_bot import create_children_nodes, mcts_search, policy_with_noise
import numpy as np
from search_node import SearchNode

def nodes_of_state(state, evaluators, priors, action_len, with_random=True):
    """Play one game from state, return the trajectory."""

    nodes = []
    policies = []
    actions = []
    trajectory = Trajectory()

    # print("start game")
    # print(state)
    is_prev_state_simultaneous = False
    while not state.is_terminal():
        policy = None
        root = None
        is_cached = False

        # if cache is not None:
        #     cached_value = cache[0](state.current_player(), state)
        #     if cached_value is not None:
        #         is_cached = True
        #         root = SearchNode(None, state.current_player(), 1)
        #         c_policy, total_reward, visit_count, outcome = cached_value
        #         root.history.total_reward = total_reward
        #         root.history.explore_count = visit_count
        #         root.outcome = outcome
        #         policy = c_policy

        if root is None:
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

            # print(root.children)
            # print(policy)
            policy /= policy.sum()
            # if cache is not None:
            #     cache[1](state.current_player(), state, policy, root.history.total_reward, root.history.explore_count, root.outcome)
        policies.append(policy)

        try:
            if not with_random:
                best_action = root.best_child().action
            else:
                best_action = np.random.choice(len(policy), p=policy)
        except: 
                print("=======", is_prev_state_simultaneous, is_cached, len(root.children))
                # print("chosen", chosen_child)
                print(root)
                for c in root.children:
                    print("==child", c, c.history.explore_count)
                print(policy)
                print("terminal", state._is_terminal)
                # print("hand len", len(state._hands[0]))
                # print("hand len", len(state._hands[1]))
                # print("hand len", len(state._hands[2]))
                raise Exception("Lanjut")

        actions.append(best_action)
        # chosen_child = filter(lambda c: c.action == best_action, root.children)
        chosen_child = next((c for c in root.children if c.action == best_action))

        trajectory.states.append(TrajectoryState(
            state.observation_tensor(state.current_player()), state.current_player(),
            state.legal_actions_mask(state.current_player()), best_action, policy,
            root.history.total_reward / root.history.explore_count if root.outcome is None else root.outcome))

        is_prev_state_simultaneous = state.is_simultaneous_node()
        # print("apply action", state.current_player(), best_action)
        state.apply_action(best_action)

    trajectory.returns = state.returns()
    return (nodes, policies, actions, trajectory)


def next_random_state(state, policies):
    policy_idx = 0
    is_prev_state_simultaneous = True
    while is_prev_state_simultaneous:
        legal_actions = state.legal_actions()
        ps = [policies[policy_idx][action] for action in legal_actions]
        action = np.random.choice(legal_actions, p=policy_with_noise(ps))
        is_prev_state_simultaneous = state.is_simultaneous_node()
        state.apply_action(action)
        policy_idx += 1

    return policy_idx

def play_and_explore(game, evaluators, prior_fns):
    state = game.new_initial_state()
    n_actions = game.num_distinct_actions()

    nodes, policies, actions, trajectory = nodes_of_state(state.clone(), evaluators, prior_fns, n_actions)
    # return trajectory
    trajectories = [trajectory]

    node_idx = 0
    while node_idx < len(nodes) and nodes[node_idx].outcome is None:
        state_copy = state.clone()
        n_steps = next_random_state(state_copy, policies[node_idx:])

        _, _, _, additional_trajectory = nodes_of_state(state_copy, evaluators, prior_fns, n_actions)
        trajectories.append(additional_trajectory)

        for k in range(n_steps):
            # state.apply_action(nodes[node_idx+k].best_child().action)
            state.apply_action(actions[node_idx + k])
        node_idx += n_steps
    return trajectories

def play_and_explain(logger, game, input_state, evaluators, prior_fns, is_shallow=False, is_stupid_enemy=False):
    """Play one game, return the trajectory."""
    state = input_state.clone()
    actions = []
    if is_shallow:
        nodes, policies, acts, trajectory = play_shallow_az(state.clone(), evaluators, prior_fns, game.num_distinct_actions(), is_stupid_enemy)
    else:
        nodes, policies, acts, trajectory = nodes_of_state(state.clone(), evaluators, prior_fns, game.num_distinct_actions(), False)

    logger.print("Initial state:\n{}".format(state))
    for idx, node in enumerate(nodes):
        logger.print("Root ({}):".format(evaluators[state.current_player()](state, state.current_player())))
        logger.print(node.to_str(state))
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
            root.history.total_reward / root.history.explore_count if root.outcome is None else root.outcome))

        is_prev_state_simultaneous = state.is_simultaneous_node()
        # print("apply action", state.current_player(), best_action)
        state.apply_action(best_action)

    trajectory.returns = state.returns()
    return (nodes, policies, actions, trajectory)
