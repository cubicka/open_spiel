from trajectory import Trajectory, TrajectoryState
from mcts_bot import mcts_search, policy_with_noise
import numpy as np
from search_node import SearchNode

def nodes_of_state(state, evaluators, priors, action_len, with_random=True):
    """Play one game from state, return the trajectory."""

    nodes = []
    policies = []
    trajectory = Trajectory()

    is_prev_state_simultaneous = False
    while not state.is_terminal():
        if is_prev_state_simultaneous:
            root = chosen_child
        else:
            root = mcts_search(evaluators[state.current_player()], priors[state.current_player()], 2, state)

        root.children.sort(key=SearchNode.sort_key)
        root.children.reverse()
        nodes.append(root)

        policy = np.zeros(action_len)
        # is_solved = root.outcome is not None
        for c in root.children:
            # if not is_solved:
                policy[c.action] = c.history.explore_count
            # else:
            #     policy[c.action] = 1 if c.outcome is not None and c.outcome[state.current_player()] >= root.outcome[state.current_player()] else 0

        policy /= policy.sum()
        policies.append(policy)

        if not with_random:
            best_action = root.best_child().action
        else:
            best_action = np.random.choice(len(policy), p=policy)

        trajectory.states.append(TrajectoryState(
            state.observation_tensor(state.current_player()), state.current_player(),
            state.legal_actions_mask(state.current_player()), best_action, policy,
            root.history.total_reward / root.history.explore_count if root.outcome is None else root.outcome))

        is_prev_state_simultaneous = state.is_simultaneous_node()
        state.apply_action(best_action)

    trajectory.returns = state.returns()
    return (nodes, policies, trajectory)


def next_random_state(state, policies):
    policy_idx = 0
    is_prev_state_simultaneous = True
    while is_prev_state_simultaneous:
        action = np.random.choice(len(policies[policy_idx]), p=policy_with_noise(policies[policy_idx]))
        is_prev_state_simultaneous = state.is_simultaneous_node()
        state.apply_action(action)
        policy_idx += 1

    return policy_idx

def play_and_explore(game, evaluators, prior_fns):
    state = game.new_initial_state()
    n_actions = game.num_distinct_actions()

    nodes, policies, trajectory = nodes_of_state(state, evaluators, prior_fns, n_actions)
    return trajectory
    # trajectories = [trajectory]

    # node_idx = 0
    # while node_idx < len(nodes) and nodes[node_idx].outcome is None:
    #     state_copy = state.clone()
    #     n_steps = next_random_state(state_copy, policies[node_idx:])
    #     _, _, additional_trajectory = nodes_of_state(state_copy, evaluators, prior_fns, n_actions)

    #     trajectories.append(additional_trajectory)
    #     for k in range(n_steps):
    #         state.apply_action(nodes[node_idx+k].best_child().action)
    #     node_idx += n_steps
    # return trajectories

def play_and_explain(logger, game, evaluators, prior_fns):
    """Play one game, return the trajectory."""
    state = game.new_initial_state()
    actions = []
    nodes, policies, trajectory = nodes_of_state(state.clone(), evaluators, prior_fns, game.num_distinct_actions(), False)

    logger.print("Initial state:\n{}".format(state))
    for idx, node in enumerate(nodes):
        logger.print("Root ({:.3f}):".format(evaluators[state.current_player()](state, state.current_player())))
        logger.print(node.to_str(state))
        # logger.print()
        logger.print("Children:")
        logger.print("\n" + node.children_str(state))

        # logger.print("Root ({:.3f}):".format(evaluators[state.current_player()](state, state.current_player())))
        # for c in node.children:
        #     cstate = state.clone()
        #     cstate.apply_action(c.action)
        #     logger.print("{}: ({:.3f})".format(state.action_to_string(c.action), evaluators[0](cstate, state.current_player())))

        action = node.best_child().action
        action_str = state.action_to_string(state.current_player(), action)
        actions.append(action_str)

        logger.print("======= Sample {}: {} ({:.3f})".format(
            state.current_player(), action_str, policies[idx][action]))
        logger.print("\n\n\n")

        state.apply_action(action)
        logger.print("Next state:\n{}".format(state))

    logger.print("Returns: {}; Actions: {}".format(
        " ".join(map(str, trajectory.returns)), " ".join(actions)))
    return
