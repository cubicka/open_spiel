from trajectory import Trajectory, TrajectoryState
from mcts_bot import mcts_search
import numpy as np

def play(logger, game, evaluators, prior_fns, fprint):
    """Play one game, return the trajectory."""
    trajectory = Trajectory()
    actions = []
    state = game.new_initial_state()

    if fprint:
        logger.print("Starting game".center(60, "-"))
        logger.print("Initial state:\n{}".format(state))

    is_prev_state_simultaneous = False
    while not state.is_terminal():
        if is_prev_state_simultaneous:
            root = chosen_child
        else:
            root = mcts_search(evaluators, prior_fns, 2, state)

        policy = np.zeros(game.num_distinct_actions())
        root_solved = root.outcome is not None
        for c in root.children:
            if not root_solved:
                policy[c.action] = c.explore_count
            else:
                policy[c.action] = 1 if c.outcome is not None and c.outcome[state.current_player()] >= root.outcome[state.current_player()] else 0

        policy /= policy.sum()        
        action = root.best_child().action
        chosen_child = filter(lambda c: c.action == action, root.children)

        trajectory.states.append(TrajectoryState(
            state.observation_tensor(), state.current_player(),
            state.legal_actions_mask(), action, policy,
            root.total_reward / root.explore_count if root.outcome is None else root.outcome))

        action_str = state.action_to_string(state.current_player(), action)
        actions.append(action_str)
        if fprint:
            logger.print("Root:")
            logger.print("\n", root.to_str(state))
            logger.print("Children:")
            logger.print("\n" + root.children_str(state))
            logger.print("======= Sample {}: {} ({})".format(
                state.current_player(), action_str, policy[action]))
            logger.print("\n\n\n")

        is_prev_state_simultaneous = state.is_simultaneous_node()
        state.apply_action(action)
        if fprint:
            logger.print("Next state:\n{}".format(state))

    trajectory.returns = state.returns()
    if fprint:
        logger.print("Returns: {}; Actions: {}".format(
            " ".join(map(str, trajectory.returns)), " ".join(actions)))
    return trajectory
