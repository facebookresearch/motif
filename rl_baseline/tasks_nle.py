# Copyright (c) Facebook, Inc. and its affiliates.
import enum
import re

import gym
import numpy as np
from gym.envs import registration

from nle import nethack
from nle.env import base
from nle.env.base import ASCII_SPACE, ASCII_ESC, SKIP_EXCEPTIONS
from nle.env.tasks import TASK_ACTIONS
from nle.nethack import tty_render


class NetHackScore(base.NLE):
    """Environment for "score" task.

    The task is an augmentation of the standard NLE task. The return function is
    defined as:
    :math:`\text{score}_t - \text{score}_{t-1} + \text{TP}`,
    where the :math:`\text{TP}` is a time penalty that grows with the amount of
    environment steps that do not change the state (such as navigating menus).

    Args:
        penalty_mode (str): name of the mode for calculating the time step
            penalty. Can be ``constant``, ``exp``, ``square``, ``linear``, or
            ``always``. Defaults to ``constant``.
        penalty_step (float): constant applied to amount of frozen steps.
            Defaults to -0.01.
        penalty_time (float): constant applied to amount of frozen steps.
            Defaults to -0.0.

    """

    def __init__(
        self,
        *args,
        penalty_mode="constant",
        penalty_step: float = -0.01,
        penalty_time: float = -0.0,
        **kwargs,
    ):
        self.penalty_mode = penalty_mode
        self.penalty_step = penalty_step
        self.penalty_time = penalty_time

        self._frozen_steps = 0

        self.dungeon_explored = {}

        actions = kwargs.pop("actions", TASK_ACTIONS)
        super().__init__(*args, 
                         actions=actions, 
                         **kwargs)

    def _get_time_penalty(self, last_observation, observation):
        blstats_old = last_observation[self._blstats_index]
        blstats_new = observation[self._blstats_index]

        old_time = blstats_old[nethack.NLE_BL_TIME]
        new_time = blstats_new[nethack.NLE_BL_TIME]

        if old_time == new_time:
            self._frozen_steps += 1
        else:
            self._frozen_steps = 0

        penalty = 0
        if self.penalty_mode == "constant":
            if self._frozen_steps > 0:
                penalty += self.penalty_step
        elif self.penalty_mode == "exp":
            penalty += 2**self._frozen_steps * self.penalty_step
        elif self.penalty_mode == "square":
            penalty += self._frozen_steps**2 * self.penalty_step
        elif self.penalty_mode == "linear":
            penalty += self._frozen_steps * self.penalty_step
        elif self.penalty_mode == "always":
            penalty += self.penalty_step
        else:  # default
            raise ValueError("Unknown penalty_mode '%s'" % self.penalty_mode)
        penalty += (new_time - old_time) * self.penalty_time
        return penalty

    def _reward_fn(self, last_observation, action, observation, end_status):
        """Score delta, but with added a state loop penalty."""
        score_diff = super()._reward_fn(
            last_observation, action, observation, end_status
        )
        time_penalty = self._get_time_penalty(last_observation, observation)
        return score_diff + time_penalty

    def _perform_known_steps(self, observation, done, exceptions=True):
        while not done:

            if observation[self._internal_index][3]:  # xwaitforspace
                # Make sure to include information about going down the stairs.
                previous_msg = observation[self._message_index].copy()
                msg_str = bytes(previous_msg)
                observation, done = self.nethack.step(ASCII_SPACE)
                if b'You descend the stairs.' in msg_str:
                    observation = (*observation[:self._message_index], previous_msg, *observation[self._message_index+1:])
                continue

            internal = observation[self._internal_index]
            in_yn_function = internal[1]
            in_getline = internal[2]

            if in_getline:  # Game asking for a line of text. We don't do that.
                observation, done = self.nethack.step(ASCII_ESC)
                continue

            if in_yn_function:  # Game asking for a single character.
                # Note: No auto-yes to final questions thanks to the disclose option.
                if exceptions:
                    # This causes an annoying unnecessary copy...
                    msg = bytes(observation[self._message_index])

                    # Do not skip some questions to allow agent to select
                    # stuff to eat, attack, and to select directions.
                    # Also do not skip if all allowed or the allowed message appears.
                    if self._allow_all_yn_questions or any(
                        el in msg for el in SKIP_EXCEPTIONS
                    ):
                        break

                # Otherwise, auto-decline.
                observation, done = self.nethack.step(ASCII_ESC)
            break

        return observation, done

    def get_scout_score(self, last_observation):
        glyphs = last_observation[self._glyph_index]
        blstats = last_observation[self._blstats_index]

        dungeon_num = blstats[nethack.NLE_BL_DNUM]
        dungeon_level = blstats[nethack.NLE_BL_DLEVEL]

        key = (dungeon_num, dungeon_level)
        explored = np.sum(glyphs != nethack.GLYPH_CMAP_OFF)
        self.dungeon_explored[key] = explored
        total_explored = 0
        for key, value in self.dungeon_explored.items():
            total_explored += value
        return total_explored

    def step(self, action: int):
        """Steps the environment.

        Args:
            action (int): action integer as defined by ``self.action_space``.

        Returns:
            (dict, float, bool, dict): a tuple containing
                - (*dict*): an observation of the state; this will contain the keys
                  specified by ``self.observation_space``.
                - (*float*): a reward; see ``self._reward_fn`` to see how it is
                  specified.
                - (*bool*): True if the state is terminal, False otherwise.
                - (*dict*): a dictionary of extra information (such as
                  `end_status`, i.e. a status info -- death, task win, etc. --
                  for the terminal state).
        """
        # Careful: By default we re-use Numpy arrays, so copy before!
        last_observation = tuple(a.copy() for a in self.last_observation)

        # Fix the eating action such that it is possible to eat all items
        last_msg = bytes(last_observation[self._message_index]).decode('utf-8')
        if 'What do you want to eat' in last_msg:
            pattern = r'\[([a-zA-Z]+)'
            match = re.search(pattern, last_msg)
            if match and self.actions[action] == ord('y'):
                # Action 'y' for 'yes' will lead to eating any random item in the inventory
                action = ord(match.group(1)[0])
            else:
                # Otherwise escape
                action = ASCII_SPACE
        else:
            action = self.actions[action]

        observation, done = self.nethack.step(action)
        is_game_over = observation[self._program_state_index][0] == 1
        if is_game_over or not self._allow_all_modes:
            observation, done = self._perform_known_steps(
                observation, done, exceptions=True
            )

        self._steps += 1

        self.last_observation = observation

        if self._check_abort(observation):
            end_status = self.StepStatus.ABORTED
        else:
            end_status = self._is_episode_end(observation)
        end_status = self.StepStatus(done or end_status)

        reward = float(
            self._reward_fn(last_observation, action, observation, end_status)
        )

        if end_status and not done:
            # Try to end the game nicely.
            self._quit_game(observation, done)
            done = True

        info = {}
        info["end_status"] = end_status
        info["is_ascended"] = self.nethack.how_done() == nethack.ASCENDED
        info['dlvl'] = last_observation[self._blstats_index][12]
        info['gold'] = last_observation[self._blstats_index][13]
        info['xlvl'] = last_observation[self._blstats_index][18]
        info['scout'] = self.get_scout_score(last_observation)

        return self._get_observation(observation), reward, done, info


class NetHackStaircase(NetHackScore):
    """Environment for "staircase" task.

    This task requires the agent to get on top of a staircase down (>).
    The reward function is :math:`I + \text{TP}`, where :math:`I` is 1 if the
    task is successful, and 0 otherwise, and :math:`\text{TP}` is the time step
    function as defined by `NetHackScore`.
    """

    class StepStatus(enum.IntEnum):
        ABORTED = -1
        RUNNING = 0
        DEATH = 1
        TASK_SUCCESSFUL = 2

    def _is_episode_end(self, observation):
        message = bytes(observation[self._message_index])
        stats = observation[self._blstats_index]
        # Success is when the agent descends the stairs to the 2nd level
        if b'You descend the stairs' in message and stats[24] == 2:
            return self.StepStatus.TASK_SUCCESSFUL
        return self.StepStatus.RUNNING

    def _reward_fn(self, last_observation, action, observation, end_status):
        del action  # Unused
        if end_status == self.StepStatus.TASK_SUCCESSFUL:
            reward = 50
        else:
            reward = 0
        return reward


class NetHackStaircaseLvl3(NetHackStaircase):
    """Environment for "staircase" task.

    This task requires the agent to get on top of a staircase down (>).
    The reward function is :math:`I + \text{TP}`, where :math:`I` is 1 if the
    task is successful, and 0 otherwise, and :math:`\text{TP}` is the time step
    function as defined by `NetHackScore`.
    """

    def _is_episode_end(self, observation):
        message = bytes(observation[self._message_index])
        stats = observation[self._blstats_index]
        if b'You descend the stairs' in message and stats[24] == 3:
            return self.StepStatus.TASK_SUCCESSFUL
        return self.StepStatus.RUNNING


class NetHackStaircaseLvl4(NetHackStaircase):
    """Environment for "staircase" task.

    This task requires the agent to get on top of a staircase down (>).
    The reward function is :math:`I + \text{TP}`, where :math:`I` is 1 if the
    task is successful, and 0 otherwise, and :math:`\text{TP}` is the time step
    function as defined by `NetHackScore`.
    """

    def _is_episode_end(self, observation):
        message = bytes(observation[self._message_index])
        stats = observation[self._blstats_index]
        if b'You descend the stairs' in message and stats[24] == 4:
            return self.StepStatus.TASK_SUCCESSFUL
        return self.StepStatus.RUNNING


class NetHackStaircaseContinual(NetHackStaircase):
    """Environment for "staircase" task.

    This task requires the agent to get on top of a staircase down (>).
    The reward function is :math:`I + \text{TP}`, where :math:`I` is 1 if the
    task is successful, and 0 otherwise, and :math:`\text{TP}` is the time step
    function as defined by `NetHackScore`.

    In this version, the episode only terminates when the agent dies, rather than
    when reaching the goal. It is used for learning from intrinsic rewards only.
    """

    def reset(self, wizkit_items=None):
        self.task_succesful = False
        self.first_time = True
        return super().reset(wizkit_items=wizkit_items)

    def _is_episode_end(self, observation):
        message = bytes(observation[self._message_index])
        stats = observation[self._blstats_index]
        # Success is when the agent descends the stairs to the 2nd level
        if b'You descend the stairs' in message and stats[24] == 2:
            self.task_succesful = True
        # In the continual version we continuing interacting with the environment until the agents dies.
        return self.StepStatus.RUNNING

    def _reward_fn(self, last_observation, action, observation, end_status):
        del action  # Unused
        if self.task_succesful and self.first_time:
            reward = 50
            self.first_time = False
        else:
            reward = 0
        return reward


class NetHackStaircaseLvl3Continual(NetHackStaircaseContinual):
    """Environment for "staircase" task.

    This task requires the agent to get on top of a staircase down (>).
    The reward function is :math:`I + \text{TP}`, where :math:`I` is 1 if the
    task is successful, and 0 otherwise, and :math:`\text{TP}` is the time step
    function as defined by `NetHackScore`.

    In this version, the episode only terminates when the agent dies, rather than
    when reaching the goal. It is used for learning from intrinsic rewards only.
    """

    def _is_episode_end(self, observation):
        message = bytes(observation[self._message_index])
        stats = observation[self._blstats_index]
        if b'You descend the stairs' in message and stats[24] == 3:
            self.task_succesful = True
        return self.StepStatus.RUNNING


class NetHackStaircaseLvl4Continual(NetHackStaircaseContinual):
    """Environment for "staircase" task.

    This task requires the agent to get on top of a staircase down (>).
    The reward function is :math:`I + \text{TP}`, where :math:`I` is 1 if the
    task is successful, and 0 otherwise, and :math:`\text{TP}` is the time step
    function as defined by `NetHackScore`.

    In this version, the episode only terminates when the agent dies, rather than
    when reaching the goal. It is used for learning from intrinsic rewards only.
    """

    def _is_episode_end(self, observation):
        message = bytes(observation[self._message_index])
        stats = observation[self._blstats_index]
        if b'You descend the stairs' in message and stats[24] == 4:
            self.task_succesful = True
        return self.StepStatus.RUNNING


class NetHackOracle(NetHackStaircase):
    """Environment for "oracle" task.

    This task requires the agent to reach the oracle (by standing next to it).
    See `NetHackStaircase` for the reward function.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.oracle_glyph = None
        for glyph in range(nethack.GLYPH_MON_OFF, nethack.GLYPH_PET_OFF):
            if nethack.permonst(nethack.glyph_to_mon(glyph)).mname == "Oracle":
                self.oracle_glyph = glyph
                break
        assert self.oracle_glyph is not None

    def _is_episode_end(self, observation):
        glyphs = observation[self._glyph_index]
        blstats = observation[self._blstats_index]
        x, y = blstats[:2]
        neighbors = glyphs[y - 1 : y + 2, x - 1 : x + 2]
        if np.any(neighbors == self.oracle_glyph):
            return self.StepStatus.TASK_SUCCESSFUL
        return self.StepStatus.RUNNING


class NetHackOracleSober(NetHackStaircase):
    """Environment for "oracle" task.

    This task requires the agent to reach the oracle (by standing next to it).
    See `NetHackStaircase` for the reward function.

    In this version, we explicitly verify for the agent not to be under the state
    of hallucination when standing next to the oracle. It is possible for an RL
    agent to hack the reward of NetHackOracle where it would learn to eat a
    particular corpse in order hallucinate and see the oracle, without ever going
    down the stairs. 
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.oracle_glyph = None
        for glyph in range(nethack.GLYPH_MON_OFF, nethack.GLYPH_PET_OFF):
            if nethack.permonst(nethack.glyph_to_mon(glyph)).mname == "Oracle":
                self.oracle_glyph = glyph
                break
        assert self.oracle_glyph is not None

    def _is_episode_end(self, observation):
        glyphs = observation[self._glyph_index]
        blstats = observation[self._blstats_index]
        x, y = blstats[:2]
        neighbors = glyphs[y - 1 : y + 2, x - 1 : x + 2]
        # When blstats[25] == 512 it means the agent is hallucinating.
        if np.any(neighbors == self.oracle_glyph) and blstats[25] != 512:
            return self.StepStatus.TASK_SUCCESSFUL
        return self.StepStatus.RUNNING


registration.register(
    id="NetHackScore-v1", entry_point=NetHackScore,
)


registration.register(
    id="NetHackStaircase-v1", entry_point=NetHackStaircase,
)


registration.register(
    id="NetHackStaircaseLvl3-v1", entry_point=NetHackStaircaseLvl3,
)


registration.register(
    id="NetHackStaircaseLvl4-v1", entry_point=NetHackStaircaseLvl4,
)


registration.register(
    id="NetHackStaircaseContinual-v1", entry_point=NetHackStaircaseContinual,
)


registration.register(
    id="NetHackStaircaseLvl3Continual-v1", entry_point=NetHackStaircaseLvl3Continual,
)


registration.register(
    id="NetHackStaircaseLvl4Continual-v1", entry_point=NetHackStaircaseLvl4Continual,
)


registration.register(
    id="NetHackOracle-v1", entry_point=NetHackOracle,
)


registration.register(
    id="NetHackOracleSober-v1", entry_point=NetHackOracleSober,
)
