#!/usr/bin/env python
import math
import numpy as np
import time

from collections import OrderedDict
from .environment import EnvironmentWrapper

try:
    from osim.env import L2M2019Env
except ImportError as msg:
    message = msg + '\n' + 'Try execute "pip install -U osim-rl"'
    raise ImportError(message)


class L2MEnvWrapper(EnvironmentWrapper):
    def __init__(
            self,
            visualize=False,
            difficulty=3,
            model="3D",

            max_episode_length=2500,
            frame_skip=1,
            action_fn=None,
            integrator_accuracy=1e-3,

            reward_scale=1.0,
            randomized_start=False,
            delay_reward=False,
            separate_reward=False,
            smooth=False,
            observe_time=False,

            living_bonus=0.0,
            death_penalty=0.0,
            side_deviation_penalty=0.0,
            side_step_penalty=False,
            better_speed=0.0,
            tilt_penalty=0.0,
            rotation_penalty=0.0,
            crossing_legs_penalty=0.0,
            bending_knees_bonus=0.0,
            height_penalty=0.0,

            ep2reload=10,
            episodes=1,
            **params):

        self.model = model
        self.difficulty = difficulty
        self.visualize = visualize
        self.randomized_start = randomized_start
        self.integrator_accuracy = integrator_accuracy
        self.env = env = L2M2019Env(visualize=visualize, integrator_accuracy=self.integrator_accuracy)
        self.env.change_model(model=self.model, difficulty=difficulty)

        super().__init__(env=env, **params)

        self.smooth = smooth
        self.frame_skip = frame_skip
        self.observe_time = observe_time
        self.max_ep_length = max_episode_length - 2
        self.delay_reward = delay_reward
        self.separate_reward = separate_reward

        self.action_fn = action_fn

        self.prev_action = np.full(self.action_space.shape, 0)
        self.previous_state = {}

        # reward shaping
        self.reward_scale = reward_scale
        self.death_penalty = np.abs(death_penalty)
        self.living_bonus = living_bonus
        self.side_dev_coef = side_deviation_penalty
        self.cross_legs_coef = crossing_legs_penalty
        self.bending_knees_coef = bending_knees_bonus
        self.side_step_penalty = side_step_penalty
        self.better_speed = better_speed
        self.tilt_penalty = tilt_penalty
        self.rotation_penalty = rotation_penalty
        self.height_penalty = height_penalty

        self.ep2reload = ep2reload
        self.episodes = episodes

    def _process_action(self, action):
        if self.action_fn == "tanh":
            action_mean = .5
            action_std = .5
            return action * action_std + action_mean
        else:
            return action

    def _process_observation(self, observation):
        observation = preprocess_obs(observation)
        if self.observe_time:
            observation = np.concatenate([observation, [(self.time_step / self.max_ep_length - 0.5) * 2.]])
        return observation

    def reset(self):
        self._last_step_time = time.time()
        self.time_step = 0

        if self.episodes % self.ep2reload == 0:
            self.env = L2M2019Env(
                visualize=self.visualize, integrator_accuracy=self.integrator_accuracy)
            self.env.change_model(
                model=self.model, difficulty=self.difficulty)

        if self.randomized_start:
            self.env.reset(project=True, obs_as_dict=True, init_pose=get_init_pose())
            self.env.osim_model.state_desc = self.env.osim_model.compute_state_desc()
            state_desc = self.env.get_observation_dict()
        else:
            state_desc = self.env.reset(project=True, obs_as_dict=True)

        self.previous_state = {}

        observation = self._process_observation(state_desc)
        if self.observe_time:
            observation = np.concatenate([observation, [-1.0]])

        return observation

    def step(self, action):
        delay_between_steps = time.time() - self._last_step_time
        time.sleep(max(0, self._min_delay_between_steps - delay_between_steps))
        self._last_step_time = time.time()
        reward = 0
        reward_origin = 0

        new_action = self._process_action(action)

        for i in range(self.frame_skip):
            self.time_step += 1
            if self.smooth:
                current_action = self.prev_action + (i + 1) * (self.prev_action - new_action) / self.frame_skip
            else:
                current_action = new_action

            observation, r, done, info = self.env.step(current_action, obs_as_dict=True, project=True)

            if not self.delay_reward:
                reward_origin += r
                reward += self.shape_reward(r, current_action)
            if done:
                self.episodes += 1
                break

        if self.delay_reward:
            reward_origin += self.frame_skip * r
            reward += self.shape_reward(reward_origin, current_action)

        self.prev_action = current_action
        observation = self._process_observation(observation)
        info["reward_origin"] = reward_origin

        return observation, reward, done, info

    def shape_reward(self, reward, current_action):
        state_desc = self.env.get_state_desc()
        shape = 0
        if self.separate_reward:
            reward = 0
        if not self.previous_state:
            self.previous_state = self.env.get_state_desc()

        # death penalty
        if self.time_step * self.frame_skip < self.max_ep_length:
            shape -= self.death_penalty

        # deviation from forward direction penalty
        vy, vz = state_desc['body_vel']['pelvis'][1:]
        side_dev_penalty = (vy ** 2 + vz ** 2)
        shape -= self.side_dev_coef * side_dev_penalty

        shape -= self.height_penalty * (max(0.0, 0.9 - state_desc['body_pos']['pelvis'][1]) +
                                        max(0.0, 1.5 - state_desc['body_pos']['head'][1]))

        # crossing legs penalty
        pelvis_xy = np.array(state_desc['body_pos']['pelvis'])
        left = np.array(state_desc['body_pos']['toes_l']) - pelvis_xy
        right = np.array(state_desc['body_pos']['toes_r']) - pelvis_xy
        axis = np.array(state_desc['body_pos']['head']) - pelvis_xy
        cross_legs_penalty = np.cross(left, right).dot(axis)
        if cross_legs_penalty > 0:
            cross_legs_penalty = 0.0
        shape += self.cross_legs_coef * cross_legs_penalty

        shape += self.living_bonus * self.time_step / self.max_ep_length

        body_parts = list(state_desc['body_vel'])
        p_body = np.array([state_desc['body_pos']['pelvis'][0], -state_desc['body_pos']['pelvis'][2]])
        p_head = np.array([state_desc['body_pos']['head'][0], -state_desc['body_pos']['head'][2]])
        p_pelvis = np.array([state_desc['body_pos']['pelvis'][0], -state_desc['body_pos']['pelvis'][2]])

        legs_xy = np.array(np.mean([-right, -left], axis=0))
        p_legs = np.array([legs_xy[0], -legs_xy[2]])
        v_tgt = self.env.vtgt.get_vtgt(p_body).T

        if self.better_speed:
            v_body = np.array([state_desc['body_vel']['pelvis'][0], -state_desc['body_vel']['pelvis'][2]])
            v_body_whole = np.mean(
                [[state_desc['body_vel'][part][0], -state_desc['body_vel'][part][2]] for part in body_parts], axis=0)
            self.env.d_reward['footstep']['del_v'] += (v_body - v_tgt) * self.env.osim_model.stepsize
            real_del_v = (v_body_whole - v_tgt) * self.env.osim_model.stepsize
            shape -= self.better_speed * self.env.d_reward['weight']['v_tgt_R2'] * np.linalg.norm(
                real_del_v) / self.env.LENGTH0

        # Calculating the angle in radians between v_tgt and torso axis or feet axis
        def to_angle_rad(v1, v2):
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            cos_val = np.clip(np.dot(v1, v2), 0.0, 1.0)
            return np.pi * np.arccos(cos_val) / 180

        vel_vector = v_tgt[0]
        torso_tilt = to_angle_rad(p_head - p_pelvis, vel_vector)
        feet_tilt = np.dot(p_legs, vel_vector)
        tilt_difference = np.power(torso_tilt - feet_tilt, 2)

        # Increase penalty, if tilts more, and scale according to v_tgt length
        shape -= float(
            self.tilt_penalty * (np.power(torso_tilt, 2) + np.power(feet_tilt, 2) + tilt_difference) * np.sqrt(
                sum(v_tgt.T ** 2)))

        # bending knees bonus
        r_knee_flexion = np.minimum(state_desc['joint_pos']['knee_r'][0], 0.)
        l_knee_flexion = np.minimum(state_desc['joint_pos']['knee_l'][0], 0.)
        bend_knees_bonus = np.abs(r_knee_flexion + l_knee_flexion)
        shape += self.bending_knees_coef * bend_knees_bonus

        y_rot = state_desc['body_pos_rot']['pelvis'][1]
        dir_vector = [np.cos(y_rot), np.sin(y_rot)]
        shape -= self.rotation_penalty * to_angle_rad(vel_vector, dir_vector) ** 2

        reward += self.frame_skip * shape

        # side step penalty
        if self.side_step_penalty:
            rx, ry, rz = state_desc['body_pos_rot']['pelvis']
            R = euler_angles_to_rotation_matrix([rx, ry, rz])
            reward *= (1.0 - math.fabs(R[2, 0]))

        self.previous_state = state_desc

        reward *= self.reward_scale

        return reward


# Calculates Rotation Matrix given euler angles.
def euler_angles_to_rotation_matrix(theta):
    R_x = np.array([
        [1, 0, 0],
        [0, math.cos(theta[0]), -math.sin(theta[0])],
        [0, math.sin(theta[0]), math.cos(theta[0])]
    ])
    R_y = np.array([
        [math.cos(theta[1]), 0, math.sin(theta[1])],
        [0, 1, 0],
        [-math.sin(theta[1]), 0, math.cos(theta[1])]
    ])
    R_z = np.array([
        [math.cos(theta[2]), -math.sin(theta[2]), 0],
        [math.sin(theta[2]), math.cos(theta[2]), 0],
        [0, 0, 1]
    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list or type(x) is np.ndarray:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out


def preprocess_obs(state_desc):
    d = flatten_json(state_desc)
    d = OrderedDict(sorted(d.items(), key=lambda t: t[0]))
    result = list(d.values())
    return result


def get_init_pose(ratio=10):
    return np.array(
        [np.random.randint(-ratio, ratio) / 100,  # forward speed
         np.random.randint(-ratio, ratio) / 100,  # rightward speed
         np.random.randint(94 - ratio, 94 + ratio) / 100,  # pelvis height
         np.random.randint(-ratio, ratio) * np.pi / (33 * 180),  # trunk lean
         np.random.randint(-ratio, ratio) * np.pi / (33 * 180),  # [right] hip adduct
         np.random.randint(-ratio, ratio) * np.pi / (33 * 180),  # hip flex
         np.random.randint(-ratio, ratio) * np.pi / (33 * 180),  # knee extend
         np.random.randint(-ratio, ratio) * np.pi / (33 * 180),  # ankle flex
         np.random.randint(-ratio, ratio) * np.pi / (33 * 180),  # [left] hip adduct
         np.random.randint(-ratio, ratio) * np.pi / (33 * 180),  # hip flex
         np.random.randint(-ratio, ratio) * np.pi / (33 * 180),  # knee extend
         np.random.randint(-ratio, ratio) * np.pi / (33 * 180)])  # ankle flex


__all__ = ["L2MEnvWrapper"]
