"""
Driving environment based on:
- https://github.com/dsadigh/driving-preferences
- https://github.com/Stanford-ILIAD/easy-active-learning/
"""

import copy
import os
from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
from matplotlib.image import AxesImage, BboxImage
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from scipy.ndimage import rotate, zoom
from scipy.special import expit

from src.constants import DRIVER_METADATA_PATH
from src.utils import get_pairs_from_list, timeit

IMG_FOLDER = str(DRIVER_METADATA_PATH)
GRASS = np.tile(plt.imread(os.path.join(IMG_FOLDER, "grass.png")), (5, 5, 1))

CAR = {
    color: zoom(
        np.array(
            plt.imread(os.path.join(IMG_FOLDER, "car-{}.png".format(color))) * 255.0,
            dtype=np.uint8,  # zoom requires uint8 format
        ),
        [0.3, 0.3, 1.0],
    )
    for color in ["gray", "orange", "purple", "red", "white", "yellow"]
}

COLOR_AGENT = "orange"
COLOR_ROBOT = "white"

CAR_AGENT = CAR[COLOR_AGENT]
CAR_ROBOT = CAR[COLOR_ROBOT]
CAR_SCALE = 0.15 / max(list(CAR.values())[0].shape[:2])

LANE_SCALE = 10.0
LANE_COLOR = (0.4, 0.4, 0.4)  # 'gray'
LANE_BCOLOR = "white"

STEPS = 100


def set_image(
    obj,
    data,
    scale=CAR_SCALE,
    x=[0.0, 0.0, 0.0, 0.0],
):
    ox = x[0]
    oy = x[1]
    angle = x[2]
    img = rotate(data, np.rad2deg(angle))
    h, w = img.shape[0], img.shape[1]
    obj.set_data(img)
    obj.set_extent(
        [
            ox - scale * w * 0.5,
            ox + scale * w * 0.5,
            oy - scale * h * 0.5,
            oy + scale * h * 0.5,
        ]
    )


class Car:
    def __init__(self, initial_state, actions):
        self.initial_state = initial_state
        self.state = self.initial_state
        self.actions = actions
        self.action_i = 0

    def reset(self):
        self.state = self.initial_state
        self.action_i = 0

    def update(self, update_fct) -> None:
        u1, u2 = self.actions[self.action_i % len(self.actions)]
        self.state = update_fct(self.state, u1, u2)
        self.action_i += 1

    def gaussian(self, x, height=0.07, width=0.03):
        car_pos = np.asarray([self.state[0], self.state[1]])
        car_theta = self.state[2]
        car_heading = (np.cos(car_theta), np.sin(car_theta))
        pos = np.asarray([x[0], x[1]])
        d = car_pos - pos
        dh = np.dot(d, car_heading)
        dw = np.cross(d, car_heading)
        return np.exp(-0.5 * ((dh / height) ** 2 + (dw / width) ** 2))


class Lane:
    def __init__(
        self,
        start_pos,
        end_pos,
        width,
    ):
        self.start_pos = np.asarray(start_pos)
        self.end_pos = np.asarray(end_pos)
        self.width = width
        d = self.end_pos - self.start_pos
        self.dir = d / np.linalg.norm(d)
        self.perp = np.asarray([-self.dir[1], self.dir[0]])

    def gaussian(self, state, sigma=0.5):
        pos = np.asarray([state[0], state[1]])
        dist_perp = np.dot(pos - self.start_pos, self.perp)
        return np.exp(-0.5 * (dist_perp / (sigma * self.width / 2.0)) ** 2)

    def direction(self, x):
        return np.cos(x[2]) * self.dir[0] + np.sin(x[2]) * self.dir[1]

    def shifted(self, m):
        return Lane(
            self.start_pos + self.perp * self.width * m,
            self.end_pos + self.perp * self.width * m,
            self.width,
        )


def get_lane_x(lane):
    if lane == "left":
        return -0.17
    elif lane == "right":
        return 0.17
    elif lane == "middle":
        return 0
    else:
        raise Exception("Unknown lane:", lane)


class Driver:
    def __init__(
        self,
        cars,
        reward_weights,
        threshold=0,
        starting_lane="middle",
        starting_speed=0.41,
    ):
        initial_x = get_lane_x(starting_lane)
        self.initial_state = [initial_x, -0.1, np.pi / 2, starting_speed]
        self.state = self.initial_state

        self.episode_length = 20
        self.dt = 0.2

        self.friction = 1
        self.vmax = 1
        self.xlim = (-0.7, 0.7)
        # self.ylim = (-0.2, 0.8)
        self.ylim = (-0.2, 2)

        lane = Lane([0.0, -1.0], [0.0, 1.0], 0.17)
        road = Lane([0.0, -1.0], [0.0, 1.0], 0.17 * 3)
        self.lanes = [lane.shifted(0), lane.shifted(-1), lane.shifted(1)]
        self.fences = [lane.shifted(2), lane.shifted(-2)]
        self.roads = [road]
        self.cars = cars

        n_features_reward = len(self.get_reward_features())
        assert reward_weights.shape == (n_features_reward,)
        self.reward_w = np.array(reward_weights)
        self.n_features_reward = n_features_reward

        self.action_d = 2
        self.action_min = np.array([-1, -1])
        self.action_max = np.array([1, 1])

        self.time = 0
        self.history = []
        self._update_history()

    def _update_history(self):
        self.history.append((np.array(self.state), self._get_car_states()))

    def _get_car_states(self):
        return [np.array(car.state) for car in self.cars]

    def get_full_state(self):
        state = copy.deepcopy(self.state)
        for car in self.cars:
            state.extend(car.state)
        return np.array(state)

    def _update_state(self, state, u1, u2):
        x, y, theta, v = state
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = v * u1
        dv = u2 - self.friction * v
        new_v = max(min(v + dv * self.dt, self.vmax), -self.vmax)
        return [x + dx * self.dt, y + dy * self.dt, theta + dtheta * self.dt, new_v]

    def _get_reward_for_state(self, full_state=None):
        if full_state is not None:
            assert len(full_state) == (
                4 + 4 * len(self.cars)
            ), "Full state has insufficient size"
        return np.dot(self.reward_w, self.get_reward_features(full_state))

    def get_comparison_from_feature_diff(self, feature_diff):
        p = expit(np.dot(feature_diff, self.reward_w)).item()
        feedback = np.random.choice([1, 0], p=[p, 1 - p])
        return feedback

    def get_comparison_from_features(self, feature_1, feature_2):
        feature_diff = feature_1 - feature_2
        p = expit(np.dot(feature_diff, self.reward_w)).item()
        feedback = np.random.choice([1, 0], p=[p, 1 - p])
        return feedback

    def get_comparison_from_full_states(self, state_1, state_2):
        feature_1 = self.get_reward_features(state_1)
        feature_2 = self.get_reward_features(state_2)
        feature_diff = feature_1 - feature_2
        p = expit(np.dot(feature_diff, self.reward_w)).item()
        feedback = np.random.choice([1, 0], p=[p, 1 - p])
        return feedback

    def get_hard_comparison_from_full_states(self, state_1, state_2):
        feature_1 = self.get_reward_features(state_1)
        feature_2 = self.get_reward_features(state_2)
        feature_diff = feature_1 - feature_2
        p = expit(np.dot(feature_diff, self.reward_w)).item()
        feedback = 1 if p >= 0.5 else 0
        return feedback

    def get_hard_comparison_from_feature_diff(self, feature_diff):
        p = expit(np.dot(feature_diff, self.reward_w)).item()
        feedback = 1 if p >= 0.5 else 0
        return feedback

    def step(self, action):
        action = np.array(action)
        u1, u2 = action

        self.state = self._update_state(self.state, u1, u2)
        for car in self.cars:
            car.update(self._update_state)

        self.time += 1
        done = bool(self.time >= self.episode_length)
        reward = self._get_reward_for_state()
        self._update_history()
        return np.array(self.state + [self.time]), reward, done, dict()

    def reset(self):
        self.state = self.initial_state
        self.time = 0
        for car in self.cars:
            car.reset()
        self.history = []
        self._update_history()
        return np.array(self.state + [self.time])

    def get_reward_features(self, full_state=None):
        if full_state is not None:
            assert len(full_state) == (
                4 + 4 * len(self.cars)
            ), "Full state has insufficient size"
        return self._get_features(full_state=full_state)

    def _get_features(self, full_state=None):
        if full_state is None:
            x, y, theta, v = self.state
        else:
            assert len(full_state) == (
                4 + 4 * len(self.cars)
            ), "Full state has insufficient size"
            x, y, theta, v = full_state[:4]
        off_street = int(np.abs(x) > self.roads[0].width / 2)

        b = 10000
        a = 10
        d_to_lane = np.min([(x - 0.17) ** 2, x**2, (x + 0.17) ** 2])
        not_in_lane = 1 / (1 + np.exp(-b * d_to_lane + a))

        big_angle = np.abs(np.cos(theta))

        drive_backward = int(v < 0)
        too_fast = int(v > 0.6)

        distance_to_other_car = 0
        b = 30
        a = 0.01

        if full_state is None:
            for car in self.cars:
                car_x, car_y, car_theta, car_v = car.state
                distance_to_other_car += np.exp(
                    -b * (10 * (x - car_x) ** 2 + (y - car_y) ** 2) + b * a
                )
        else:
            car_states = [
                full_state[4 * (i + 1) : 4 * (i + 2)] for i in range(len(self.cars))
            ]
            for car_state in car_states:
                car_x, car_y, car_theta, car_v = car_state
                distance_to_other_car += np.exp(
                    -b * (10 * (x - car_x) ** 2 + (y - car_y) ** 2) + b * a
                )
        keeping_speed = -np.square(v - 0.4)
        target_location = -np.square(x - 0.17)

        return np.array(
            [
                keeping_speed,
                target_location,
                off_street,
                not_in_lane,
                big_angle,
                drive_backward,
                too_fast,
                distance_to_other_car,
            ],
            dtype=float,
        )

    def simulate(self, policy):
        done = False
        s = self.reset()
        r = 0
        while not done:
            a = policy[int(s[-1])]
            s, reward, done, info = self.step(a)
            r += reward
        return reward

    def _get_features_from_flat_policy(self, policy):
        a_dim = self.action_d
        n_policy_steps = len(policy) // a_dim
        n_repeat = self.episode_length // n_policy_steps

        self.reset()
        r_features = np.zeros_like(self.get_reward_features())
        for i in range(self.episode_length):
            if i % n_repeat == 0:
                action_i = a_dim * (i // n_repeat)
                action = (policy[action_i], policy[action_i + 1])
            s, _, done, _ = self.step(action)
            assert (i < self.episode_length - 1) or done
            r_features += self.get_reward_features()
        return r_features

    def _get_representation_from_flat_policy(
        self, policy, representation_function: Callable
    ):
        a_dim = self.action_d
        n_policy_steps = len(policy) // a_dim
        n_repeat = self.episode_length // n_policy_steps

        self.reset()
        representations = []
        for i in range(self.episode_length):
            if i % n_repeat == 0:
                action_i = a_dim * (i // n_repeat)
                action = (policy[action_i], policy[action_i + 1])
            s, _, done, _ = self.step(action)
            assert (i < self.episode_length - 1) or done
            representations.append(representation_function())
        return np.vstack(representations)

    def get_render_state(self):
        render_state = copy.deepcopy(self.state)
        for car in self.cars:
            pos_x, pos_y, theta, vel = car.state
            render_state.append(pos_x)
            render_state.append(pos_y)
        return render_state

    def get_optimal_policy(self, theta=None, restarts=30, n_action_repeat=10):
        a_dim = self.action_d
        eps = 1e-5
        n_policy_steps = self.episode_length // n_action_repeat
        a_low = list(self.action_min + eps)
        a_high = list(self.action_max - eps)
        if theta is None:
            theta = self.reward_w

        def func(policy):
            reward_features = self._get_features_from_flat_policy(policy)
            return -np.array(reward_features).dot(theta)

        opt_val = np.inf
        bounds = list(zip(a_low, a_high)) * n_policy_steps
        for i in range(restarts):
            x0 = np.random.uniform(
                low=a_low * n_policy_steps,
                high=a_high * n_policy_steps,
                size=(n_policy_steps * a_dim,),
            )
            temp_res = opt.fmin_l_bfgs_b(func, x0=x0, bounds=bounds, approx_grad=True)
            if temp_res[1] < opt_val:
                optimal_policy = temp_res[0]
                opt_val = temp_res[1]
        policy_repeat = []
        for i in range(n_policy_steps):
            policy_repeat.extend([optimal_policy[2 * i : 2 * i + 2]] * n_action_repeat)
        return np.array(policy_repeat)

    @timeit
    def get_optimal_policy_from_reward_function(
        self,
        reward_function: Callable,
        representation_function: Callable,
        restarts=30,
        n_action_repeat=10,
    ):
        """_summary_

        Args:
            reward_function (Callable): _description_
            representation_function (Callable): _description_
            restarts (int, optional): _description_. Defaults to 30.
            n_action_repeat (int, optional): _description_. Defaults to 10.

        Returns:
            _type_: _description_
        """
        a_dim = self.action_d
        eps = 1e-5
        n_policy_steps = self.episode_length // n_action_repeat
        a_low = list(self.action_min + eps)
        a_high = list(self.action_max - eps)

        def func(policy):
            representations = self._get_representation_from_flat_policy(
                policy, representation_function
            )
            rewards = 0
            for representation in representations:
                rewards += reward_function(representation.tobytes())
            return -rewards

        opt_val = np.inf
        bounds = list(zip(a_low, a_high)) * n_policy_steps
        for i in range(restarts):
            x0 = np.random.uniform(
                low=a_low * n_policy_steps,
                high=a_high * n_policy_steps,
                size=(n_policy_steps * a_dim,),
            )
            temp_res = opt.fmin_l_bfgs_b(func, x0=x0, bounds=bounds, approx_grad=True)
            if temp_res[1] < opt_val:
                optimal_policy = temp_res[0]
                opt_val = temp_res[1]
        policy_repeat = []
        for i in range(n_policy_steps):
            policy_repeat.extend([optimal_policy[2 * i : 2 * i + 2]] * n_action_repeat)
        return np.array(policy_repeat)

    def render(self, mode="human"):
        if mode not in ("human", "rgb_array", "human_static"):
            raise NotImplementedError("render mode {} not supported".format(mode))
        fig = plt.figure(figsize=(7, 7))

        ax = plt.gca()
        ax.set_xlim(self.xlim[0], self.xlim[1])
        ax.set_ylim(self.ylim[0], self.ylim[1])
        ax.set_aspect("equal")

        grass = BboxImage(ax.bbox, interpolation="bicubic", zorder=-1000)
        grass.set_data(GRASS)
        ax.add_artist(grass)

        for lane in self.lanes:
            path = Path(
                [
                    lane.start_pos
                    - LANE_SCALE * lane.dir
                    - lane.perp * lane.width * 0.5,
                    lane.start_pos
                    - LANE_SCALE * lane.dir
                    + lane.perp * lane.width * 0.5,
                    lane.end_pos + LANE_SCALE * lane.dir + lane.perp * lane.width * 0.5,
                    lane.end_pos + LANE_SCALE * lane.dir - lane.perp * lane.width * 0.5,
                    lane.start_pos
                    - LANE_SCALE * lane.dir
                    - lane.perp * lane.width * 0.5,
                ],
                [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY],
            )
            ax.add_artist(
                PathPatch(
                    path,
                    facecolor=LANE_COLOR,
                    lw=0.5,
                    edgecolor=LANE_BCOLOR,
                    zorder=-100,
                )
            )

        for car in self.cars:
            img = AxesImage(ax, interpolation="bicubic", zorder=20)
            set_image(img, CAR_ROBOT, x=car.state)
            ax.add_artist(img)

        human = AxesImage(ax, interpolation=None, zorder=100)
        set_image(human, CAR_AGENT, x=self.state)
        ax.add_artist(human)

        plt.axis("off")
        plt.tight_layout()
        if mode != "human_static":
            fig.canvas.draw()
            rgb = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
            rgb = rgb.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            del fig
            if mode == "rgb_array":
                return rgb
            elif mode == "human":
                plt.imshow(rgb, origin="upper")
                plt.axis("off")
                plt.tight_layout()
                plt.pause(0.05)
                plt.clf()
        return fig, ax

    def get_policy_frames(self, policy):
        imgs = []
        s = self.reset()
        done = False
        while not done:
            a = policy[int(s[-1])]
            s, _, done, _ = self.step(a)
            img = self.render("rgb_array")
            imgs.append(img)
        self.reset()
        return imgs

    def get_trajectory_frames(self, trajectory):
        ims = []
        self.reset()
        for step in range(trajectory.shape[0]):
            self.state = trajectory[step, :4]
            for car in self.cars:
                car.update(self._update_state)
            self.time += 1
            done = bool(self.time >= self.episode_length)
            reward = self._get_reward_for_state()
            self._update_history()
            ims.append(self.render("rgb_array"))
        return ims

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def plot_query_states_pair(self, query_state_1, query_state_2, label):
        """Assumes that the query state is the full state of the agent plus the
           position of each other agent.
        Args:
            query_state_1 (_type_): _description_
            query_state_2 (_type_): _description_
            label (_type_): _description_

        Returns:
            _type_: _description_
        """
        fig, axs = plt.subplots(2, figsize=(7, 14))

        if label == 1:
            queries = (query_state_1, query_state_2)
        else:
            queries = (query_state_2, query_state_1)

        for i, query in enumerate(queries):

            x, y, angle, v, *cars_pos = query
            assert len(cars_pos) == 2 * len(
                self.cars
            ), "car coordinates doesn't equal the number of cars"
            cars_pos = [cars_pos[i : i + 2] for i in range(0, len(cars_pos), 2)]

            axs[i].set_xlim(self.xlim[0], self.xlim[1])
            axs[i].set_ylim(self.ylim[0], self.ylim[1])
            axs[i].set_aspect("equal")
            axs[i].grid(False)
            axs[i].set_yticks([])
            axs[i].set_xticks([])

            grass = BboxImage(axs[i].bbox, interpolation="bicubic", zorder=-1000)

            grass.set_data(GRASS)
            axs[i].add_artist(grass)
            for lane in self.lanes:
                path = Path(
                    [
                        lane.start_pos
                        - LANE_SCALE * lane.dir
                        - lane.perp * lane.width * 0.5,
                        lane.start_pos
                        - LANE_SCALE * lane.dir
                        + lane.perp * lane.width * 0.5,
                        lane.end_pos
                        + LANE_SCALE * lane.dir
                        + lane.perp * lane.width * 0.5,
                        lane.end_pos
                        + LANE_SCALE * lane.dir
                        - lane.perp * lane.width * 0.5,
                        lane.start_pos
                        - LANE_SCALE * lane.dir
                        - lane.perp * lane.width * 0.5,
                    ],
                    [
                        Path.MOVETO,
                        Path.LINETO,
                        Path.LINETO,
                        Path.LINETO,
                        Path.CLOSEPOLY,
                    ],
                )
                axs[i].add_artist(
                    PathPatch(
                        path,
                        facecolor=LANE_COLOR,
                        lw=0.5,
                        edgecolor=LANE_BCOLOR,
                        zorder=-100,
                    )
                )
            for j, car in enumerate(self.cars):
                car_pos = cars_pos[j]
                img = AxesImage(axs[i], interpolation="bicubic", zorder=20)
                set_image(img, CAR_ROBOT, x=[car_pos[0], car_pos[1], np.pi / 2, 0])
                axs[i].add_artist(img)

            if i == 0:
                axs[i].set_title("Better")
            else:
                axs[i].set_title("Worse")

            human = AxesImage(axs[i], interpolation=None, zorder=100)
            set_image(human, CAR_AGENT, x=[x, y, angle, v])
            axs[i].add_artist(human)

            plt.axis("off")
            plt.tight_layout()
        return fig

    def plot_query_trajectory_pair(self, query_trajectory_1, query_trajectory_2, label):
        fig, ax = self.render("human_static")
        ax.plot(
            query_trajectory_1[:, 0],
            query_trajectory_1[:, 1],
            zorder=10,
            linestyle="-",
            color="green" if label == 1 else "red",
            linewidth=2.5,
            marker="o",
            markersize=8,
            markevery=1,
        )
        ax.plot(
            query_trajectory_2[:, 0],
            query_trajectory_2[:, 1],
            zorder=10,
            linestyle="-",
            color="green" if label == 0 else "red",
            linewidth=2.5,
            marker="o",
            markersize=8,
            markevery=1,
        )
        for i in range(4, query_trajectory_1.shape[1], 2):

            ax.plot(
                query_trajectory_1[:, i],
                query_trajectory_1[:, i + 1],
                zorder=10,
                linestyle="-",
                color=COLOR_ROBOT,
                linewidth=2.5,
                marker="o",
                markersize=8,
                markevery=1,
            )

        return fig

    def plot_history(self):
        x_player = []
        y_player = []
        N_cars = len(self.cars)
        x_cars = [[] for _ in range(N_cars)]
        y_cars = [[] for _ in range(N_cars)]
        for player_state, car_states in self.history:
            x_player.append(player_state[0])
            y_player.append(player_state[1])
            for i in range(N_cars):
                x_cars[i].append(car_states[i][0])
                y_cars[i].append(car_states[i][1])
        self.reset()
        self.render(mode="human_static")
        plt.axis("off")
        plt.tight_layout()
        for i in range(N_cars):
            plt.plot(
                x_cars[i],
                y_cars[i],
                zorder=10,
                linestyle="-",
                color=COLOR_ROBOT,
                linewidth=2.5,
                marker="o",
                markersize=8,
                markevery=1,
            )
            plt.plot(
                x_cars[i][self.episode_length // 2],
                y_cars[i][self.episode_length // 2],
                zorder=10,
                linestyle="-",
                color="red",
                linewidth=2.5,
                marker="o",
                markersize=8,
                markevery=1,
            )
        plt.plot(
            x_player,
            y_player,
            zorder=10,
            linestyle="-",
            color=COLOR_AGENT,
            linewidth=2.5,
            marker="o",
            markersize=8,
            markevery=1,
        )
        plt.plot(
            x_player[self.episode_length // 2],
            y_player[self.episode_length // 2],
            zorder=10,
            linestyle="-",
            color="red",
            linewidth=2.5,
            marker="o",
            markersize=8,
            markevery=1,
        )

    def sample_features_rewards(self, n_samples):
        min_val = -1
        max_val = 1
        samples = min_val + (max_val - min_val) * np.random.sample(
            (n_samples, self.Ndim_repr)
        )
        samples[:, -1] = 0
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        samples[:, -1] = 1
        return samples, np.matmul(samples, self.reward_w.T)


def get_cars(cars_trajectory):
    if cars_trajectory == "blocked":
        # three cars
        x1 = -0.17
        y1 = 0.6
        s1 = 0
        car1 = Car([x1, y1, np.pi / 2.0, s1], [(0, s1)] * 20)
        x2 = 0
        y2 = 0.65
        s2 = 0
        car2 = Car([x2, y2, np.pi / 2.0, s2], [(0, s2)] * 20)
        x3 = 0.17
        y3 = 0.7
        s3 = 0
        car3 = Car([x3, y3, np.pi / 2.0, s3], [(0, s3)] * 20)
        cars = [car1, car2, car3]
    elif cars_trajectory == "changing_lane":
        # car driving from right to middle lane
        car_x = get_lane_x("right")
        straight_speed = 0.328
        car = Car(
            [car_x, 0, np.pi / 2.0, 0.41],
            [(0, straight_speed)] * 5
            + [(1, straight_speed)] * 6
            + [(-1, straight_speed)] * 6
            + [(0, straight_speed)] * 3,
        )
        cars = [car]
    else:
        raise Exception("Unknown cars trajectory:", cars_trajectory)
    return cars


def get_reward_weights(goal, penalty_lambda):
    if goal == "target_velocity":
        goal_weights = np.array(
            [
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            dtype=float,
        )
    elif goal == "target_location":
        goal_weights = np.array(
            [
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            dtype=float,
        )
    else:
        raise Exception("Unknown goal:", goal)
    penalty_weights = np.array(
        [
            0,  # keep speed
            0,  # target location
            0.3,  # off street
            0.05,  # not in lane
            0.02,  # big angle
            0.5,  # drive backward
            0.3,  # too fast
            0.8,  # crash
        ],
        dtype=float,
    )
    return goal_weights - penalty_lambda * penalty_weights


def get_driver(
    cars_trajectory,
    goal,
    penalty_lambda=0,
    reward_weights=None,
):
    cars = get_cars(cars_trajectory)
    if reward_weights is None:
        reward_weights = get_reward_weights(goal, penalty_lambda)
    else:
        reward_weights = reward_weights

    if cars_trajectory == "blocked":
        starting_speed = 0.1
    else:
        starting_speed = 0.41

    return Driver(
        cars=cars,
        reward_weights=reward_weights,
        starting_speed=starting_speed,
    )


def get_driver_target_velocity(blocking_cars=False, reward_weights=None):
    if blocking_cars:
        cars_trajectory = "blocked"
    else:
        cars_trajectory = "changing_lane"
    return get_driver(
        cars_trajectory,
        "target_velocity",
        penalty_lambda=1,
        reward_weights=reward_weights,
    )


def get_driver_target_velocity_only_reward(blocking_cars=False):
    if blocking_cars:
        cars_trajectory = "blocked"
    else:
        cars_trajectory = "changing_lane"
    return get_driver(cars_trajectory, "target_velocity", penalty_lambda=0)


def get_driver_target_location():
    return get_driver("changing_lane", "target_location", penalty_lambda=0.5)


def get_driver_target_location_only_reward():
    return get_driver("changing_lane", "target_location", penalty_lambda=0)


if __name__ == "__main__":
    import time

    env = get_driver_target_velocity()
    policy = env.get_optimal_policy()
    s = env.reset()
    done = False
    r = 0
    while not done:
        a = policy[int(s[-1])]
        s, reward, done, info = env.step(a)
        r += reward
        env.render("human")
        time.sleep(0.2)
    env.plot_history()
    plt.show()
