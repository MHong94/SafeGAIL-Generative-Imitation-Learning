import gym
import numpy as np

from GAIL import GAIL


class Env:
    """
    Environment wrapper for CarRacing
    """

    def __init__(self):
        self.env = gym.make('CarRacing-v0')
        self.env.seed(0)
        self.reward_threshold = self.env.spec.reward_threshold

    def reset(self):
        self.av_r = self.reward_memory()
        self.die = False
        img_rgb = self.env.reset()
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * 4  # four frames for decision
        return np.array(self.stack)

    def step(self, action):
        total_reward = 0
        green_penalty = 0
        for i in range(8):
            # img_rgb, reward, die, _ = self.env.step(action)
            img_rgb, reward, die, _, position, playfeild = self.env.step(action)
            # don't penalize "die state"
            if die:
                reward += 100
            # green penalty
            if np.mean(img_rgb[:, :, 1]) > 175.0:
                green_penalty += 0.05 * np.mean(img_rgb[:, :, 1])
                reward -= 0.05
            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 else False
            if done or die:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == 4
        return np.array(self.stack), total_reward, done, die, green_penalty, position, playfeild

    def render(self, mode, *arg):
        return self.env.render(mode=mode, *arg)

    @staticmethod
    def rgb2gray(rgb, norm=True):
        # rgb image -> gray [0, 1]
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray

    @staticmethod
    def reward_memory():
        # record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory


def get_distance(position):
    x = position[0]
    y = position[1]

    distance = max(x, y)

    return distance * distance


def test():
    env_name = "CarRacing-v0"
    env = Env()

    random_seed = 0
    lr = 0.0002
    betas = (0.5, 0.999)

    directory = "./preTrained/{}".format(env_name)
    filename = "GAIL_Unconstrained_1"
    filename_safe = "GAIL_Constrained_1"
    filename_RAIL = "GAIL_CarRacing-v0_0_RAIL"

    state_dim = env.env.observation_space.shape[0]
    action_dim = env.env.action_space.shape[0]
    max_action = float(env.env.action_space.high[0])

    # policy_unconstrained = GAIL(env_name, state_dim, action_dim, max_action, lr, betas)
    # policy_constrained = GAIL(env_name, state_dim, action_dim, max_action, lr, betas)
    policy_RAIL = GAIL(env_name, state_dim, action_dim, max_action, lr, betas)

    # policy_unconstrained.load(directory, filename)
    # policy_constrained.load(directory, filename_safe)
    policy_RAIL.load(directory, filename_RAIL)

    # policy_unconstrained.actor.eval()
    # policy_constrained.actor.eval()
    policy_RAIL.actor.eval()

    unc_losses = 0
    c_losses = 0
    rail_losses = 0

    iter = 100
    for i in range(iter):
        print("Iter:", i + 1, " out of:", iter)
        states, actions = policy_RAIL.expert.sample(250)
        # _, mod_actions_unc = policy_unconstrained.select_action(states)
        # _, mod_actions_c = policy_constrained.select_action(states)
        _, mod_actions_rail = policy_RAIL.select_action(states)

        # l2_unc = l2(actions, mod_actions_unc)
        # l2_c = l2(actions, mod_actions_c)
        l2_rail = l2(actions, mod_actions_rail)

        # unc_losses += l2_unc
        # c_losses += l2_c
        rail_losses += l2_rail

    # print("unconstrained:", unc_losses / iter)
    # print("constrained:", c_losses / iter)
    print("rail:", rail_losses / iter)


def l2(x, mean):
    return np.mean(np.square(x - mean), axis=0).sum()


test()
