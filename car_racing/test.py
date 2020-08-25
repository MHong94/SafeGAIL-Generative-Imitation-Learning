import gym
import numpy as np
from PIL import Image

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
    n_episodes = 15
    max_timesteps = 1000
    render = False
    save_gif = True

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

    policies = [policy_RAIL]  # , policy_constrained]
    green_penalties = 0
    total_steps = 0
    distances = 0
    positions = []

    for ix, policy in enumerate(policies):
        curr_green_penalty = 0

        imgs = []
        for ep in range(1, n_episodes + 1):
            ep_reward = 0
            state = env.reset()
            for t in range(max_timesteps):
                action = policy.select_action(state)
                state, reward, done, die, green_penalty, position, playfeild = env.step(action)
                total_steps += 1
                positions.append(position)
                distances += get_distance(position)
                curr_green_penalty += green_penalty
                ep_reward += reward
                if render:
                    env.render('rgb_array')
                if save_gif:
                    img = env.render(mode='rgb_array')
                    img = Image.fromarray(img)
                    imgs.append(img)
                    img.save('./gif/safe/{}_{}.jpg'.format(ep, t))
                if done or die:
                    if save_gif:
                        imgs[0].save('gif/' + str(1) + '/' + str(ep) + '.gif', format='GIF', append_images=imgs[1:],
                                     save_all=True, duration=100, loop=0)
                    break

            print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
            env.env.close()
        green_penalties += curr_green_penalty

    print("Green Penalty:", green_penalties / n_episodes)
    print("Distance:", distances / total_steps)
    np.save("positions/positions_unconstrained.out", np.array(positions))


test()
