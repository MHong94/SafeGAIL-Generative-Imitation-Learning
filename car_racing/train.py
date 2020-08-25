import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from GAIL import GAIL
from RAIL import RAIL


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
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                green_penalty += 0.05
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

    def render(self, *arg):
        self.env.render(*arg)

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


def get_distance(position, playfeild):
    x = position[0]
    y = position[1]

    distance_x = abs(playfeild - abs(x))
    distance_y = abs(playfeild - abs(y))

    distance = max(distance_x, distance_y)

    return distance * distance


def train():
    ######### Hyperparameters #########
    env_name = "CarRacing-v0"
    # env_name = "LunarLanderContinuous-v2"
    # env = gym.make(env_name)
    env = Env()

    solved_reward = env.env.spec.reward_threshold  # stop training if solved_reward > avg_reward
    random_seed = 0
    max_timesteps = 1000  # max time steps in one episode
    n_eval_episodes = 20  # evaluate average reward over n episodes
    lr = 0.0002  # learing rate
    discount = 0.995  # reward discount
    cvar_alpha = 0.9  # cvar alpha parameter
    # cvar_beta = 0.  # cvar beta parameter
    cvar_lr = 0.01  # cvar learning rate
    cvar_lambda = 0.5  # cvar lambda parameter
    betas = (0.5, 0.999)  # betas for adam optimizer
    n_epochs = 800  # number of epochs
    n_iter = 100  # updates per epoch
    batch_size = 100  # num of transitions sampled from expert
    directory = "./preTrained/{}".format(env_name)  # save trained models
    filename = "GAIL_{}_{}".format(env_name, random_seed)
    safe = True
    ###################################

    state_dim = env.env.observation_space.shape[0]
    action_dim = env.env.action_space.shape[0]
    max_action = float(env.env.action_space.high[0])

    print("Action space:", env.env.action_space, env.env.action_space.shape,
          env.env.action_space.high[0])
    # policy = GAIL(env_name, state_dim, action_dim, max_action, lr, betas, safe)
    policy = RAIL(env_name, state_dim, action_dim, max_action, lr, betas, discount, cvar_alpha, cvar_lr, cvar_lambda, safe)

    # graph logging variables:
    epochs = []
    rewards = []
    green_penalties = []
    distances_from_mid = []

    running_reward_list = []
    running_green_penalty_list = []

    running_reward = 0
    running_green_penalty = 0

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        env.env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    pretrain_gen_iter = 500
    pretrain_gen_batch_size = 100
    policy.pretrain_generator(pretrain_gen_batch_size, pretrain_gen_iter)

    pretrain_disc_iter = 500
    pretrain_disc_batch_size = 100
    policy.pretrain_discriminator(pretrain_disc_batch_size, pretrain_disc_iter)

    # training procedure
    for epoch in range(1, n_epochs + 1):
        # update policy n_iter times
        # policy.actor.train()
        policy.update(n_iter, discount, batch_size)

        # evaluate in environment
        total_reward = 0
        total_green_penalty = 0
        total_distance_from_mid = 0

        # policy.actor.eval()
        # for episode in range(n_eval_episodes):
        for episode in range(10):
            count_till_start = 0
            state = env.reset()
            for t in range(max_timesteps):
                action = policy.select_action(state)

                if count_till_start > 0:
                    action = [0, 1, 0]
                    count_till_start -= 1

                if t % 50 == 0:
                    print("Action:", action)

                state, reward, done, die, green_penalty, position, playfeild = env.step(action)

                # env.render()

                total_green_penalty += green_penalty
                total_distance_from_mid += get_distance(position, playfeild)
                total_reward += reward

                if done or die:
                    break

        avg_reward = int(total_reward / n_eval_episodes)
        avg_green_penalty = int(total_green_penalty / n_eval_episodes)
        avg_distance_from_mid = int(total_distance_from_mid / n_eval_episodes)

        running_green_penalty = 0.99 * running_green_penalty + 0.01 * avg_green_penalty
        running_reward = 0.99 * running_reward + 0.01 * avg_reward

        tb.add_scalar("Avg Reward", avg_reward, epoch)
        tb.add_scalar("Avg Green Penalty", avg_green_penalty, epoch)
        tb.add_scalar("Avg distance from midz", avg_distance_from_mid, epoch)
        tb.add_scalar("Running Reward", running_reward, epoch)
        tb.add_scalar("Running Green Penalty", running_green_penalty, epoch)

        print("Epoch: {}\tAvg Reward: {}".format(epoch, avg_reward))
        print("Epoch: {}\tAvg Green Penalty: {}".format(epoch, avg_green_penalty))

        # add data for graph
        epochs.append(epoch)
        rewards.append(avg_reward)
        green_penalties.append(avg_green_penalty)
        distances_from_mid.append(avg_distance_from_mid)
        running_reward_list.append(running_reward)
        running_green_penalty_list.append(running_green_penalty)

        if epoch % 20 == 0:
            policy.save(directory, filename)
            np.savetxt('stats/rewards.out', np.array(rewards))
            np.savetxt('stats/green_penalties.out', np.array(green_penalties))
            np.savetxt('stats/distances_from_mid.out', np.array(distances_from_mid))
            np.savetxt('stats/running_reward_list.out', np.array(running_reward_list))
            np.savetxt('stats/running_green_penalty_list.out', np.array(running_green_penalty_list))

        if avg_reward > solved_reward:
            print("########### Solved! ###########")
            policy.save(directory, filename)
            break

    # plot and save graph
    plt.plot(epochs, rewards)
    plt.xlabel('Epochs')
    plt.ylabel('Average Reward')
    plt.title('{}  {}  {} '.format(env_name, lr, betas))
    plt.savefig('./gif/graph_{}.png'.format(env_name))


if __name__ == '__main__':
    comment = f'Unconstrained'
    tb = SummaryWriter(comment=comment)

    train()

    tb.close()
