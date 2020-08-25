import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ExpertTraj:
    def __init__(self, env_name, safe=True):
        print("=====Loading Data=====")

        states_1 = np.load("./expert_traj/{}/{}_expert_states.npy".format(env_name, env_name))
        states_2 = np.load("./expert_traj/{}/states_new.npy".format(env_name, env_name))
        self.exp_states = np.concatenate([states_1, states_2], 0)

        if safe:
            self.exp_actions = np.load("./expert_traj/{}/safe_actions.npy".format(env_name, env_name))
        else:
            actions_1 = np.load("./expert_traj/{}/{}_expert_actions.npy".format(env_name, env_name))
            actions_2 = np.load("./expert_traj/{}/actions_new.npy".format(env_name, env_name))
            self.exp_actions = np.concatenate([actions_1, actions_2], 0)

        self.n_transitions = len(self.exp_actions)
        print("Total datapoints: ", self.n_transitions)

        print("=====Data Loaded=====")

    def sample(self, batch_size):
        indexes = np.random.randint(0, self.n_transitions, size=batch_size)
        state, action = [], []
        for i in indexes:
            s = self.exp_states[i]
            a = self.exp_actions[i]

            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))

        return np.array(state), np.array(action)


def l2_loss(x, mean):
    return (x - mean).pow(2)
