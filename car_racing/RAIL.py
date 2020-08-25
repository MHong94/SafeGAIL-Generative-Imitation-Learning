import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import ExpertTraj, l2_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(4, 8, kernel_size=4, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.BatchNorm2d(16),
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.BatchNorm2d(32),
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.BatchNorm2d(64),
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.BatchNorm2d(128),
            nn.ReLU(),  # activation
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),  # activation
        )  # output shape (256, 1, 1)

        self.l1 = nn.Linear(256, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        # self.first = nn.Linear(300, 1)
        # self.rest = nn.Linear(300, 2)

        self.max_action = max_action

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))

        # x_first = torch.tanh(self.first(x)) * self.max_action
        # x_rest = torch.sigmoid(self.rest(x)) * self.max_action
        #
        # x = torch.cat([x_first, x_rest], 1)
        x = torch.tanh(self.l3(x)) * self.max_action
        return x


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()

        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(4, 8, kernel_size=4, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.BatchNorm2d(16),
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.BatchNorm2d(32),
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.BatchNorm2d(64),
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.BatchNorm2d(128),
            nn.ReLU(),  # activation
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),  # activation
        )  # output shape (256, 1, 1)

        self.l1 = nn.Linear(256 + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        state = self.cnn_base(state)
        state = state.view(-1, 256)

        # print("Shape:", state.shape, action.shape)
        state_action = torch.cat([state, action], 1)
        x = torch.tanh(self.l1(state_action))
        x = torch.tanh(self.l2(x))
        # x = self.l3(x)
        x = torch.sigmoid(self.l3(x))
        return x


class RAIL:
    def __init__(self, env_name, state_dim, action_dim, max_action, lr, betas, discount, cvar_alpha, cvar_lr, cvar_lambda, safe=True,):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr, betas=betas)

        self.discriminator = Discriminator(state_dim, action_dim).to(device)
        self.optim_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)

        self.max_action = max_action
        self.expert = ExpertTraj(env_name, safe)
        self.discount = discount
        
        self.cvar_nu = np.random.random()
        self.cvar_alpha = cvar_alpha
        self.cvar_lr = cvar_lr
        self.cvar_lambda = cvar_lambda

        self.loss_fn = nn.BCELoss()

    def select_action(self, state):
        # state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        state = torch.FloatTensor(state).to(device)
        actions = self.actor(state).cpu().data.numpy()
        return actions.flatten(), actions

    def update(self, n_iter, discount, batch_size=100):
        for i in range(n_iter):
            # sample expert transitions
            exp_state, exp_action = self.expert.sample(batch_size)
            exp_state = torch.FloatTensor(exp_state).to(device)
            exp_action = torch.FloatTensor(exp_action).to(device)

            # sample expert states for actor
            state, _ = self.expert.sample(batch_size)
            state = torch.FloatTensor(state).to(device)
            action = self.actor(state)
            
            # calculate trajectory costs
            traj_cost = -self.discriminator(state, action)

            #######################
            # update discriminator
            #######################
            self.optim_discriminator.zero_grad()

            # label tensors
            exp_label = torch.ones(size=(batch_size, 1), device=device)
            policy_label = torch.zeros(size=(batch_size, 1), device=device)

            # with expert transitions
            prob_exp = self.discriminator(exp_state, exp_action)
            loss = self.loss_fn(prob_exp, exp_label)

            # with policy transitions
            prob_policy = self.discriminator(state, action.detach())
            loss += self.loss_fn(prob_policy, policy_label)
            
            # with cvar
            geometric_sum = (1-self.discount**len(state))/(1-self.discount)
            cvar_discrim_loss = self.cvar_lambda/(1-self.cvar_alpha) * geometric_sum * (traj_cost >= self.cvar_nu)

            # take gradient step
            loss.backward()
            self.optim_discriminator.step()

            ################
            # update policy
            ################
            if i > 5:
                self.optim_actor.zero_grad()

                loss_actor = -self.discriminator(state, action)
                traj_cost = loss_actor
                cvar_policy_loss = self.cvar_lambda/(1-self.cvar_alpha) * (traj_cost - self.cvar_nu) * (traj_cost >= self.cvar_nu)
                loss_actor.mean().backward()
                self.optim_actor.step()
                
            
            ################
            # update cvar parameter
            ################
            self.cvar_nu -= self.cvar_lr * np.mean(traj_cost >= self.cvar_nu)

    def pretrain_discriminator(self, batch_size, pretrain_iter):
        for i in range(pretrain_iter):
            # sample expert transitions
            exp_state, exp_action = self.expert.sample(batch_size)
            exp_state = torch.FloatTensor(exp_state).to(device)
            exp_action = torch.FloatTensor(exp_action).to(device)

            # sample expert states for actor
            state, _ = self.expert.sample(batch_size)
            state = torch.FloatTensor(state).to(device)
            action = self.actor(state)

            #######################
            # update discriminator
            #######################
            self.optim_discriminator.zero_grad()

            # label tensors
            exp_label = torch.ones(size=(batch_size, 1), device=device)
            policy_label = torch.zeros(size=(batch_size, 1), device=device)

            # with expert transitions
            prob_exp = self.discriminator(exp_state, exp_action)
            loss = self.loss_fn(prob_exp, exp_label)

            # with policy transitions
            prob_policy = self.discriminator(state, action.detach())
            loss += self.loss_fn(prob_policy, policy_label)

            loss.backward()
            self.optim_discriminator.step()

            print("Discriminator Pretrain Iteration:", i + 1)

    def pretrain_generator(self, batch_size, pretrain_iter):
        for i in range(pretrain_iter):
            # sample expert states for actor
            state, exp_action = self.expert.sample(batch_size)
            state = torch.FloatTensor(state).to(device)
            exp_action = torch.FloatTensor(exp_action).to(device)
            action = self.actor(state)

            self.optim_actor.zero_grad()
            loss_actor = l2_loss(exp_action, action)
            loss_actor.mean().backward()
            self.optim_actor.step()

            print("Generator Pretrain Iteration:", i + 1)

    def save(self, directory='./preTrained', name='GAIL'):
        torch.save(self.actor.state_dict(), '{}/{}_actor.pth'.format(directory, name))
        torch.save(self.discriminator.state_dict(), '{}/{}_discriminator.pth'.format(directory, name))

    def load(self, directory='./preTrained', name='GAIL'):
        self.actor.load_state_dict(
            torch.load('{}/{}_actor.pth'.format(directory, name), map_location=torch.device('cpu')))
        self.discriminator.load_state_dict(
            torch.load('{}/{}_discriminator.pth'.format(directory, name), map_location=torch.device('cpu')))
