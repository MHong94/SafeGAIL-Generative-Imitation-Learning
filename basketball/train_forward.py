import argparse
import os
import sys
import pickle
import time
import shutil
import visdom

from model import *
from helpers import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

Tensor = torch.DoubleTensor
torch.set_default_tensor_type('torch.DoubleTensor')


def printlog(line):
    print(line)
    with open(save_path + 'log.txt', 'a') as file:
        file.write(line + '\n')


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--trial', type=int, required=True)
parser.add_argument('--model', type=str, required=True, help='NAOMI, SingleRes')
parser.add_argument('--task', type=str, required=True, help='basketball, billiard')
parser.add_argument('--y_dim', type=int, required=True)
parser.add_argument('--rnn_dim', type=int, required=True)
parser.add_argument('--dec1_dim', type=int, required=True)
parser.add_argument('--dec2_dim', type=int, required=True)
parser.add_argument('--dec4_dim', type=int, required=True)
parser.add_argument('--dec8_dim', type=int, required=True)
parser.add_argument('--dec16_dim', type=int, required=True)
parser.add_argument('--n_layers', type=int, required=False, default=2)
parser.add_argument('--seed', type=int, required=False, default=123)
parser.add_argument('--clip', type=int, required=True, help='gradient clipping')
parser.add_argument('--pre_start_lr', type=float, required=True, help='pretrain starting learning rate')
parser.add_argument('--batch_size', type=int, required=False, default=64)
parser.add_argument('--save_every', type=int, required=False, default=50, help='periodically save model')
parser.add_argument('--pretrain', type=int, required=False, default=50,
                    help='num epochs to use supervised learning to pretrain')
parser.add_argument('--highest', type=int, required=False, default=1,
                    help='highest resolution in terms of step size in NAOMI')
parser.add_argument('--cuda', action='store_true', default=True, help='use GPU')

parser.add_argument('--discrim_rnn_dim', type=int, required=True)
parser.add_argument('--discrim_layers', type=int, required=True, default=2)
parser.add_argument('--policy_learning_rate', type=float, default=1e-6,
                    help='policy network learning rate for GAN training')
parser.add_argument('--discrim_learning_rate', type=float, default=1e-3,
                    help='discriminator learning rate for GAN training')
parser.add_argument('--max_iter_num', type=int, default=60000,
                    help='maximal number of main iterations (default: 60000)')
parser.add_argument('--log_interval', type=int, default=1, help='interval between training status logs (default: 1)')
parser.add_argument('--draw_interval', type=int, default=200,
                    help='interval between drawing and more detailed information (default: 50)')
parser.add_argument('--pretrain_disc_iter', type=int, default=2000,
                    help="pretrain discriminator iteration (default: 2000)")
parser.add_argument('--save_model_interval', type=int, default=800, help="interval between saving model (default: 50)")

args = parser.parse_args()

if not torch.cuda.is_available():
    args.cuda = False

# model parameters
params = {
    'task': args.task,
    'batch': args.batch_size,
    'y_dim': args.y_dim,
    'rnn_dim': args.rnn_dim,
    'dec1_dim': args.dec1_dim,
    'dec2_dim': args.dec2_dim,
    'dec4_dim': args.dec4_dim,
    'dec8_dim': args.dec8_dim,
    'dec16_dim': args.dec16_dim,
    'n_layers': args.n_layers,
    'discrim_rnn_dim': args.discrim_rnn_dim,
    'discrim_num_layers': args.discrim_layers,
    'cuda': args.cuda,
    'highest': args.highest,
}

# hyperparameters
pretrain_epochs = args.pretrain
clip = args.clip
start_lr = args.pre_start_lr
batch_size = args.batch_size
save_every = args.save_every

# manual seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if use_gpu:
    torch.cuda.manual_seed_all(args.seed)

# build model
# policy_net = eval(args.model)(params)
policy_net = ConditionalNAOMIExtraMLP(params)
discrim_net = ConditionalDiscriminator(params).double()
if args.cuda:
    policy_net, discrim_net = policy_net.cuda(), discrim_net.cuda()
params['total_params'] = num_trainable_params(policy_net)
print(params)

# create save path and saving parameters
save_path = 'saved/' + args.model + '_' + args.task + '_%03d/' % args.trial
if not os.path.exists(save_path):
    os.makedirs(save_path)
    os.makedirs(save_path + 'model/')

    
previous_path = 'saved/NAOMI_basketball_028/'
policy_state_dict = torch.load(previous_path + 'model/policy_step' + str(args.highest) + '_training_79200.pth')
# policy_state_dict = torch.load(save_path+'model/policy_step'+str(args.highest)+'_training.pth')
policy_net.load_state_dict(policy_state_dict)

discrim_state_dict = torch.load(previous_path + 'model/discrim_step' + str(args.highest) + '_training_79200.pth')
discrim_net.load_state_dict(discrim_state_dict)
    
# Data
# if args.task == 'basketball':
#     test_data = torch.Tensor(pickle.load(open('data/Xte_role.p', 'rb'))).transpose(0, 1)[:, :-1, :]
#     train_data = torch.Tensor(pickle.load(open('data/Xtr_role.p', 'rb'))).transpose(0, 1)[:, :-1, :]
# elif args.task == 'billiard':
#     test_data = torch.Tensor(pickle.load(open('data/billiard_eval.p', 'rb'), encoding='latin1'))[:, :, :]
#     train_data = torch.Tensor(pickle.load(open('data/billiard_train.p', 'rb'), encoding='latin1'))[:, :, :]
# else:
#     print('no such task')
#     exit()


# train_data = torch.Tensor(pickle.load(open('data/Xtr_role.p', 'rb'))).transpose(0, 1)[:, :-1, :]
# safe_set_1 = torch.Tensor([26, 20, 36, 20, 36, 40, 26, 40, 10, 0, 20, 0, 20, 10, 10, 10])
# safe_set_2 = torch.Tensor([10, 20, 20, 20, 20, 30, 10, 30, 26, 0, 36, 0, 36, 20, 26, 20])

# half_data_size = int(train_data.shape[0] / 2)
# print("Half size:", half_data_size)

# safe_set_1 = safe_set_1.repeat(half_data_size, 49, 1)
# safe_set_2 = safe_set_2.repeat(half_data_size, 49, 1)

# print(safe_set_1.shape, safe_set_2.shape)
# safe = torch.cat([safe_set_1, safe_set_2], 0).type(torch.float64)

# train_data = torch.cat([train_data, safe], 2)


train_data = torch.Tensor(pickle.load(open('data/Xtr_role.p', 'rb'))).transpose(0, 1)[:, :-1, :]

safe_in_1 = torch.Tensor([26, 20, 36, 20, 36, 40, 26, 40])
safe_in_2 = torch.Tensor([10, 20, 20, 20, 20, 30, 10, 30])

safe_1 = safe_in_1.repeat(train_data.shape[0], 49, 1).type(torch.float64)
safe_2 = safe_in_2.repeat(train_data.shape[0], 49, 1).type(torch.float64)
print("safe_before:", safe_1.shape, safe_2.shape)

safe = torch.cat([safe_1, safe_2], 0)
# safe = safe_in.repeat(train_data.shape[0], 49, 1).type(torch.float64)
train_data = torch.cat([train_data, train_data], 0)
print(safe.shape, "safe")
train_data = torch.cat([train_data, safe], 2)
print(train_data.shape, "train")


weighted_train_data_1 = torch.load('data/train_data_all_one_mini_safeset.p').to('cuda')
weighted_train_data_2 = torch.load('data/train_data_all_one_mini_safeset_2.p').to('cuda')
weighted_train_data = torch.cat([weighted_train_data_1, weighted_train_data_2], 0)

test_data_1 = torch.load('data/test_data_one_mini_safeset.p').to('cuda')
test_data_2 = torch.load('data/test_data_one_mini_safeset_2.p').to('cuda')
test_data = torch.cat([test_data_1, test_data_2], 0)

perm = torch.randperm(train_data.shape[0])

train_data = train_data[perm]
weighted_train_data = weighted_train_data[perm]

test_data = test_data[torch.randperm(test_data.shape[0])]

print(test_data.shape, train_data.shape, weighted_train_data.shape, "======")

print(test_data.shape, train_data.shape, weighted_train_data.shape, "======")

# figures and statistics
path = 'imgs/conditional_forward_two_mini_safeset_all_data_just_gen_extra_mlp/'
if os.path.exists(path):
    shutil.rmtree(path)
if not os.path.exists(path):
    os.makedirs(path)
vis = visdom.Visdom(env=args.model + args.task + str(args.trial))
win_pre_policy = None
win_pre_path_length = None
win_pre_out_of_bound = None
win_pre_step_change = None


# optimizer
optimizer_policy = torch.optim.Adam(
    filter(lambda p: p.requires_grad, policy_net.parameters()),
    lr=args.policy_learning_rate)
optimizer_discrim = torch.optim.Adam(discrim_net.parameters(), lr=args.discrim_learning_rate)
discrim_criterion = nn.BCELoss()
if use_gpu:
    discrim_criterion = discrim_criterion.cuda()

# stats
exp_p = []
win_exp_p = None
mod_p = []
win_mod_p = None
win_path_length = None
win_out_of_bound = None
win_step_change = None

# Pretrain Discriminator
for i in range(args.pretrain_disc_iter):
    exp_states, exp_actions, w_exp_states, w_exp_actions, exp_seq, model_states_var, model_actions_var, model_seq, mod_stats, exp_stats = \
        collect_samples_interpolate(policy_net, train_data, weighted_train_data, use_gpu, i, args.task, name="pretraining", draw=False,
                                    stats=False)
    model_states = model_states_var.data
    model_actions = model_actions_var.data
    pre_mod_p, pre_exp_p = update_discrim(discrim_net, optimizer_discrim, discrim_criterion, exp_states,
                                          exp_actions, w_exp_states, w_exp_actions, model_states, model_actions, i, dis_times=3.0, use_gpu=use_gpu,
                                          train=True)

    print(i, 'exp: ', pre_exp_p, 'mod: ', pre_mod_p)

    if pre_mod_p < 0.3:
        break

# Save pretrained model
if args.pretrain_disc_iter > 250:
    torch.save(policy_net.state_dict(), save_path + 'model/policy_step' + str(args.highest) + '_pretrained.pth')
    torch.save(discrim_net.state_dict(), save_path + 'model/discrim_step' + str(args.highest) + '_pretrained.pth')

# GAN training
# for i_iter in range(args.max_iter_num):
for i_iter in range(100000):
    ts0 = time.time()
    print("Collecting Data")
    exp_states, exp_actions, w_exp_states, w_exp_actions, exp_seq, model_states_var, model_actions_var, model_seq, mod_stats, exp_stats = \
        collect_samples_interpolate(policy_net, train_data, weighted_train_data, use_gpu, i_iter, args.task, draw=False, stats=False)
    model_states = model_states_var.data
    model_actions = model_actions_var.data

    # draw and stats
    if i_iter % args.draw_interval == 0:
        _, _, _, _, _, _, _, _, mod_stats, exp_stats = \
            collect_samples_interpolate(policy_net, test_data, test_data, use_gpu, i_iter, args.task, draw=True, stats=True)

        # print(mod_stats)
        update = 'append' if i_iter > 0 else None
        win_path_length = vis.line(X=np.array([i_iter // args.draw_interval]),
                                   Y=np.column_stack(
                                       (np.array([exp_stats['ave_length']]), np.array([mod_stats['ave_length']]))),
                                   win=win_path_length, update=update,
                                   opts=dict(legend=['expert', 'model'], title="average path length"))
        win_out_of_bound = vis.line(X=np.array([i_iter // args.draw_interval]),
                                    Y=np.column_stack((np.array([exp_stats['ave_out_of_bound']]),
                                                       np.array([mod_stats['ave_out_of_bound']]))),
                                    win=win_out_of_bound, update=update,
                                    opts=dict(legend=['expert', 'model'], title="average out of bound rate"))
        win_step_change = vis.line(X=np.array([i_iter // args.draw_interval]),
                                   Y=np.column_stack((np.array([exp_stats['ave_change_step_size']]),
                                                      np.array([mod_stats['ave_change_step_size']]))),
                                   win=win_step_change, update=update,
                                   opts=dict(legend=['expert', 'model'], title="average step size change"))

    ts1 = time.time()

    t0 = time.time()
    # update discriminator
    mod_p_epoch, exp_p_epoch = update_discrim(discrim_net, optimizer_discrim, discrim_criterion, exp_states,
                                              exp_actions, w_exp_states, w_exp_actions,
                                              model_states, model_actions, i_iter, dis_times=3.0, use_gpu=use_gpu,
                                              train=True)
    exp_p.append(exp_p_epoch)
    mod_p.append(mod_p_epoch)

    # update policy network
    if i_iter > 3 and mod_p[-1] < 0.8:
        update_policy(policy_net, optimizer_policy, discrim_net, discrim_criterion, model_states_var, model_actions_var, i_iter, use_gpu)
    t1 = time.time()

    if i_iter % args.log_interval == 0:
        print('{}\tT_sample {:.4f}\tT_update {:.4f}\texp_p {:.3f}\tmod_p {:.3f}'.format(
            i_iter, ts1 - ts0, t1 - t0, exp_p[-1], mod_p[-1]))

        update = 'append'
        if win_exp_p is None:
            update = None
        win_exp_p = vis.line(X=np.array([i_iter]),
                             Y=np.column_stack((np.array([exp_p[-1]]), np.array([mod_p[-1]]))),
                             win=win_exp_p, update=update,
                             opts=dict(legend=['expert_prob', 'model_prob'], title="training curve probs"))

    if args.save_model_interval > 0 and (i_iter + 1) % args.save_model_interval == 0:
        torch.save(policy_net.state_dict(), save_path + 'model/policy_step' + str(args.highest) + '_training_' + str(i_iter) + '.pth')
        torch.save(discrim_net.state_dict(), save_path + 'model/discrim_step' + str(args.highest) + '_training_' + str(i_iter) + '.pth')
