import pickle

from helpers import *
from model import *

params = {
    'task': 'basketball',
    'batch': 64,
    'y_dim': 10,
    'rnn_dim': 300,
    'dec1_dim': 200,
    'dec2_dim': 200,
    'dec4_dim': 200,
    'dec8_dim': 200,
    'dec16_dim': 200,
    'n_layers': 2,
    'discrim_rnn_dim': 128,
    'discrim_num_layers': 1,
    'cuda': True,
    'highest': 8,
}

train_data = torch.tensor(pickle.load(open('data/Xtr_role.p', 'rb')), dtype=torch.float64).transpose(0, 1)[:, :-1, :]
test_data = torch.tensor(pickle.load(open('data/Xte_role.p', 'rb')), dtype=torch.float64).transpose(0, 1)[:, :-1, :]

policy_net_old = NAOMI(params)
policy_net_new = NAOMI(params)

save_path_old = "saved/NAOMI_basketball_003/"
save_path_new = "saved/NAOMI_basketball_005/"

policy_state_dict_old = torch.load(save_path_old + 'model/policy_step' + str(8) + '_training.pth',
                                   map_location=torch.device('cpu'))
policy_net_old.load_state_dict(policy_state_dict_old)

policy_state_dict_new = torch.load(save_path_new + 'model/policy_step' + str(8) + '_training.pth',
                                   map_location=torch.device('cpu'))
policy_net_new.load_state_dict(policy_state_dict_new)

# def check_similar(policy_net_o, policy_net_n):
#
#     for p1, p2 in zip(policy_net_o.parameters(), policy_net_n.parameters()):
#         if p1.data.ne(p2.data).sum() > 0:
#             print(False)
#         else:
#             print(True)
#
#
# check_similar(policy_net_old, policy_net_new)


ave_change_step_size_old = []
ave_len_old = []
ave_out_of_bound_old = []

ave_change_step_size_new = []
ave_len_new = []
ave_out_of_bound_new = []

ave_change_step_size_exp = []
ave_len_exp = []
ave_out_of_bound_exp = []

num_missings = np.arange(37, 48)
print(num_missings)

i_iter = 250
for num_missing in num_missings:
    temp_ave_change_step_size_old = []
    temp_ave_len_old = []
    temp_ave_out_of_bound_old = []

    temp_ave_change_step_size_new = []
    temp_ave_len_new = []
    temp_ave_out_of_bound_new = []

    temp_ave_change_step_size_exp= []
    temp_ave_len_exp = []
    temp_ave_out_of_bound_exp = []

    for i in range(i_iter):
        exp_states, exp_actions, ground_truth, states_old, actions_old, samples_old, \
            mod_stats_old, states_new, actions_new, samples_new, mod_stats_new, exp_stats = \
            collect_samples_interpolate_compare(policy_net_old, policy_net_new, test_data, use_gpu, i_iter=i,
                                                task='basketball', name_old='old/test_iter', name_new='new/test_iter',
                                                draw=True, stats=True, num_missing=num_missing)

        temp_ave_len_old.append(mod_stats_old['ave_length'])
        temp_ave_change_step_size_old.append(mod_stats_old['ave_change_step_size'])
        temp_ave_out_of_bound_old.append(mod_stats_old['ave_out_of_bound'])

        temp_ave_len_new.append(mod_stats_new['ave_length'])
        temp_ave_change_step_size_new.append(mod_stats_new['ave_change_step_size'])
        temp_ave_out_of_bound_new.append(mod_stats_new['ave_out_of_bound'])

        temp_ave_len_exp.append(exp_stats['ave_length'])
        temp_ave_change_step_size_exp.append(exp_stats['ave_change_step_size'])
        temp_ave_out_of_bound_exp.append(exp_stats['ave_out_of_bound'])

    ave_len_old.append(temp_ave_len_old)
    ave_change_step_size_old.append(temp_ave_change_step_size_old)
    ave_out_of_bound_old.append(temp_ave_out_of_bound_old)

    ave_len_new.append(temp_ave_len_new)
    ave_change_step_size_new.append(temp_ave_change_step_size_new)
    ave_out_of_bound_new.append(temp_ave_out_of_bound_new)

    ave_len_exp.append(temp_ave_len_exp)
    ave_change_step_size_exp.append(temp_ave_change_step_size_exp)
    ave_out_of_bound_exp.append(temp_ave_out_of_bound_exp)

ave_len_old = np.array(ave_len_old)
ave_change_step_size_old = np.array(ave_change_step_size_old)
ave_out_of_bound_old = np.array(ave_out_of_bound_old)

ave_len_new = np.array(ave_len_new)
ave_change_step_size_new = np.array(ave_change_step_size_new)
ave_out_of_bound_new = np.array(ave_out_of_bound_new)

ave_len_exp = np.array(ave_len_exp)
ave_change_step_size_exp = np.array(ave_change_step_size_exp)
ave_out_of_bound_exp = np.array(ave_out_of_bound_exp)

np.savetxt("ave_len_old.out", ave_len_old)
np.savetxt("ave_change_step_size_old.out", ave_change_step_size_old)
np.savetxt("ave_out_of_bound_old.out", ave_out_of_bound_old)

np.savetxt("ave_len_new.out", ave_len_new)
np.savetxt("ave_change_step_size_new.out", ave_change_step_size_new)
np.savetxt("ave_out_of_bound_new.out", ave_out_of_bound_new)

np.savetxt("ave_len_exp.out", ave_len_exp)
np.savetxt("ave_change_step_size_exp.out", ave_change_step_size_exp)
np.savetxt("ave_out_of_bound_exp.out", ave_out_of_bound_exp)
