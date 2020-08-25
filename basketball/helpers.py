import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.transform import resize
from torch import nn
from torch.autograd import Variable

matplotlib.use('Agg')

use_gpu = torch.cuda.is_available()


# training function used in pre training
def run_epoch(train, model, exp_data, clip, optimizer=None, batch_size=64, num_missing=None, teacher_forcing=True):
    losses = []
    inds = np.random.permutation(exp_data.shape[0])

    i = 0
    while i + batch_size <= exp_data.shape[0]:
        ind = torch.from_numpy(inds[i:i + batch_size]).long()
        i += batch_size
        data = exp_data[ind]

        if use_gpu:
            data = data.cuda()

        # change (batch, time, x) to (time, batch, x)
        data = Variable(data.squeeze().transpose(0, 1))

        safe_in = data[0][0][10:]

        data = data[:, :, :10]
        ground_truth = data.clone()

        if num_missing is None:
            # num_missing = np.random.randint(data.shape[0] * 18 // 20, data.shape[0])
            num_missing = np.random.randint(data.shape[0] * 4 // 5, data.shape[0])
            # num_missing = 40
        missing_list = torch.from_numpy(
            np.random.choice(np.arange(2, data.shape[0]), num_missing - 1, replace=False)).long()

        # num_missing = 44
        # missing_list = torch.arange(5, 49)

        data[missing_list] = 0.0
        has_value = Variable(torch.ones(data.shape[0], data.shape[1], 1))
        if use_gpu:
            has_value = has_value.cuda()
        has_value[missing_list] = 0.0
        data = torch.cat([has_value, data], 2)
        seq_len = data.shape[0]

        if teacher_forcing:
            batch_loss = model(data, ground_truth, safe_in)
        else:
            data_list = []
            for j in range(seq_len):
                data_list.append(data[j:j + 1])
            samples = model.sample(data_list)
            batch_loss = torch.mean((ground_truth - samples).pow(2))

        if train:
            optimizer.zero_grad()
            total_loss = batch_loss
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        losses.append(batch_loss.data.cpu().numpy())

    return np.mean(losses)


def ones(*shape):
    return torch.ones(*shape).cuda() if use_gpu else torch.ones(*shape)


def zeros(*shape):
    return torch.zeros(*shape).cuda() if use_gpu else torch.zeros(*shape)


# train and pretrain discriminator
def update_discrim(discrim_net, optimizer_discrim, discrim_criterion, exp_states, exp_actions, w_exp_states,
                   w_exp_actions, states, actions, i_iter, dis_times, use_gpu, train=True):
    if use_gpu:
        exp_states, exp_actions, states, actions = exp_states.cuda(), exp_actions.cuda(), states.cuda(), actions.cuda()

    """update discriminator"""
    g_o_ave = 0.0
    e_o_ave = 0.0
    for _ in range(int(dis_times)):
        g_o = discrim_net(Variable(states), Variable(actions))
        e_o = discrim_net(Variable(w_exp_states), Variable(w_exp_actions))

        g_o_ave += g_o.cpu().data.mean()
        e_o_ave += e_o.cpu().data.mean()

        if train:
            optimizer_discrim.zero_grad()
            discrim_loss = discrim_criterion(g_o, Variable(zeros((g_o.shape[0], g_o.shape[1], 1)))) + \
                           discrim_criterion(e_o, Variable(ones((e_o.shape[0], e_o.shape[1], 1))))
            discrim_loss.backward()
            optimizer_discrim.step()

    if dis_times > 0:
        return g_o_ave / dis_times, e_o_ave / dis_times


# train policy network
def update_policy(policy_net, optimizer_policy, discrim_net, discrim_criterion,
                  states_var, actions_var, i_iter, use_gpu):
    optimizer_policy.zero_grad()
    g_o = discrim_net(states_var, actions_var)
    policy_loss = discrim_criterion(g_o, Variable(ones((g_o.shape[0], g_o.shape[1], 1))))
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm(policy_net.parameters(), 10)
    optimizer_policy.step()


# sample trajectories used in GAN training
def collect_samples_interpolate(policy_net, expert_data, weighted_expert_data, use_gpu, i_iter, task, size=64,
                                name="sampling_inter",
                                draw=False, stats=False, num_missing=None):
    exp_ind = torch.from_numpy(np.random.choice(expert_data.shape[0], size)).long()
    data = expert_data[exp_ind].clone()
    weighted_data = weighted_expert_data[exp_ind].clone()
    seq_len = data.shape[1]

    if use_gpu:
        data = data.cuda()

    data = Variable(data.squeeze().transpose(0, 1))

    weighted_data = weighted_data.squeeze().transpose(0, 1)
    ground_truth = data.clone()

    # if num_missing is None:
    #     num_missing = np.random.randint(seq_len * 4 // 5, seq_len)
    #     num_missing = np.random.randint(seq_len * 18 // 20, seq_len)
    #     num_missing = 40

    missing_list = torch.from_numpy(np.random.choice(np.arange(2, seq_len), num_missing - 1, replace=False)).long()
    sorted_missing_list, _ = torch.sort(missing_list)

    # num_missing = 44
    # missing_list = torch.arange(5, 49)
    # sorted_missing_list, _ = torch.sort(missing_list)

    print("num_missing:", num_missing)
    print("collect sample:", sorted_missing_list)
    data[missing_list] = 0.0
    has_value = Variable(torch.ones(seq_len, size, 1))
    if use_gpu:
        has_value = has_value.cuda()
    has_value[missing_list] = 0.0
    data = torch.cat([has_value, data], 2)

    data_list = []
    for i in range(seq_len):
        data_list.append(data[i:i + 1])
    samples = policy_net.sample(data_list)

    states = samples[:-1, :, :]
    actions = samples[1:, :, :]

    exp_states = ground_truth[:-1, :, :]
    exp_actions = ground_truth[1:, :, :]

    w_exp_states = weighted_data[:-1, :, :]
    w_exp_actions = weighted_data[1:, :, :]

    mod_stats = draw_and_stats(samples.data, name + '_' + str(num_missing), i_iter, task, draw=draw,
                               compute_stats=stats, missing_list=missing_list)
    exp_stats = draw_and_stats(ground_truth.data, name + '_expert' + '_' + str(num_missing), i_iter, task, draw=draw,
                               compute_stats=stats, missing_list=missing_list)

    return exp_states.data, exp_actions.data, w_exp_states.data, w_exp_actions.data, ground_truth.data, \
           states, actions, samples.data, mod_stats, exp_stats


# transfer a 1-d vector into a string
def to_string(x):
    ret = ""
    for i in range(x.shape[0]):
        ret += "{:.3f} ".format(x[i])
    return ret


def ave_player_distance(states):
    # states: numpy (seq_lenth, batch, 10)
    count = 0
    ret = np.zeros(states.shape)
    for i in range(5):
        for j in range(i + 1, 5):
            ret[:, :, count] = np.sqrt(np.square(states[:, :, 2 * i] - states[:, :, 2 * j]) +
                                       np.square(states[:, :, 2 * i + 1] - states[:, :, 2 * j + 1]))
            count += 1
    return ret


# draw and compute statistics
def draw_and_stats(model_states, name, i_iter, task, compute_stats=True, draw=True, missing_list=None):
    stats = {}
    if compute_stats:
        model_actions = model_states[1:, :, :] - model_states[:-1, :, :]

        val_data = model_states.cpu().numpy()
        val_actions = model_actions.cpu().numpy()

        step_size = np.sqrt(np.square(val_actions[:, :, ::2]) + np.square(val_actions[:, :, 1::2]))
        change_of_step_size = np.abs(step_size[1:, :, :] - step_size[:-1, :, :])
        stats['ave_change_step_size'] = np.mean(change_of_step_size)
        val_seqlength = np.sum(np.sqrt(np.square(val_actions[:, :, ::2]) + np.square(val_actions[:, :, 1::2])), axis=0)
        stats['ave_length'] = np.mean(val_seqlength)  # when sum along axis 0, axis 1 becomes axis 0
        stats['ave_out_of_bound'] = np.mean((val_data < -0.51) + (val_data > 0.51))
        # stats['ave_player_distance'] = np.mean(ave_player_distance(val_data))
        # stats['diff_max_min'] = np.mean(np.max(val_seqlength, axis=1) - np.min(val_seqlength, axis=1))

    if draw:
        print("Drawing")
        draw_data = model_states.cpu().numpy()[:, 0, :]
        draw_data = unnormalize(draw_data, task)

        colormap = ['b', 'r', 'g', 'm', 'y']
        plot_sequence(draw_data, task, colormap=colormap,
                      save_name="imgs/interpolation/{}_{}".format(name, i_iter), missing_list=missing_list)

    return stats


def unnormalize(x, task):
    dim = x.shape[-1]

    if task == 'basketball':
        NORMALIZE = [94, 50] * int(dim / 2)
        SHIFT = [25] * dim
        return np.multiply(x, NORMALIZE) + SHIFT
    else:
        NORMALIZE = [128, 128] * int(dim / 2)
        SHIFT = [1] * dim
        return np.multiply(x + SHIFT, NORMALIZE)


def normalize(x, task):
    x = x.numpy()
    dim = x.shape[-1]

    if task == 'basketball':
        UNNORMALIZE = [94, 50] * int(dim / 2)
        SHIFT = [25] * dim
        return torch.from_numpy(np.divide(x - SHIFT, UNNORMALIZE))
    else:
        UNNORMALIZE = [128, 128] * int(dim / 2)
        SHIFT = [1] * dim
        return torch.from_numpy(np.divide(x, UNNORMALIZE) - SHIFT)


def _set_figax(task):
    fig = plt.figure(figsize=(5, 5))

    if task == 'basketball':
        img = plt.imread('data/court.png')
        #         img = resize(img, (500, 940, 3))
        img = resize(img, (500, 470, 4))
        ax = fig.add_subplot(111)
        ax.imshow(img)

        # show just the left half-court
        #         ax.set_xlim([-50, 550])
        #         ax.set_ylim([-50, 550])

        ax.set_xlim([0, 470])
        ax.set_ylim([0, 500])

    else:
        img = plt.imread('data/world.jpg')
        img = resize(img, (256, 256, 3))

        ax = fig.add_subplot(111)
        ax.imshow(img)

        ax.set_xlim([-50, 300])
        ax.set_ylim([-50, 300])

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    return fig, ax


def plot_sequence(seq, task, colormap, save_name='', missing_list=None):
    n_players = int(len(seq[0]) / 2)

    while len(colormap) < n_players:
        colormap += 'b'

    fig, ax = _set_figax(task)

    if task == 'basketball':
        SCALE = 10
    else:
        SCALE = 1

    for k in range(n_players):
        x = seq[:, (2 * k)]
        y = seq[:, (2 * k + 1)]
        color = colormap[k]
        ax.plot(SCALE * x, SCALE * y, color=color, linewidth=3, alpha=0.7)
        ax.plot(SCALE * x, SCALE * y, 'o', color=color, markersize=8, alpha=0.5)

    # starting positions
    x = seq[0, ::2]
    y = seq[0, 1::2]
    ax.plot(SCALE * x, SCALE * y, 'o', color='black', markersize=12)

    if missing_list is not None:
        missing_list = missing_list.numpy()
        for i in range(seq.shape[0]):
            if i not in missing_list:
                x = seq[i, ::2]
                y = seq[i, 1::2]
                ax.plot(SCALE * x, SCALE * y, 'o', color='black', markersize=8)

    plt.tight_layout(pad=0)

    if len(save_name) > 0:
        plt.savefig(save_name + '.png')
    else:
        plt.show()

    plt.close(fig)


def collect_samples_interpolate_compare(policy_net_old, policy_net_new, expert_data, use_gpu, i_iter, task, size=64,
                                        name_old="sampling_inter", name_new="sampling_iter",
                                        draw=False, stats=False, num_missing=None):
    exp_ind = torch.from_numpy(np.random.choice(expert_data.shape[0], size)).long()

    data = expert_data[exp_ind].clone().type(torch.FloatTensor)

    seq_len = data.shape[1]
    # print(data.shape, seq_len)
    if use_gpu:
        data = data.cuda()
    data = data.squeeze().transpose(0, 1)
    ground_truth = data.clone()

    if num_missing is None:
        num_missing = np.random.randint(seq_len * 4 // 5, seq_len)

    missing_list = torch.from_numpy(np.random.choice(np.arange(1, seq_len), num_missing, replace=False)).long()
    sorted_missing_list, _ = torch.sort(missing_list)
    print("num_missing:", num_missing)

    # num_missing = 44
    # missing_list = torch.arange(6, 49)
    # sorted_missing_list, _ = torch.sort(missing_list)

    print("collect sample:", sorted_missing_list)
    data[missing_list] = 0.0

    has_value = torch.ones(seq_len, size, 1)

    if use_gpu:
        has_value = has_value.cuda()
    has_value[missing_list] = 0.0
    data = torch.cat([has_value, data], 2)
    data_list_old = []
    data_list_new = []

    for i in range(seq_len):
        data_list_old.append(data[i:i + 1])
        data_list_new.append(data[i:i + 1])

    # for p1, p2 in zip(policy_net_old.parameters(), policy_net_new.parameters()):
    #     if p1.data.ne(p2.data).sum() > 0:
    #         print(False)
    #     else:
    #         print(True)

    samples_old = policy_net_old.sample(data_list_old)
    samples_new = policy_net_new.sample(data_list_new)

    states_old = samples_old[:-1, :, :]
    actions_old = samples_old[1:, :, :]

    states_new = samples_new[:-1, :, :]
    actions_new = samples_new[1:, :, :]

    exp_states = ground_truth[:-1, :, :]
    exp_actions = ground_truth[1:, :, :]

    # print("sum:", torch.sum(samples_old - samples_new))

    mod_stats_old = draw_and_stats(samples_old.data, name_old + '_' + str(num_missing), i_iter, task, draw=draw,
                                   compute_stats=stats, missing_list=missing_list)
    mod_stats_new = draw_and_stats(samples_new.data, name_new + '_' + str(num_missing), i_iter, task, draw=draw,
                                   compute_stats=stats, missing_list=missing_list)
    name = ' '
    exp_stats = draw_and_stats(ground_truth.data, name + 'expert' + '_' + str(num_missing), i_iter, task, draw=draw,
                               compute_stats=stats, missing_list=missing_list)

    return exp_states.data, exp_actions.data, ground_truth.data, states_old, actions_old, samples_old.data, \
           mod_stats_old, states_new, actions_new, samples_new.data, mod_stats_new, exp_stats


# NEW ADDED
def updated_actions_safe_set(x, safe_set):
    seq_len = x.shape[0]
    batch_size = x.shape[1]

    unnormalized_x = normalize(x, 'basketball')

    covariance_matrix = [[2., 0.], [0., 1.]]

    m_distances = torch.empty((1, seq_len, batch_size, 10)).to('cuda')
    for safe in safe_set:
        m_distances = torch.cat((m_distances,
                                 mahanalobis_distance_seq_batch_safe_set(unnormalized_x, safe, covariance_matrix)
                                 .reshape(-1, seq_len, batch_size, 10)), dim=0)
    m_distances = m_distances[1:]

    m_distances_odd = m_distances[:, :, :, ::2]
    m_distances_sum = torch.sum(m_distances_odd, axis=0)

    m_distances_div = m_distances_odd / m_distances_sum

    safe_set_new = torch.matmul(torch.transpose(m_distances_div, 0, 3), safe_set).reshape(seq_len, batch_size, 5, 2)

    distances = mahanalobis_distance_seq_batch_safe_set(unnormalized_x, safe_set_new, covariance_matrix)

    transormed = 13 - (distances / 4.5 + 0.01)
    square = transormed * transormed

    regularizer = 0.001 * torch.exp(transormed) / square
    regularizer_plus_one = regularizer + 1

    one_by_lambda = 1 / regularizer_plus_one

    lambda_by_lambda = regularizer / regularizer_plus_one

    output = one_by_lambda * unnormalized_x.squeeze() + \
             lambda_by_lambda * safe_set_new.view(seq_len, batch_size, 10).squeeze()

    output = normalize(output, 'basketball')

    return output, distances


def mahanalobis_distance_seq_batch_safe_set(x, safe, covariance_matrix):
    seq_len = x.shape[0]
    batch_size = x.shape[1]

    x = x.reshape(seq_len, batch_size, 5, 2)

    t = x - safe

    covariance_matrix = torch.Tensor(covariance_matrix * (seq_len * batch_size)).reshape(seq_len, batch_size, 2, 2)
    inverse_covariance_matrix = torch.inverse(covariance_matrix).to('cuda')

    mid = torch.bmm(t.view(-1, 5, 2), inverse_covariance_matrix.view(-1, 2, 2)).reshape(seq_len, batch_size, 5, 2)

    before_final = torch.matmul(mid.unsqueeze(3), torch.transpose(t.unsqueeze(3), 3, 4))

    before_final = before_final.squeeze().reshape(seq_len, batch_size, 5, 1)

    final = torch.cat((before_final, before_final), 3).reshape(seq_len, batch_size, 1, 10).squeeze()
    return torch.sqrt(final)


def get_closest_set(d, temp_per_safe_set, distances):
    min_distance, min_index = torch.min(distances, 0)
    final_actions = torch.zeros([48, 10])

    for i in range(min_index.shape[0]):
        for j in range(min_index.shape[1]):
            if min_index[i][j] == 0:
                final_actions[i][j] == temp_per_safe_set[0][i][j]
            else:
                final_actions[i][j] == temp_per_safe_set[1][i][j]

    return final_actions


def updated_actions_for_conditional(x):
    seq_len = x.shape[0]
    batch_size = x.shape[1]

    output = torch.empty(seq_len, 1, 26).to('cuda')

    for i in range(batch_size):
        temp_data = x[:, i, :]
        safe_mid = temp_data[:, 10:][0]
        safe = safe_mid.reshape(2, 4, 2)
        d = temp_data[:, :10]
        temp_per_safe_set = torch.empty((1, seq_len, 10)).to('cuda')
        distances = torch.empty((1, seq_len, 10)).to('cuda')
        for j in range(safe.shape[0]):
            mid, distance = updated_actions_safe_set(d.unsqueeze(1), safe[j])
            temp_per_safe_set = torch.cat([temp_per_safe_set, mid.to('cuda').unsqueeze(0)], 0)
            distances = torch.cat([distances, distance.unsqueeze(0)], 0)

        temp_per_safe_set = temp_per_safe_set[1:]
        distances = distances[1:]

        closest_safe_set_actions = get_closest_set(d, temp_per_safe_set, distances)

        for_output = torch.cat([closest_safe_set_actions.to('cuda'), safe_mid.repeat(seq_len, 1)], 1)

        output = torch.cat([output, for_output.unsqueeze(1)], 1)

    output = output[:, 1:, :]

    return output
