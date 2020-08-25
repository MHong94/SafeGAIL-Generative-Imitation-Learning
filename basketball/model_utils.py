import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from helpers import unnormalize, normalize


######################################################################
########################## MISCELLANEOUS #############################
######################################################################


def one_hot_encode(inds, N):
    # inds should be a torch.Tensor, not a Variable
    dims = [inds.size(i) for i in range(len(inds.size()))]
    inds = inds.unsqueeze(-1).cpu().long()
    dims.append(N)
    ret = torch.zeros(dims)
    ret.scatter_(-1, inds, 1)
    return ret


def logsumexp(x, axis=None):
    x_max = torch.max(x, axis, keepdim=True)[0]  # torch.max() returns a tuple
    ret = torch.log(torch.sum(torch.exp(x - x_max), axis, keepdim=True)) + x_max
    return ret


######################################################################
############################ SAMPLING ################################
######################################################################


def sample_gumbel(logits, tau=1, eps=1e-20):
    u = torch.zeros(logits.size()).uniform_()
    u = Variable(u)
    if logits.is_cuda:
        u = u.cuda()
    g = -torch.log(-torch.log(u + eps) + eps)
    y = (g + logits) / tau
    return F.softmax(y)


def reparam_sample_gauss(mean, std):
    eps = torch.DoubleTensor(std.size()).normal_()
    eps = Variable(eps)
    if mean.is_cuda:
        eps = eps.cuda()
    return eps.mul(std).add_(mean)


def sample_gmm(mean, std, coeff):
    k = coeff.size(-1)
    if k == 1:
        return reparam_sample_gauss(mean, std)

    mean = mean.view(mean.size(0), -1, k)
    std = std.view(std.size(0), -1, k)
    index = torch.multinomial(coeff, 1).squeeze()

    # TODO: replace with torch.gather or torch.index_select
    comp_mean = Variable(torch.zeros(mean.size()[:-1]))
    comp_std = Variable(torch.zeros(std.size()[:-1]))
    if mean.is_cuda:
        comp_mean = comp_mean.cuda()
        comp_std = comp_std.cuda()
    for i in range(index.size(0)):
        comp_mean[i, :] = mean.data[i, :, index.data[i]]
        comp_std[i, :] = std.data[i, :, index.data[i]]

    return reparam_sample_gauss(comp_mean, comp_std), index


def sample_multinomial(probs):
    inds = torch.multinomial(probs, 1).data.cpu().long().squeeze()
    ret = one_hot_encode(inds, probs.size(-1))
    if probs.is_cuda:
        ret = ret.cuda()
    return ret


######################################################################
######################### KL DIVERGENCE ##############################
######################################################################


def kld_gauss(mean_1, std_1, mean_2, std_2):
    kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
                   (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
                   std_2.pow(2) - 1)
    return 0.5 * torch.sum(kld_element)


def kld_categorical(logits_1, logits_2):
    kld_element = torch.exp(logits_1) * (logits_1 - logits_2)
    return torch.sum(kld_element)


######################################################################
###################### NEGATIVE LOG-LIKELIHOOD #######################
######################################################################


def nll_gauss(mean, std, x):
    pi = Variable(torch.DoubleTensor([np.pi]))
    if mean.is_cuda:
        pi = pi.cuda()

    # x = x[:,:10]
    #     if x.is_cuda:
    #         x = updated_actions_safe_set(x).cuda()
    #     else:
    #         x = updated_actions_safe_set(x)

    nll_element = (x - mean).pow(2) / std.pow(2) + 2 * torch.log(std) + torch.log(2 * pi)

    return 0.5 * torch.sum(nll_element)


def nll_gmm(mean, std, coeff, x):
    # mean: (batch, x_dim*k)
    # std: (batch, x_dim*k)
    # coeff: (batch, k)
    # x: (batch, x_dim)

    k = coeff.size(-1)
    if k == 1:
        return nll_gauss(mean, std, x)

    pi = Variable(torch.DoubleTensor([np.pi]))
    if mean.is_cuda:
        pi = pi.cuda()
    mean = mean.view(mean.size(0), -1, k)
    std = std.view(std.size(0), -1, k)

    nll_each = (x.unsqueeze(-1) - mean).pow(2) / std.pow(2) + 2 * torch.log(std) + torch.log(2 * pi)
    nll_component = -0.5 * torch.sum(nll_each, 1)
    terms = torch.log(coeff) + nll_component

    return -torch.sum(logsumexp(terms, axis=1))


######################################################################
###################### METHODS FOR LOG-VARIANCE ######################
######################################################################


def sample_gauss_logvar(mean, logvar):
    eps = torch.DoubleTensor(mean.size()).normal_()
    eps = Variable(eps)
    if mean.is_cuda:
        eps = eps.cuda()
    return eps.mul(torch.exp(logvar / 2)).add_(mean)


def kld_gauss_logvar(mean_1, logvar_1, mean_2, logvar_2):
    kld_element = (logvar_2 - logvar_1 +
                   (torch.exp(logvar_1) + (mean_1 - mean_2).pow(2)) /
                   torch.exp(logvar_2) - 1)
    return 0.5 * torch.sum(kld_element)


def nll_gauss_logvar(mean, logvar, x):
    pi = Variable(torch.DoubleTensor([np.pi]))
    if mean.is_cuda:
        pi = pi.cuda()
    nll_element = (x - mean).pow(2) / torch.exp(logvar) + logvar + torch.log(2 * pi)

    return 0.5 * torch.sum(nll_element)


# NEW ADDED

def updated_actions(x):
    #     safe_1 = torch.Tensor([47, 0])
    #     safe_2 = torch.Tensor([47, 50])

    safe = torch.Tensor([47, 25])
    covariance_matrix = [[2, 0], [0, 1]]

    unnormalized_x = torch.from_numpy(unnormalize(x.cpu().numpy(), 'basketball'))

    m_distance = mahalanobis_distance(unnormalized_x, covariance_matrix)

    min_max_transformed = m_distance / 5.5 + 0.01

    square = min_max_transformed * min_max_transformed

    regularizer = 0.001 * torch.exp(min_max_transformed) / square

    regularizer_plus_one = regularizer + 1

    one_by_lambda = 1 / regularizer_plus_one

    lambda_by_lambda = regularizer / regularizer_plus_one

    output = one_by_lambda * x.cpu() + lambda_by_lambda * normalize(safe.repeat(1, 5), 'basketball')

    #     output = normalize(output, 'basketball')
    return output


def get_weighted_safe_point(m1, m2, safe_1, safe_2):
    d1 = m1 / (m1 + m2)
    d2 = m2 / (m1 + m2)
    final = d1 * safe_1.repeat(5) + d2 * safe_2.repeat(5)

    return final


def mahalanobis_distance_safe_set(x, safe, covariance_matrix):
    batch_size = x.shape[0]
    # x = x.reshape(-1, 5, 2)

    t = x - safe

    t = t.reshape(-1, 5, 2)

    covariance_matrix = torch.Tensor(covariance_matrix * batch_size).reshape(-1, 2, 2)
    inverse_covariance_matrix = torch.inverse(covariance_matrix)
    mid = torch.bmm(t, inverse_covariance_matrix)
    before_final = torch.matmul(mid.unsqueeze(2), torch.transpose(t.unsqueeze(2), 2, 3))

    before_final = before_final.squeeze().reshape(batch_size, 5, 1)

    final = torch.cat((before_final, before_final), 2).reshape(batch_size, 1, 10).squeeze()
    return torch.sqrt(final)


def mahalanobis_distance(x, safe=torch.Tensor([47, 25]), covariance_matrix=[[2, 0], [0, 1]]):
    batch_size = x.shape[0]
    x = x.reshape(-1, 5, 2)

    safe = safe.unsqueeze(0)

    t = x - safe

    covariance_matrix = torch.Tensor(covariance_matrix * batch_size).reshape(-1, 2, 2)
    inverse_covariance_matrix = torch.inverse(covariance_matrix)
    mid = torch.bmm(t, inverse_covariance_matrix)
    before_final = torch.matmul(mid.unsqueeze(2), torch.transpose(t.unsqueeze(2), 2, 3))

    before_final = before_final.squeeze().reshape(batch_size, 5, 1)

    final = torch.cat((before_final, before_final), 2).reshape(batch_size, 1, 10).squeeze()
    return torch.sqrt(final)
