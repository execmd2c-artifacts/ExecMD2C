# ============================================================
# ground_truth.py - HL-VAE Core Model Components
# Source:
#   Miscellaneous/HL-VAE-main/HLVAE.py
#   Miscellaneous/HL-VAE-main/HL_VAE/utils.py
#   Miscellaneous/HL-VAE-main/HL_VAE/loglik.py
#
# Contains ONLY model architecture components and direct dependencies.
# No training loops, datasets, evaluation loops, checkpoint I/O, or CLI code.
# ============================================================

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td


# --- [Original file: HL_VAE/utils.py] ---
def one_hot(seq_batch, depth):
    # seq_batch.size() should be [seq,batch] or [batch,]
    # return size() would be [seq,batch,depth] or [batch,depth]
    out = torch.zeros(seq_batch.size() + torch.Size([depth])).to(seq_batch.device.type)
    dim = len(seq_batch.size())
    index = seq_batch.view(seq_batch.size() + torch.Size([1]))
    return out.scatter_(dim, index, 1)


def sequence_mask(lengths, maxlen, dtype=torch.bool):
    if maxlen is None:
        maxlen = lengths.max()
    # mask = ~(torch.ones((len(lengths), maxlen)).cumsum(dim=1).t() > lengths).t()
    mask = (torch.ones((lengths.shape[0], lengths.shape[1], maxlen), device=lengths.device).cumsum(dim=2)) <= lengths[:, :, None]
    mask = mask.type(dtype).to(lengths.device)
    return mask


def batch_normalization(batch_data_list, miss_list, param_mask, types_info):

    normalization_parameters = [[], []]
    # batch_data_list = torch.cat(batch_data_list, 1)
    normalized_data = torch.zeros_like(batch_data_list)

    for i, tpl in enumerate(types_info['set_of_types']):
        missing_block = miss_list[:, types_info['data_types_indexes'] == i]
        d = batch_data_list[:, types_info['exp_types_indexes'] == i]
        if tpl[0] == 'real':
            observed_data = d * missing_block
            if types_info['conv']:
                ##[HealthMNIST]
                norm_data = observed_data / 255
                normalization_parameters[0] = []
            # if types_info['real_ranges'] == []:
            else:
                data_mean = (observed_data * missing_block).sum(dim=0) / missing_block.sum(dim=0)
                data_var = torch.sum(((observed_data-data_mean) * missing_block)**2,0) / missing_block.sum(dim=0)
                norm_data = observed_data.sub(data_mean[None, :])/torch.sqrt(data_var+1e-5) * missing_block
                normalization_parameters[0] = [data_mean, data_var]
            normalized_data[:, types_info['exp_types_indexes'] == i] = norm_data
        elif tpl[0] == 'count':
            observed_data = d * missing_block
            # Input log of the data
            aux_X = torch.log(observed_data)
            # aux_X[missing_block == 0] = d[missing_block == 0]
            aux_X[missing_block == 0] = 0
            normalized_data[:, types_info['exp_types_indexes'] == i] = aux_X
        elif tpl[0] == 'pos':
            # We transform the log of the data to a gaussian with mean 0 and std 1
            observed_data = d * missing_block
            observed_data_log = torch.log(1.0 + observed_data)
            data_mean_log = (observed_data_log * missing_block).sum(dim=0) / missing_block.sum(dim=0)
            data_var_log = torch.sum(((observed_data_log-data_mean_log) * missing_block)**2, 0) / missing_block.sum(dim=0)
            data_var_log = torch.clamp(data_var_log, 1e-6, 1e20)  # Avoid zero values
            norm_data_log = observed_data_log.sub(data_mean_log[None, :])/torch.sqrt(data_var_log+1e-5) * missing_block
            normalized_data[:, types_info['exp_types_indexes'] == i] = norm_data_log
            normalization_parameters[1] = [data_mean_log, data_var_log]
        elif tpl[0] in ['cat','ordinal']:
            try:
                missing_block = missing_block.repeat(1, int(tpl[1])).reshape(missing_block.shape[0],  int(tpl[1]), -1).\
                permute(0, 2, 1).reshape(missing_block.shape[0], -1)
            except:
                print('mine')
            normalized_data[:, types_info['exp_types_indexes'] == i] = d * missing_block
        else:
            normalized_data[:, types_info['exp_types_indexes'] == i] = d * missing_block

    return normalized_data, normalization_parameters


# --- [Original file: HL_VAE/loglik.py] ---
def loglik_real(batch_data, list_type, theta, normalization_params, extra_params=None):
    output = dict()
    epsilon = 3e-4
    min_log_vy = torch.Tensor([-8.0]).to(theta.device)

    # Data outputs
    data, missing_mask = batch_data
    missing_mask = missing_mask.float()

    if normalization_params != []:
        data_mean, data_var = normalization_params
        data_var = torch.clamp(data_var, epsilon, np.inf)
    else:
        data_mean = torch.tensor(0.)
        data_var = torch.tensor(1.)

    indx = np.r_[0:data.shape[1]]

    if extra_params == None:
        est_mean, est_var = theta[:, indx], theta[:, indx + len(indx)]
        est_log_vy = min_log_vy + F.softplus(est_var - min_log_vy)
        est_var = torch.exp(est_log_vy)
    else:
        est_mean = theta[:, indx]
        est_log_vy = min_log_vy + F.softplus(extra_params - min_log_vy)
        est_var = torch.exp(est_log_vy)

    # Affine transformation of the parameters
    est_mean = torch.sqrt(data_var) * est_mean + data_mean
    est_var = data_var * est_var
    # Compute loglik
    log_p_x = -0.5 * torch.pow(data - est_mean, 2) / est_var - 0.5 * np.log(2 * np.pi) - 0.5 * torch.log(est_var)
    normal = td.Normal(est_mean, torch.sqrt(est_var))

    # Outputs
    output['log_p_x'] = torch.mul(log_p_x, missing_mask)
    output['log_p_x_missing'] = torch.mul(log_p_x, 1.0 - missing_mask)
    if est_var.shape == est_mean.shape:
        output['params'] = [est_mean, est_var]
    else:
        output['params'] = est_mean
    output['samples'] = normal.rsample()

    return output


def loglik_count(batch_data, list_type, theta, normalization_params, extra_params=None):
    output = dict()
    epsilon = 1e-6

    # Data outputs
    data, missing_mask = batch_data
    data = data.reshape(data.shape[0], -1)
    # data = torch.cat(data, 1)
    missing_mask = missing_mask.float()

    # est_lambda = theta[:, :, 0]
    est_lambda = theta.reshape(theta.shape[0], -1)
    est_lambda = torch.clamp(torch.nn.Softplus()(est_lambda), epsilon, 1e20)

    poisson = td.Poisson(est_lambda)
    log_p_x = poisson.log_prob(data)  # .sum(1)

    output['log_p_x'] = torch.mul(log_p_x, missing_mask)
    output['log_p_x_missing'] = torch.mul(log_p_x, 1.0 - missing_mask)
    output['params'] = est_lambda
    output['samples'] = poisson.sample()

    return output


def loglik_cat(batch_data, list_type, theta, normalization_params, extra_params=None):
    output = dict()

    # Data outputs
    data, missing_mask = batch_data
    missing_mask = missing_mask.float()

    log_pi = theta.reshape((theta.shape[0], missing_mask.shape[1], -1))

    # Compute loglik
    log_pi = torch.sub(log_pi, torch.logsumexp(log_pi, 2).reshape(data.shape[0], -1, 1))
    log_p_x = torch.sum(data.reshape((log_pi.shape[0], missing_mask.shape[1], -1)) * F.log_softmax(log_pi, 2), -1)

    output['log_p_x'] = torch.mul(log_p_x.reshape(log_p_x.shape[0], -1), missing_mask)
    output['log_p_x_missing'] = torch.mul(log_p_x.reshape(log_p_x.shape[0], -1), 1.0 - missing_mask)
    output['params'] = log_pi
    try:
        output['samples'] = one_hot(td.Categorical(probs=nn.Softmax(1)(log_pi)).sample(),
                                depth=int(list_type[1])).to(torch.float64)
    except:
        print('mine')

    return output


class _LoglikNamespace:
    """Local namespace mirroring the source loglik module."""


loglik = _LoglikNamespace()
loglik.loglik_real = loglik_real
loglik.loglik_count = loglik_count
loglik.loglik_cat = loglik_cat


# --- [Original file: HLVAE.py] ---
class Observation_Count(nn.Module):
    def __init__(self, covariate_dim, y_dim):
        super(Observation_Count, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(covariate_dim, y_dim, 1, requires_grad=True))
        self.bias = torch.nn.Parameter(torch.rand(covariate_dim, 1, requires_grad=True))

        torch.nn.init.normal_(self.weight, mean=0.0, std=.05)
        torch.nn.init.normal_(self.bias, mean=0.0, std=.05)

    def forward(self, gamma, logvar_network):
        theta = torch.add(torch.einsum("bdy, dya->bda", gamma, self.weight), self.bias)
        return theta


class Observation_Real_Pos_Beta(nn.Module):
    def __init__(self, covariate_dim, y_dim, logvar_network):
        super(Observation_Real_Pos_Beta, self).__init__()

        mean_input_dim = y_dim
        if logvar_network:
            log_var_input_dim = y_dim
            self.weight_logvar = torch.nn.Parameter(torch.rand(covariate_dim, log_var_input_dim, 1, requires_grad=True))
            self.bias_logvar = torch.nn.Parameter(torch.rand(covariate_dim, 1, requires_grad=True))
            torch.nn.init.normal_(self.weight_logvar, mean=0.0, std=.05)
            torch.nn.init.normal_(self.bias_logvar, mean=0.0, std=.05)

        self.weight_mean = torch.nn.Parameter(torch.rand(covariate_dim, mean_input_dim, 1, requires_grad=True))
        self.bias_mean = torch.nn.Parameter(torch.rand(covariate_dim, 1, requires_grad=True))
        torch.nn.init.normal_(self.weight_mean, mean=0.0, std=.05)
        torch.nn.init.normal_(self.bias_mean, mean=0.0, std=.05)

    def forward(self, gamma, logvar_network):
        if logvar_network:
            logvar_input = gamma
            theta_logvar = torch.add(torch.einsum("bdy, dya->bda", logvar_input, self.weight_logvar),
                                     self.bias_logvar)
        else:
            theta_logvar = torch.empty(0).to(gamma.device)

        theta_mean = torch.add(torch.einsum("bdy, dya->bda", gamma, self.weight_mean), self.bias_mean)
        return torch.cat([theta_mean, theta_logvar], 1)


class Observation_Cat(nn.Module):
    def __init__(self, covariate_dim, input_dim, nclass):
        super(Observation_Cat, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(covariate_dim, input_dim, nclass-1, requires_grad=True))
        self.bias = torch.nn.Parameter(torch.rand(covariate_dim, nclass-1, requires_grad=True))

        torch.nn.init.normal_(self.weight, mean=0.0, std=.05)
        torch.nn.init.normal_(self.bias, mean=0.0, std=.05)

    def forward(self, gamma, logvar_network):

        theta = torch.add(torch.einsum("bdy, dya->bda", gamma, self.weight), self.bias)
        d = torch.zeros([theta.shape[0], theta.shape[1]], dtype=torch.float64).to(theta.device)
        theta = torch.cat((d.unsqueeze(2), theta), dim=-1)
        return theta


class Observation_Ordinal(nn.Module):
    def __init__(self, covariate_dim, y_dim, nclass):
        super(Observation_Ordinal, self).__init__()
        input_dim = y_dim
        self.weight_region = torch.nn.Parameter(torch.rand(covariate_dim, input_dim, 1, requires_grad=True))
        self.bias_region = torch.nn.Parameter(torch.rand(covariate_dim, 1, requires_grad=True))

        self.weight_thresholds = torch.nn.Parameter(torch.rand(covariate_dim, nclass-1, requires_grad=True))
        # torch.nn.init.ones_(self.weight_thresholds)
        self.weight_thresholds.data.fill_(1.)

        torch.nn.init.normal_(self.weight_region, mean=0.0, std=.05)
        torch.nn.init.normal_(self.bias_region, mean=0.0, std=.05)

    def forward(self, gamma, logvar_network):
        theta = self.weight_thresholds.repeat((gamma.shape[0], 1, 1))

        theta = torch.cat((theta, torch.add(torch.einsum("bdy, dya->bda", gamma, self.weight_region),
                                            self.bias_region)), dim=-1)
        return theta


class Representation_One_Hot(nn.Module):
    def __init__(self, covariate_class_num, nclass):
        super(Representation_One_Hot, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(covariate_class_num, nclass, requires_grad=True))
        self.bias = torch.nn.Parameter(torch.rand(covariate_class_num, requires_grad=True))

        torch.nn.init.normal_(self.weight, mean=0.0, std=.05)
        torch.nn.init.normal_(self.bias, mean=0.0, std=.05)

    def forward(self, input):
        representation = torch.add(torch.einsum("bdc, dc->bd", input, self.weight), self.bias)
        return representation


class HLVAE(nn.Module):
    """

    """

    def __init__(self, dims, types_info, n_variables, vy_init=[1., .5], vy_fixed=False, logvar_network=False, conv=True):
        super(HLVAE, self).__init__()

        [x_dim, h_dim_e, z_dim, h_dim_d, y_dim] = dims
        h_dim_d = [i for i in reversed(h_dim_d)]
        self.z_dim = z_dim
        self.num_dim = n_variables
        self.y_dim = y_dim
        self.logvar_network = logvar_network
        self.tau = 1e-3

        types_list = types_info['types_dict']
        self.types_info = types_info
        self.conv = conv

        ## Encoder Network
        if not self.conv:
            e_lin_layers = nn.ModuleList()
            input_dim = x_dim
            if h_dim_e is not None and h_dim_e != [] and h_dim_e != 0 and h_dim_e != [0]:
                neurons = [input_dim, *h_dim_e]
                for i in range(len(neurons) - 1):
                    e_lin_layers.append(nn.Linear(neurons[i], neurons[i + 1]))
                    torch.nn.init.normal_(e_lin_layers[-1].weight, mean=0.0, std=.05)
                    torch.nn.init.normal_(e_lin_layers[-1].bias, mean=0.0, std=.05)
                    e_lin_layers.append(nn.ReLU())
                input_dim = h_dim_e[-1]

            self.VAE_encoder_common_layers = nn.Sequential(*e_lin_layers)
        else:
            print('Conv HLVAE is being created!')
            self.representation_layer = nn.ModuleList()
            for i, tpl in enumerate(self.types_info['set_of_types']):
                if tpl[0] == 'cat' or tpl[0] == 'ordinal':
                    self.representation_layer.append(
                        Representation_One_Hot(len(self.types_info['data_types_indexes'][self.types_info['data_types_indexes'] == i]),
                                        int(tpl[1])))
            # first convolution layer
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

            # second convolution layer
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

            e_lin_layers = nn.ModuleList()
            input_dim = 32 * 9 * 9
            if h_dim_e is not None and h_dim_e != [] and h_dim_e != 0 and h_dim_e != [0]:
                neurons = [input_dim, *h_dim_e]
                for i in range(len(neurons) - 1):
                    e_lin_layers.append(nn.Linear(neurons[i], neurons[i + 1]))
                    torch.nn.init.normal_(e_lin_layers[-1].weight, mean=0.0, std=.05)
                    torch.nn.init.normal_(e_lin_layers[-1].bias, mean=0.0, std=.05)
                    e_lin_layers.append(nn.ReLU())
                input_dim = h_dim_e[-1]

            self.VAE_encoder_common_layers = nn.Sequential(*e_lin_layers)

        self.mean_layer = nn.ModuleList()
        self.mean_layer.append(nn.Linear(input_dim, z_dim))
        torch.nn.init.normal_(self.mean_layer[0].weight, mean=0.0, std=.05)
        torch.nn.init.normal_(self.mean_layer[0].bias, mean=0.0, std=.05)
        self.mean_layer = nn.Sequential(*self.mean_layer)

        self.log_var_layer = nn.ModuleList()
        self.log_var_layer.append(nn.Linear(input_dim, z_dim))
        torch.nn.init.normal_(self.log_var_layer[0].weight, mean=0.0, std=.05)
        torch.nn.init.normal_(self.log_var_layer[0].bias, mean=0.0, std=.05)
        self.log_var_layer = nn.Sequential(*self.log_var_layer)

        ## Decoder Network
        self.theta_cov_indexes = [0]
        self.cov_block_indexes = [0]
        self.obs_dim = 0
        self.real_dim = 0
        self.pos_dim = 0
        for d, t in enumerate(types_list):
            self.cov_block_indexes.append(self.cov_block_indexes[-1] + t['dim'])
            if t['type'] in ['pos', 'real', 'beta']:
                if t['type'] in ['real']:
                    self.real_dim += t['dim']
                else:
                    self.pos_dim += t['dim']
                self.obs_dim += 2
                self.theta_cov_indexes.append(self.theta_cov_indexes[-1] + 2 * t['dim'])
            elif t['type'] == 'count':
                self.obs_dim += 1
                self.theta_cov_indexes.append(self.theta_cov_indexes[-1] + t['dim'])
            elif t['type'] == 'ordinal':
                self.obs_dim += t['nclass']
            else:
                self.obs_dim += t['nclass'] - 1
                self.theta_cov_indexes.append(self.theta_cov_indexes[-1] + t['dim'] * t['nclass'])

        ########################
        if (not logvar_network):
            min_log_vy = torch.Tensor([-8.0])

            log_vy_init_real = torch.log(vy_init[0] - torch.exp(min_log_vy))
            log_vy_init_pos = torch.log(vy_init[1] - torch.exp(min_log_vy))
            # log variance
            if isinstance(vy_init[0], float):
                self._log_vy_real = nn.Parameter(torch.Tensor(self.real_dim * [log_vy_init_real]))
                self._log_vy_pos = nn.Parameter(torch.Tensor(self.pos_dim * [log_vy_init_pos]))
            else:
                self._log_vy_real = nn.Parameter(torch.Tensor(log_vy_init_real))
                self._log_vy_pos = nn.Parameter(torch.Tensor(log_vy_init_pos))

            if vy_fixed:
                self._log_vy_real.requires_grad_(False)
                self._log_vy_pos.requires_grad_(False)
        else:
            self._log_vy_real = None
            self._log_vy_pos = None
        #########################

        self._disp_param = nn.Parameter(torch.Tensor([1.]))
        self._disp_param.requires_grad_(True)

        self.y_dim_partition = y_dim * np.ones(self.num_dim, dtype=int)
        self.y_dim_output = np.sum(self.y_dim_partition)
        y_input_dim = z_dim
        self.d_layers = nn.ModuleList()
        if h_dim_d is not None and h_dim_d != [] and h_dim_d != 0 and h_dim_d != [0]:
            neurons = [z_dim, *h_dim_d]
            for i in range(len(neurons) - 1):
                self.d_layers.append(nn.Linear(neurons[i], neurons[i + 1]))
                torch.nn.init.normal_(self.d_layers[-1].weight, mean=0.0, std=.05)
                torch.nn.init.normal_(self.d_layers[-1].bias, mean=0.0, std=.05)
                self.d_layers.append(nn.ReLU())
            y_input_dim = h_dim_d[-1]
        #
        self.hidden = nn.Sequential(*self.d_layers)

        self.y_layer = nn.ModuleList()
        if self.conv:
            self.y_layer.append(nn.Linear(y_input_dim, 32 * 9 * 9))
        else:
            self.y_layer.append(nn.Linear(y_input_dim, self.y_dim_output))
        torch.nn.init.normal_(self.y_layer[0].weight, mean=0.0, std=.05)
        torch.nn.init.normal_(self.y_layer[0].bias, mean=0.0, std=.05)
        self.y_layer = nn.Sequential(*self.y_layer)

        if self.conv:
            self.deconv_layer = nn.ModuleList()
            self.deconv_layer.append(nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1))
            self.deconv_layer.append(nn.ReLU())
            self.deconv_layer.append(
                nn.ConvTranspose2d(in_channels=16, out_channels=self.y_dim, kernel_size=4, stride=2, padding=1))
            self.Decoder_Conv_layer = nn.Sequential(*self.deconv_layer)

        self.obs_layer = nn.ModuleList()
        for i, tpl in enumerate(self.types_info['set_of_types']):
            obs_input_dim = self.y_dim
            type_dim_num = sum(self.types_info['data_types_indexes'] == i)
            if tpl[0] == 'count':
                self.obs_layer.append(
                    Observation_Count(type_dim_num, obs_input_dim))
            elif tpl[0] in ['pos', 'real', 'beta']:
                self.obs_layer.append(
                    Observation_Real_Pos_Beta(type_dim_num, self.y_dim, self.logvar_network))
                if tpl[0] == 'real' and self.conv:
                    ##For convolutional HLVAE, for real type
                    self.obs_layer.append(nn.Sigmoid())
            elif tpl[0] == 'cat':
                self.obs_layer.append(
                    Observation_Cat(type_dim_num, obs_input_dim,
                                    int(tpl[1])))
            elif tpl[0] == 'ordinal':
                self.obs_layer.append(
                    Observation_Ordinal(type_dim_num, self.y_dim,
                                    int(tpl[1])))

    def encode(self, data, mask, param_mask, types_info, norm_params=None, X_list=None):
        if norm_params == None:
            # Batch normalization of the data
            X_list, norm_params = batch_normalization(data,
                                                              mask, param_mask, types_info)
        ##Encoder
        q_params = dict.fromkeys(['s', 'z'], None)
        samples = dict.fromkeys(['s', 'z'], None)

        if self.conv:
            one_to_one = torch.zeros_like(mask).to(torch.float64)
            layer_indx = 0
            for i, tpl in enumerate(self.types_info['set_of_types']):
                nclass = int(tpl[1])
                if tpl[0] in ['cat', 'ordinal']:
                    representation = self.representation_layer[layer_indx](X_list[:, self.types_info['exp_types_indexes'] == i].reshape(mask.shape[0],-1, nclass))
                    layer_indx += 1
                else:
                    representation = X_list[:, self.types_info['exp_types_indexes'] == i]
                one_to_one[:, self.types_info['data_types_indexes'] == i] = representation * mask[:, self.types_info['data_types_indexes'] == i]
            X_list_one_to_one = one_to_one.view(X_list.shape[0], 1, 36, 36)
            z = F.relu(self.conv1(X_list_one_to_one))
            z = self.pool1(z)
            z = F.relu(self.conv2(z))
            X_list_feature = self.pool2(z).view(-1, 32 * 9 * 9)

        if self.conv:
            encoder_input = X_list_feature
        else:
            encoder_input = X_list

        mean_qz = self.mean_layer(self.VAE_encoder_common_layers(encoder_input))
        log_var_qz = self.log_var_layer(self.VAE_encoder_common_layers(encoder_input))

        log_var_qz = torch.clamp(log_var_qz, -15.0, 15.0)

        samples['z'] = self.sample_latent(mean_qz, log_var_qz)
        q_params['z'] = [mean_qz, log_var_qz]

        return samples, q_params

    def decode(self, z, batch_x, miss_list, param_mask, norm_params=None):
        if norm_params == None:
            ## Data Preparation
            # Batch normalization of the data
            X_list, norm_params = batch_normalization(batch_x,
                                                      miss_list, param_mask, self.types_info)
        # Deterministic homogeneous representation y = g(z)
        p_params = dict()
        p_samples = dict()

        z = self.hidden(z)
        y = self.y_layer(z)
        if self.conv:
            y = y.view(-1, 32, 9, 9)
            y = self.Decoder_Conv_layer(y)
            y_grouped = y.view(y.shape[0], y.shape[1], -1).permute(0, 2, 1)
        else:
            y_grouped = torch.reshape(y, [y.shape[0], miss_list.shape[1], -1])
        theta = self.theta_estimation(y_grouped, miss_list, param_mask)

        log_p_x, log_p_x_missing, p_samples['x'], p_params['x'] = self.loglik_and_reconstruction(theta, batch_x,
                                                                                       miss_list, param_mask,
                                                                                       norm_params)
        return log_p_x, log_p_x_missing, p_samples, p_params

    def sample_latent(self, mu, log_var):
        """
        Sample from the latent space

        :param mu: variational mean
        :param log_var: variational variance
        :return: latent sample
        """
        # generate samples
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data, mask, param_mask, types_info, do_test=False):
        # ## Data Preparation
        X_list, norm_params = batch_normalization(data,
                                                  mask, param_mask, self.types_info)
        ##Encoder
        q_samples, q_params = self.encode(data, mask, param_mask, types_info, norm_params, X_list)
        mean_qz, log_var_qz = q_params['z']

        log_p_x, log_p_x_missing, p_samples, p_params = self.decode(q_samples['z'], data, mask, param_mask,
                                                                    norm_params=norm_params)

        return p_samples, mean_qz, log_var_qz, log_p_x, log_p_x_missing, p_params, q_samples, q_params

    def loss_function(self, log_px):
        ll = torch.sum(log_px, 1)
        return -ll

    def loglik_and_reconstruction(self, theta, batch_data_list, miss_list, param_miss_list, normalization_params, s=None):
        log_p_x = torch.zeros_like(miss_list).to(torch.float64)
        log_p_x_missing = torch.zeros_like(miss_list).to(torch.float64)
        samples_x = []
        params_x = []

        for i, tpl in enumerate(self.types_info['set_of_types']):
            loglik_function = getattr(loglik, 'loglik_' + tpl[0])
            batch_data_list_of_type = batch_data_list[:, self.types_info['exp_types_indexes'] == i]
            extra_params = None
            if tpl[0] == 'real':
                normalization_params_of_type = normalization_params[0]
                if self.conv:
                    batch_data_list_of_type = batch_data_list_of_type / 255
                extra_params = self._log_vy_real
            elif tpl[0] == 'pos':
                normalization_params_of_type = normalization_params[1]
                extra_params = self._log_vy_pos
            elif tpl[0] == 'beta':
                normalization_params_of_type = np.concatenate(self.types_info['beta_ranges'])
                extra_params = self._disp_param

            else:
                normalization_params_of_type = torch.tensor(0.)
            out = loglik_function([batch_data_list_of_type, miss_list[:, self.types_info['data_types_indexes'] == i]], tpl,
                                  theta[:, self.types_info['param_indexes'] == i],
                                  normalization_params_of_type, extra_params)

            log_p_x[:, self.types_info['data_types_indexes'] == i] = out['log_p_x']
            log_p_x_missing[:, self.types_info['data_types_indexes'] == i] = out['log_p_x_missing']
            samples_x.append(out['samples'])
            params_x.append(out['params'])

        return log_p_x, log_p_x_missing, samples_x, params_x

    def theta_estimation(self, y, miss_list, param_miss_list):
        # independent yd -> Compute p(xd|yd)
        theta = torch.zeros([y.shape[0], len(self.types_info['param_indexes'])], dtype=torch.float64).to(y.device.type)
        observed_y = y * miss_list[:, :, None]
        missing_y = y * (1 - miss_list)[:, :, None]
        layer_num = 0
        for i, tpl in enumerate(self.types_info['set_of_types']):
            obs_output = observed_y[:, self.types_info['data_types_indexes'] == i, :]
            cov_dim = obs_output.shape[1]

            obs_output = self.obs_layer[layer_num](obs_output, self.logvar_network)

            if tpl[0] == 'real' and self.conv:
                ##Sigmoid Layer
                obs_output[:, :cov_dim] = self.obs_layer[layer_num+1](obs_output[:, :cov_dim])
            dim = int(tpl[1])
            obs_output = obs_output * param_miss_list[:, self.types_info['param_indexes'] == i].reshape((param_miss_list.shape[0]
                                                                                           , -1, dim))

            with torch.no_grad():
                miss_output = missing_y[:, self.types_info['data_types_indexes'] == i, :]

                miss_output = self.obs_layer[layer_num](miss_output, self.logvar_network)

                if tpl[0] == 'real' and self.conv:
                    ##Sigmoid Layer
                    miss_output[:, :cov_dim] = self.obs_layer[layer_num + 1](miss_output[:, :cov_dim])
                    layer_num += 1
                miss_output = miss_output * (1 - param_miss_list[:, self.types_info['param_indexes'] == i].reshape((param_miss_list.shape[0]
                                                                                           , -1, dim)))

            layer_num += 1
            theta[:, self.types_info['param_indexes'] == i] = obs_output.reshape(obs_output.shape[0], -1)
            tmp = theta[:, self.types_info['param_indexes'] == i]
            tmp[param_miss_list[:, self.types_info['param_indexes'] == i] == 0] = miss_output.reshape(obs_output.shape[0], -1)[
                param_miss_list[:, self.types_info['param_indexes'] == i] == 0]
            theta[:, self.types_info['param_indexes'] == i] = tmp
        return theta

    def get_test_samples(self, data, miss_list, param_mask, data_list = None, X_list=None, norm_params=None, s=None):
        with torch.no_grad():
            self.eval()
            if norm_params == None:
                # Batch normalization of the data
                X_list, norm_params = batch_normalization(data,
                                                          miss_list, param_mask, self.types_info)

            q_samples_test, q_params_test = self.encode(data, miss_list, param_mask, self.types_info, norm_params, X_list)
            mean_qz, log_var_qz = q_params_test['z']

            if q_params_test['s'] is not None:
                s_samples_test = F.one_hot(torch.argmax(q_params_test['s'], 1),
                                                          num_classes=q_params_test['s'].shape[1]).to(torch.double)
            else:
                s_samples_test = None

            log_p_x_test, log_p_x_missing_test, p_samples_test, p_params_test = self.decode(mean_qz, data, miss_list, param_mask,
                                                                        norm_params=norm_params)
        self.train()
        return q_samples_test, q_params_test, p_samples_test, p_params_test, log_p_x_test, log_p_x_missing_test


# ============================================================
# __main__: Automated test suite for 4 ablated functions
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    passed = 0
    failed = 0

    def check(test_name, condition, detail=""):
        global passed, failed
        if bool(condition):
            passed += 1
            print(f"  [{test_name}] PASS")
        else:
            failed += 1
            suffix = f" - {detail}" if detail else ""
            print(f"  [{test_name}] FAIL{suffix}")

    def skip_checks(count, reason):
        global failed
        failed += count
        print(f"  [skipped {count} check(s)] FAIL - {reason}")

    def build_types_info():
        return {
            "types_dict": [
                {"type": "real", "dim": 1},
                {"type": "count", "dim": 1},
                {"type": "cat", "dim": 1, "nclass": 3},
            ],
            "set_of_types": [("real", "1"), ("count", "1"), ("cat", "3")],
            "data_types_indexes": np.array([0, 1, 2]),
            "exp_types_indexes": np.array([0, 1, 2, 2, 2]),
            "param_indexes": np.array([0, 0, 1, 2, 2, 2]),
            "beta_ranges": [],
            "conv": False,
        }

    def build_model():
        types_info = build_types_info()
        model = HLVAE(
            dims=[5, [8], 3, [7], 2],
            types_info=types_info,
            n_variables=3,
            vy_init=[1.0, 0.5],
            vy_fixed=False,
            logvar_network=False,
            conv=False,
        ).double()
        return model, types_info

    def build_batch():
        data = torch.tensor([
            [1.0, 2.0, 1.0, 0.0, 0.0],
            [2.0, 3.0, 0.0, 1.0, 0.0],
            [3.0, 4.0, 0.0, 0.0, 1.0],
        ], dtype=torch.float64)
        mask = torch.ones(3, 3, dtype=torch.float64)
        param_mask = torch.ones(3, 6, dtype=torch.float64)
        return data, mask, param_mask

    print("=" * 70)
    print("HL-VAE: Heterogeneous Longitudinal Variational Autoencoder")
    print("Automated reproduction benchmark - 4 ablated functions")
    print("=" * 70)
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    # ==============================================================
    # Test 1/4: HLVAE.sample_latent
    # ==============================================================
    print("-" * 60)
    print("[Test 1/4] HLVAE.sample_latent - reparameterized Gaussian latent sample")
    try:
        model, _ = build_model()
        mu = torch.zeros(4, 3, dtype=torch.float64, requires_grad=True)
        log_var = torch.zeros(4, 3, dtype=torch.float64, requires_grad=True)
        z = model.sample_latent(mu, log_var)
        check("sample_latent output not None", z is not None)
        if z is not None:
            check("sample_latent output shape", tuple(z.shape) == (4, 3), f"expected (4, 3), got {tuple(z.shape)}")
            check("sample_latent output finite", torch.isfinite(z).all().item())
            check("sample_latent stochastic at unit variance", not torch.allclose(z, mu))
            z.sum().backward()
            grad_ok = mu.grad is not None and torch.allclose(mu.grad, torch.ones_like(mu.grad))
            check("sample_latent preserves gradient through mean", grad_ok)
        else:
            skip_checks(4, "HLVAE.sample_latent returned None")
    except Exception as e:
        skip_checks(5, f"HLVAE.sample_latent raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 2/4: HLVAE.encode
    # ==============================================================
    print("-" * 60)
    print("[Test 2/4] HLVAE.encode - heterogeneous input normalization to variational z")
    try:
        model, types_info = build_model()
        data, mask, param_mask = build_batch()
        X_list, norm_params = batch_normalization(data, mask, param_mask, types_info)
        samples, q_params = model.encode(data, mask, param_mask, types_info, norm_params, X_list)
        z = samples["z"] if samples is not None else None
        check("encode output not None", samples is not None and q_params is not None and z is not None)
        if samples is not None and q_params is not None and z is not None:
            mean_qz, log_var_qz = q_params["z"]
            check("encode latent sample shape", tuple(z.shape) == (3, 3), f"expected (3, 3), got {tuple(z.shape)}")
            check("encode parameter shapes", tuple(mean_qz.shape) == (3, 3) and tuple(log_var_qz.shape) == (3, 3))
            check("encode parameters finite", torch.isfinite(mean_qz).all().item() and torch.isfinite(log_var_qz).all().item())
            check("encode clamps log variance", (log_var_qz.min().item() >= -15.0) and (log_var_qz.max().item() <= 15.0))
        else:
            skip_checks(4, "HLVAE.encode returned None output(s)")
    except Exception as e:
        skip_checks(5, f"HLVAE.encode raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 3/4: HLVAE.theta_estimation
    # ==============================================================
    print("-" * 60)
    print("[Test 3/4] HLVAE.theta_estimation - heterogeneous decoder parameter routing")
    try:
        model, _ = build_model()
        _, mask, param_mask = build_batch()
        y = torch.randn(3, 3, 2, dtype=torch.float64, requires_grad=True)
        theta = model.theta_estimation(y, mask, param_mask)
        check("theta_estimation output not None", theta is not None)
        if theta is not None:
            check("theta_estimation output shape", tuple(theta.shape) == (3, 6), f"expected (3, 6), got {tuple(theta.shape)}")
            check("theta_estimation output finite", torch.isfinite(theta).all().item())
            cat_first_logit_zero = torch.allclose(theta[:, 3], torch.zeros(3, dtype=torch.float64), atol=1e-8)
            check("theta_estimation includes categorical baseline logit", cat_first_logit_zero)
            theta.sum().backward()
            grad_ok = model.obs_layer[0].weight_mean.grad is not None and model.obs_layer[0].weight_mean.grad.abs().sum().item() > 0
            check("theta_estimation routes gradients through observed decoder path", grad_ok)
        else:
            skip_checks(4, "HLVAE.theta_estimation returned None")
    except Exception as e:
        skip_checks(5, f"HLVAE.theta_estimation raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 4/4: HLVAE.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 4/4] HLVAE.forward - end-to-end heterogeneous VAE pass")
    try:
        model, types_info = build_model()
        data, mask, param_mask = build_batch()
        outputs = model(data, mask, param_mask, types_info)
        check("forward output not None", outputs is not None)
        if outputs is not None:
            p_samples, mean_qz, log_var_qz, log_p_x, log_p_x_missing, p_params, q_samples, q_params = outputs
            check("forward latent parameter shapes", tuple(mean_qz.shape) == (3, 3) and tuple(log_var_qz.shape) == (3, 3))
            check("forward log-likelihood shapes", tuple(log_p_x.shape) == (3, 3) and tuple(log_p_x_missing.shape) == (3, 3))
            finite_ok = torch.isfinite(mean_qz).all().item() and torch.isfinite(log_var_qz).all().item() and torch.isfinite(log_p_x).all().item()
            check("forward outputs finite latent params and observed loglik", finite_ok)
            check("forward returns one reconstruction block per heterogeneous type", isinstance(p_samples["x"], list) and len(p_samples["x"]) == 3)
        else:
            skip_checks(4, "HLVAE.forward returned None")
    except Exception as e:
        skip_checks(5, f"HLVAE.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Final Score
    # ==============================================================
    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed == 0:
        print("All tests PASSED! The model code is complete and correct.")
    else:
        print(f"{failed} check(s) FAILED - some ablated functions may not be implemented correctly.")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
