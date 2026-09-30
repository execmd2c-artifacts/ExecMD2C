"""
Standalone benchmark ground truth for RioGNN core model components.

This file consolidates the model-side implementation needed to reproduce the
reinforced neighbor selection and multi-relation aggregation logic of RioGNN.
Training loops, dataset loading, evaluation utilities, and CLI code are excluded.
"""

from operator import itemgetter
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable


# --- [Original file: RL/actor_critic.py] ---
class PGNetwork(nn.Module):

    def __init__(self, state_dim, action_dim):
        """
        Initialize PGNetwork.
        :param state_dim: dimension of the state
        :param action_dim: dimension of the action
        """
        super(PGNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 20)
        self.fc2 = nn.Linear(20, action_dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            nn.init.normal_(m.weight.data, 0, 0.1)
            nn.init.constant_(m.bias.data, 0.01)


class Actor(object):

    def __init__(self, state_dim, action_dim, device, LR):
        # Dimensions of state space and action space
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.LR = LR

        # init network parameters
        self.network = PGNetwork(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.LR)

        # init some parameters
        self.time_step = 0

    def choose_action(self, observation):
        observation = torch.FloatTensor(observation).to(self.device)
        network_output = self.network.forward(observation)
        with torch.no_grad():
            # prob_weights = F.softmax(network_output, dim=0).cuda().data.cpu().numpy()
            prob_weights = F.softmax(network_output, dim=0).data.cpu().numpy()
        # prob_weights = F.softmax(network_output, dim=0).detach().numpy()
        action = np.random.choice(range(prob_weights.shape[0]),
                                  p=prob_weights)  # select action w.r.t the actions prob
        return action

    def learn(self, state, action, td_error):
        self.time_step += 1
        # Step 1: Forward propagation
        softmax_input = self.network.forward(torch.FloatTensor(state).to(self.device)).unsqueeze(0)
        action = torch.LongTensor([action]).to(self.device)
        neg_log_prob = F.cross_entropy(input=softmax_input, target=action, reduction='none')

        # Step 2: Backpropagation
        # Here you need to maximize the value of the current strategy,
        # so you need to maximize "neg_log_prob * tf_error", that is, minimize "-neg_log_prob * td_error"
        loss_a = -neg_log_prob * td_error
        self.optimizer.zero_grad()
        loss_a.backward()
        self.optimizer.step()


class QNetwork(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            nn.init.normal_(m.weight.data, 0, 0.1)
            nn.init.constant_(m.bias.data, 0.01)


class Critic(object):

    def __init__(self, state_dim, action_dim, device, LR, GAMMA):
        # Dimensions of state space and action space
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.LR = LR
        self.GAMMA = GAMMA

        # init network parameters
        self.network = QNetwork(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.LR)
        self.loss_func = nn.MSELoss()

    def train_Q_network(self, state, reward, next_state):
        s, s_ = torch.FloatTensor(state).to(self.device), torch.FloatTensor(next_state).to(self.device)
        # Forward propagation
        v = self.network.forward(s)  # v(s)
        v_ = self.network.forward(s_)  # v(s')

        # Backpropagation
        loss_q = self.loss_func(reward + self.GAMMA * v_, v)
        self.optimizer.zero_grad()
        loss_q.backward()
        self.optimizer.step()

        with torch.no_grad():
            td_error = reward + self.GAMMA * v_ - v

        return td_error


# --- [Original file: RL/rl_model.py] ---
class RLForest:

    def __init__(self, width_rl, height_rl, device, LR, GAMMA, stop_num, r_num):
        """
        Initialize the RL Forest.
        :param width_rl: width of each relation tree
        :param height_rl: height of each relation tree
        :param device: "cuda" / "cpu"
        :param LR: Actor learning rate (hyper-parameters of AC)
        :param GAMMA: Actor discount factor (hyper-parameters of AC)
        :param stop_num: deep switching or termination conditions
        :param r_num: the number of relations
        """

        self.actors = [[Actor(1, width_rl[r], device, LR) for j in range(height_rl[r])]
                       for r in range(r_num)]
        self.critics = [[Critic(1, width_rl[r], device, LR, GAMMA) for j in range(height_rl[r])]
                        for r in range(r_num)]
        self.r_num = r_num

        # current RLT depth for each relation
        self.init_rl = [0 for r in range(r_num)]
        # number of epochs performed at the current depth for each relation
        self.init_termination = [0 for r in range(r_num)]
        # action interval of current depth for each relation
        self.init_action = [0 for r in range(r_num)]

        # backtracking
        self.max_auc = 0
        self.max_thresholds = [0 for r in range(r_num)]

        # termination and boundary conditions
        self.width = list(width_rl)
        self.stop_num = stop_num

        # log
        self.thresholds_log = []
        self.actions_log = []
        self.states_log = []
        self.scores_log = []
        self.rewards_log = []

    def get_threshold(self, scores, labels, previous_thresholds, batch_num, auc):
        """
        The reinforcement learning module.
        It updates the neighbor filtering threshold for each relation based
        on the average neighbor distances between two consecutive epochs.
        :param scores: the neighbor nodes label-aware scores for each relation
        :param labels: the batch node labels used to select positive nodes
        :param previous_thresholds: the current neighbor filtering thresholds for each relation
        :param batch_num: numbers batches in an epoch
        :param auc: the auc of the previous filter thresholds for each relation
        """

        new_scores = get_scores(scores, labels)
        rl_flag0 = 0

        # during the epoch
        if len(self.scores_log) % batch_num != 0 or len(self.scores_log) < batch_num:

            # do not call RL module within the epoch or within the first two epochs
            new_thresholds = list(previous_thresholds)

        # after completing each epoch
        else:

            # STATE
            # get current states according to average scores
            # Eq.(8) in the paper
            current_epoch_states = [sum(s) / batch_num for s in zip(*self.scores_log[-batch_num:])]
            new_states = [np.array([s], float) for i, s in enumerate(current_epoch_states)]

            # backtracking
            if auc >= self.max_auc:
                self.max_auc = auc
                self.max_thresholds = list(previous_thresholds)

            new_actions = [0 for r in range(self.r_num)]
            new_thresholds = [0 for r in range(self.r_num)]

            # the first epoch
            if len(self.states_log) == 0:
                # update the record of the number of epochs in the current depth
                self.init_termination = [i + 1 for i in self.init_termination]
                # ACTION
                # get current actions for current states
                # Eq.(11) in the paper
                for r_num in range(self.r_num):
                    new_actions[r_num], new_thresholds[r_num] = self.get_action(new_states, r_num)

            # after the first epoch
            else:
                # STATE
                # get previous states
                previous_states = self.states_log[-1]
                # ACTION
                # get previous actions
                previous_actions = self.actions_log[-1]

                # REWARD
                # compute reward for each relation
                # Eq. (9) in the paper
                new_rewards = [s if 0 < previous_thresholds[i] and previous_thresholds[i] <= 1 else -100 for i, s in
                               enumerate(current_epoch_states)]

                # determine whether to enter the next depth
                r_flag = self.adjust_depth()

                # after the smallest continuous epoch
                for r_num in range(self.r_num):

                    # go to the next depth
                    if r_flag[r_num] == 1:

                        if len(self.actors[r_num]) == self.init_rl[r_num] + 1:
                            # relation tree remains unchanged after converging
                            self.init_termination[r_num] = self.init_termination[r_num]
                            # ACTION
                            new_actions[r_num] = previous_actions[r_num]
                            new_thresholds[r_num] = self.max_thresholds[r_num]
                            rl_flag0 += 1
                            print("Relation {0} is complete 锛侊紒锛侊紒!".format(str(r_num + 1)), flush=True)

                        else:
                            # update the parameter space when entering the next depth
                            # Eq. (7) in the paper
                            self.init_termination[r_num] = 0
                            self.init_rl[r_num] = self.init_rl[r_num] + 1
                            self.init_action[r_num] = self.max_thresholds[r_num] - (self.width[r_num] / 2) * \
                                                      pow(1 / self.width[r_num], self.init_rl[r_num] + 1)
                            # ACTION
                            # Eq. (11) in the paper
                            new_actions[r_num], new_thresholds[r_num] = self.get_action(new_states, r_num)

                    # keep current depth
                    else:
                        self.init_termination[r_num] = self.init_termination[r_num] + 1
                        # POLICY
                        # Eq. (10) in the paper
                        self.learn(previous_states, previous_actions, new_states, new_rewards, r_num)
                        # ACTION
                        # Eq. (11) in the paper
                        new_actions[r_num], new_thresholds[r_num] = self.get_action(new_states, r_num)

                self.rewards_log.append(new_rewards)
                print('Rewards:  ' + str(new_rewards), flush=True)

            self.states_log.append(new_states)
            print('States:  ' + str(new_states), flush=True)
            self.thresholds_log.append(new_thresholds)
            print('Thresholds:  ' + str(new_thresholds), flush=True)
            self.actions_log.append(new_actions)

        self.scores_log.append(new_scores)

        print("Historical maximum AUC:  " + str(self.max_auc), flush=True)
        print("Thresholds to obtain the historical maximum AUC:  " + str(self.max_thresholds), flush=True)
        print('Current depth of each RL Tree:  ' + str(self.init_rl), flush=True)

        # RLF termination
        rl_flag = False if rl_flag0 == self.r_num else True
        print('Completion flag of the entire RL Forest:  ' + str(rl_flag), flush=True)

        return new_thresholds, rl_flag

    def learn(self, previous_states, previous_actions, new_states, new_rewards, r_num):
        """
        :param previous_states: the previous states
        :param previous_actions: the previous actions
        :param new_states: the current states
        :param new_rewards: the current rewards
        :param r_num: the index of relation
        """

        td_error = self.critics[r_num][self.init_rl[r_num]].train_Q_network(previous_states[r_num],
                                                                            new_rewards[r_num],
                                                                            new_states[r_num])
        self.actors[r_num][self.init_rl[r_num]].learn(previous_states[r_num],
                                                      previous_actions[r_num],
                                                      td_error)
        return

    def get_action(self, new_states, r_num):
        """
        :param new_states: the current states
        :param r_num: the index of relation
        :returns: new actions and thresholds for new_states under relation r_num
        """

        new_actions = self.actors[r_num][self.init_rl[r_num]].choose_action(new_states[r_num])
        new_thresholds = self.init_action[r_num] + (new_actions + 1) * \
                         pow(1 / self.width[r_num], self.init_rl[r_num] + 1)
        new_thresholds = 1 if new_thresholds >= 1 else new_thresholds

        return new_actions, new_thresholds

    def adjust_depth(self):
        """
        :returns: the depth flag of each relation
        """

        r_flag = [1 for r in range(self.r_num)]
        for r_num in range(self.r_num):
            if self.init_termination[r_num] > self.stop_num:
                for s in range(self.stop_num - 1):
                    r_flag[r_num] = r_flag[r_num] * (
                        1 if self.actions_log[-1 * (s + 1)][r_num] == self.actions_log[-1 * (s + 2)][r_num] else 0
                    )
            else:
                r_flag[r_num] = 0

        return r_flag


def get_scores(scores, labels):
    """
    Get the scores of current batch.
    :param scores: the neighbor nodes label-aware scores for each relation
    :param labels: the batch node labels used to select positive nodes
    :returns: the state of current batch
    """

    relation_scores = []

    # only compute the average neighbor distances for positive nodes
    pos_index = (labels == 1).nonzero().tolist()
    pos_index = [i[0] for i in pos_index]

    # compute average neighbor distances for each relation
    for score in scores:
        pos_scores = itemgetter(*pos_index)(score)
        neigh_count = sum([1 if isinstance(i, float) else len(i) for i in pos_scores])
        pos_sum = [i if isinstance(i, float) else sum(i) for i in pos_scores]
        relation_scores.append(sum(pos_sum) / neigh_count)

    return relation_scores


# --- [Original file: model/layers.py] ---
class InterAgg(nn.Module):

    def __init__(self, width_rl, height_rl, device, LR, GAMMA, stop_num,
                 features, feature_dim,
                 embed_dim, adj_lists, intra_aggs,
                 inter, cuda=True):
        """
        Initialize the inter-relation aggregator
        :param width_rl: width of each relation tree
        :param height_rl: height of each relation tree
        :param device: "cuda" / "cpu"
        :param LR: Actor learning rate (hyper-parameters of AC)
        :param GAMMA: Actor discount factor (hyper-parameters of AC)
        :param stop_num: deep switching or termination conditions
        :param features: the input node features or embeddings for all nodes
        :param feature_dim: the input dimension
        :param embed_dim: the output dimension
        :param adj_lists: a list of adjacency lists for each single-relation graph
        :param intra_aggs: the intra-relation aggregators used by each single-relation graph
        :param inter: the aggregator type: 'Att', 'Weight', 'Mean', 'GNN'
        :param cuda: whether to use GPU
        """
        super(InterAgg, self).__init__()

        self.features = features
        self.dropout = 0.6
        self.adj_lists = adj_lists
        self.intra_aggs = intra_aggs
        self.embed_dim = embed_dim
        self.feat_dim = feature_dim
        self.inter = inter
        self.cuda = cuda

        # initial filtering thresholds
        self.thresholds = [0.5 for r in range(len(intra_aggs))]

        # RL condition flag
        self.RL = True
        self.rl_tree = RLForest(width_rl, height_rl, device, LR, GAMMA, stop_num, len(intra_aggs))

        # number of batches for current epoch, assigned during training
        self.batch_num = 0
        self.auc = 0

        # the activation function used by attention mechanism
        self.leakyrelu = nn.LeakyReLU(0.2)

        # parameter used to transform node embeddings before inter-relation aggregation
        self.weight = nn.Parameter(torch.FloatTensor(self.embed_dim, self.feat_dim))
        init.xavier_uniform_(self.weight)

        # weight parameter for each relation used by Rio-Weight
        self.alpha = nn.Parameter(torch.FloatTensor(self.embed_dim, len(intra_aggs)))
        init.xavier_uniform_(self.alpha)

        # parameters used by attention layer
        self.a = nn.Parameter(torch.FloatTensor(2 * self.embed_dim, 1))
        init.xavier_uniform_(self.a)

        # label predictor for similarity measure
        self.label_clf = nn.Linear(self.feat_dim, 2)

        # initialize the parameter logs
        self.weights_log = []

    def forward(self, nodes, labels, train_flag=True):
        """
        :param nodes: a list of batch node ids
        :param labels: a list of batch node labels, only used by the RLModule
        :param train_flag: indicates whether in training or testing mode
        :return combined: the embeddings of a batch of input node features
        :return center_scores: the label-aware scores of batch nodes
        """

        # extract 1-hop neighbor ids from adj lists of each single-relation graph
        to_neighs = []
        for adj_list in self.adj_lists:
            to_neighs.append([set(adj_list[int(node)]) for node in nodes])

        # find unique nodes and their neighbors used in current batch
        unique_nodes = set.union(*(set.union(*to_neighs[r]) for r in range(len(self.intra_aggs))), set(nodes))

        # calculate label-aware scores
        if self.cuda:
            batch_features = self.features(torch.cuda.LongTensor(list(unique_nodes)))
        else:
            batch_features = self.features(torch.LongTensor(list(unique_nodes)))
        batch_scores = self.label_clf(batch_features)
        id_mapping = {node_id: index for node_id, index in zip(unique_nodes, range(len(unique_nodes)))}

        # the label-aware scores for current batch of nodes
        center_scores = batch_scores[itemgetter(*nodes)(id_mapping), :]

        # get neighbor node id list for each batch node and relation
        r_list = [[list(to_neigh) for to_neigh in to_neighs[r]] for r in range(len(self.intra_aggs))]

        # assign label-aware scores to neighbor nodes for each batch node and relation
        r_scores = [[batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r_list[r]]
                    for r in range(len(self.intra_aggs))]

        # count the number of neighbors kept for aggregation for each batch node and relation
        r_sample_num_list = [[math.ceil(len(neighs) * self.thresholds[r]) for neighs in r_list[r]]
                             for r in range(len(self.intra_aggs))]

        # intra-aggregation steps for each relation
        # Eq. (8) in the paper
        r_feats, r_scores = tuple(
            zip(*list(self.intra_aggs[r].forward(nodes, r_list[r], center_scores, r_scores[r], r_sample_num_list[r])
                      for r in range(len(self.intra_aggs)))))

        # concat the intra-aggregated embeddings from each relation
        neigh_feats = torch.cat(r_feats, dim=0)

        # get features or embeddings for batch nodes
        if self.cuda and isinstance(nodes, list):
            index = torch.LongTensor(nodes).cuda()
        else:
            index = torch.LongTensor(nodes)
        self_feats = self.features(index)

        # number of nodes in a batch
        n = len(nodes)

        # inter-relation aggregation steps
        # Eq. (9) in the paper
        if self.inter == 'Att':
            # 1) Rio-Att Inter-relation Aggregator
            combined, attention = att_inter_agg(len(self.adj_lists), self.leakyrelu, self_feats, neigh_feats,
                                                self.embed_dim,
                                                self.weight, self.a, n, self.dropout, self.training, self.cuda)
        elif self.inter == 'Weight':
            # 2) Rio-Weight Inter-relation Aggregator
            combined = weight_inter_agg(len(self.adj_lists), self_feats, neigh_feats, self.embed_dim, self.weight,
                                        self.alpha, n, self.cuda)
            gem_weights = F.softmax(torch.sum(self.alpha, dim=0), dim=0).tolist()
            if train_flag:
                print(f'Weights: {gem_weights}')
        elif self.inter == 'Mean':
            # 3) Rio-Mean Inter-relation Aggregator
            combined = mean_inter_agg(len(self.adj_lists), self_feats, neigh_feats, self.embed_dim, self.weight, n,
                                      self.cuda)
        elif self.inter == 'GNN':
            # 4) Rio-GNN Inter-relation Aggregator
            combined = threshold_inter_agg(len(self.adj_lists), self_feats, neigh_feats, self.embed_dim, self.weight,
                                           self.thresholds, n, self.cuda)

        # the reinforcement learning module
        if self.RL and train_flag:
            thresholds, stop_flag = self.rl_tree.get_threshold(list(r_scores), labels, self.thresholds, self.batch_num,
                                                               self.auc)
            self.thresholds = thresholds
            self.RL = stop_flag

        return combined, center_scores


class IntraAgg(nn.Module):

    def __init__(self, features, feat_dim, cuda=False):
        """
        Initialize the intra-relation aggregator
        :param features: the input node features or embeddings for all nodes
        :param feat_dim: the input dimension
        :param cuda: whether to use GPU
        """
        super(IntraAgg, self).__init__()

        self.features = features
        self.cuda = cuda
        self.feat_dim = feat_dim

    def forward(self, nodes, to_neighs_list, batch_scores, neigh_scores, sample_list):
        """
        Code partially from https://github.com/williamleif/graphsage-simple/
        :param nodes: list of nodes in a batch
        :param to_neighs_list: neighbor node id list for each batch node in one relation
        :param batch_scores: the label-aware scores of batch nodes
        :param neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
        :param sample_list: the number of neighbors kept for each batch node in one relation
        :return to_feats: the aggregated embeddings of batch nodes neighbors in one relation
        :return samp_scores: the average neighbor distances for each relation after filtering
        """

        # filer neighbors under given relation
        samp_neighs, samp_scores = filter_neighs_ada_threshold(batch_scores, neigh_scores, to_neighs_list, sample_list)

        # find the unique nodes among batch nodes and the filtered neighbors
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

        # intra-relation aggregation only with sampled neighbors
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for _ in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        to_feats = mask.mm(embed_matrix)
        to_feats = F.relu(to_feats)
        return to_feats, samp_scores


def filter_neighs_ada_threshold(center_scores, neigh_scores, neighs_list, sample_list):
    """
    Filter neighbors according label predictor result with adaptive thresholds
    :param center_scores: the label-aware scores of batch nodes
    :param neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
    :param neighs_list: neighbor node id list for each batch node in one relation
    :param sample_list: the number of neighbors kept for each batch node in one relation
    :return samp_neighs: the neighbor indices and neighbor simi scores
    :return samp_scores: the average neighbor distances for each relation after filtering
    """

    samp_neighs = []
    samp_scores = []
    for idx, center_score in enumerate(center_scores):
        center_score = center_scores[idx][0]
        neigh_score = neigh_scores[idx][:, 0].view(-1, 1)
        center_score = center_score.repeat(neigh_score.size()[0], 1)
        neighs_indices = neighs_list[idx]
        num_sample = sample_list[idx]

        # compute the L1-distance of batch nodes and their neighbors
        # Eq. (2) in paper
        score_diff = torch.abs(center_score - neigh_score).squeeze()
        sorted_scores, sorted_indices = torch.sort(score_diff, dim=0, descending=False)
        selected_indices = sorted_indices.tolist()

        # top-p sampling according to distance ranking and thresholds
        # Section 3.3.1 in paper
        if len(neigh_scores[idx]) > num_sample + 1:
            selected_neighs = [neighs_indices[n] for n in selected_indices[:num_sample]]
            selected_scores = sorted_scores.tolist()[:num_sample]
        else:
            selected_neighs = neighs_indices
            selected_scores = score_diff.tolist()
            if isinstance(selected_scores, float):
                selected_scores = [selected_scores]

        samp_neighs.append(set(selected_neighs))
        samp_scores.append(selected_scores)

    return samp_neighs, samp_scores


def mean_inter_agg(num_relations, self_feats, neigh_feats, embed_dim, weight, n, cuda):
    """
    Mean inter-relation aggregator
    :param num_relations: number of relations in the graph
    :param self_feats: batch nodes features or embeddings
    :param neigh_feats: intra-relation aggregated neighbor embeddings for each relation
    :param embed_dim: the dimension of output embedding
    :param weight: parameter used to transform node embeddings before inter-relation aggregation
    :param n: number of nodes in a batch
    :param cuda: whether use GPU
    :return: inter-relation aggregated node embeddings
    """

    # transform batch node embedding and neighbor embedding in each relation with weight parameter
    center_h = weight.mm(self_feats.t())
    neigh_h = weight.mm(neigh_feats.t())

    # initialize the final neighbor embedding
    if cuda:
        aggregated = torch.zeros(size=(embed_dim, n)).cuda()
    else:
        aggregated = torch.zeros(size=(embed_dim, n))

    # sum neighbor embeddings together
    for r in range(num_relations):
        aggregated += neigh_h[:, r * n:(r + 1) * n]

    # sum aggregated neighbor embedding and batch node embedding
    # take the average of embedding and feed them to activation function
    combined = F.relu((center_h + aggregated) / 4.0)

    return combined


def weight_inter_agg(num_relations, self_feats, neigh_feats, embed_dim, weight, alpha, n, cuda):
    """
    Weight inter-relation aggregator
    Reference: https://arxiv.org/abs/2002.12307
    :param num_relations: number of relations in the graph
    :param self_feats: batch nodes features or embeddings
    :param neigh_feats: intra-relation aggregated neighbor embeddings for each relation
    :param embed_dim: the dimension of output embedding
    :param weight: parameter used to transform node embeddings before inter-relation aggregation
    :param alpha: weight parameter for each relation used by Rio-Weight
    :param n: number of nodes in a batch
    :param cuda: whether use GPU
    :return: inter-relation aggregated node embeddings
    """

    # transform batch node embedding and neighbor embedding in each relation with weight parameter
    center_h = weight.mm(self_feats.t())
    neigh_h = weight.mm(neigh_feats.t())

    # compute relation weights using softmax
    w = F.softmax(alpha, dim=1)

    # initialize the final neighbor embedding
    if cuda:
        aggregated = torch.zeros(size=(embed_dim, n)).cuda()
    else:
        aggregated = torch.zeros(size=(embed_dim, n))

    # add weighted neighbor embeddings in each relation together
    for r in range(num_relations):
        aggregated += torch.mul(w[:, r].unsqueeze(1).repeat(1, n), neigh_h[:, r * n:(r + 1) * n])

    # sum aggregated neighbor embedding and batch node embedding
    # feed them to activation function
    combined = F.relu(center_h + aggregated)

    return combined


def att_inter_agg(num_relations, att_layer, self_feats, neigh_feats, embed_dim, weight, a, n, dropout, training, cuda):
    """
    Attention-based inter-relation aggregator
    Reference: https://github.com/Diego999/pyGAT
    :param num_relations: num_relations: number of relations in the graph
    :param att_layer: the activation function used by the attention layer
    :param self_feats: batch nodes features or embeddings
    :param neigh_feats: intra-relation aggregated neighbor embeddings for each relation
    :param embed_dim: the dimension of output embedding
    :param weight: parameter used to transform node embeddings before inter-relation aggregation
    :param a: parameters used by attention layer
    :param n: number of nodes in a batch
    :param dropout: dropout for attention layer
    :param training: a flag indicating whether in the training or testing mode
    :param cuda: whether use GPU
    :return combined: inter-relation aggregated node embeddings
    :return att: the attention weights for each relation
    """

    # transform batch node embedding and neighbor embedding in each relation with weight parameter
    center_h = self_feats.mm(weight.t())
    neigh_h = neigh_feats.mm(weight.t())

    # compute attention weights
    combined = torch.cat((center_h.repeat(num_relations, 1), neigh_h), dim=1)
    e = att_layer(combined.mm(a))
    attention = torch.cat((e[0:n, :], e[n:2 * n, :], e[2 * n:num_relations * n, :]), dim=1)
    ori_attention = F.softmax(attention, dim=1)
    attention = F.dropout(ori_attention, dropout, training=training)

    # initialize the final neighbor embedding
    if cuda:
        aggregated = torch.zeros(size=(n, embed_dim)).cuda()
    else:
        aggregated = torch.zeros(size=(n, embed_dim))

    # add neighbor embeddings in each relation together with attention weights
    for r in range(num_relations):
        aggregated += torch.mul(attention[:, r].unsqueeze(1).repeat(1, embed_dim), neigh_h[r * n:(r + 1) * n, :])

    # sum aggregated neighbor embedding and batch node embedding
    # feed them to activation function
    combined = F.relu((center_h + aggregated).t())

    # extract the attention weights
    att = F.softmax(torch.sum(ori_attention, dim=0), dim=0)

    return combined, att


def threshold_inter_agg(num_relations, self_feats, neigh_feats, embed_dim, weight, threshold, n, cuda):
    """
    Rio-GNN inter-relation aggregator
    Eq. (9) in the paper
    :param num_relations: number of relations in the graph
    :param self_feats: batch nodes features or embeddings
    :param neigh_feats: intra-relation aggregated neighbor embeddings for each relation
    :param embed_dim: the dimension of output embedding
    :param weight: parameter used to transform node embeddings before inter-relation aggregation
    :param threshold: the neighbor filtering thresholds used as aggregating weights
    :param n: number of nodes in a batch
    :param cuda: whether use GPU
    :return: inter-relation aggregated node embeddings
    """

    # transform batch node embedding and neighbor embedding in each relation with weight parameter
    center_h = weight.mm(self_feats.t())
    neigh_h = weight.mm(neigh_feats.t())

    if cuda:
        # use thresholds as aggregating weights
        w = torch.FloatTensor(threshold).repeat(weight.size(0), 1).cuda()

        # initialize the final neighbor embedding
        aggregated = torch.zeros(size=(embed_dim, n)).cuda()
    else:
        w = torch.FloatTensor(threshold).repeat(weight.size(0), 1)
        aggregated = torch.zeros(size=(embed_dim, n))

    # add weighted neighbor embeddings in each relation together
    for r in range(num_relations):
        aggregated += torch.mul(w[:, r].unsqueeze(1).repeat(1, n), neigh_h[:, r * n:(r + 1) * n])

    # sum aggregated neighbor embedding and batch node embedding
    # feed them to activation function
    combined = F.relu(center_h + aggregated)

    return combined


# --- [Original file: model/model.py] ---
class OneLayerRio(nn.Module):
    """
    The Rio-GNN model in one layer
    """

    def __init__(self, num_classes, inter1, lambda_1):
        """
        Initialize the Rio-GNN model
        :param num_classes: number of classes (2 in our paper)
        :param inter1: the inter-relation aggregator that output the final embedding
        """
        super(OneLayerRio, self).__init__()
        self.inter1 = inter1
        self.xent = nn.CrossEntropyLoss()

        # the parameter to transform the final embedding
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, inter1.embed_dim))
        init.xavier_uniform_(self.weight)
        self.lambda_1 = lambda_1

    def forward(self, nodes, labels, train_flag=True):
        embeds1, label_scores = self.inter1(nodes, labels, train_flag)
        scores = self.weight.mm(embeds1)
        return scores.t(), label_scores

    def to_prob(self, nodes, labels, train_flag=True):
        gnn_logits, label_logits = self.forward(nodes, labels, train_flag)
        gnn_scores = torch.sigmoid(gnn_logits)
        label_scores = torch.sigmoid(label_logits)
        return gnn_scores, label_scores

    def loss(self, nodes, labels, train_flag=True):
        gnn_scores, label_scores = self.forward(nodes, labels, train_flag)
        # Simi loss, Eq. (4) in the paper
        label_loss = self.xent(label_scores, labels.squeeze())
        # GNN loss, Eq. (10) in the paper
        gnn_loss = self.xent(gnn_scores, labels.squeeze())
        # the loss function of Rio-GNN, Eq. (11) in the paper
        final_loss = gnn_loss + self.lambda_1 * label_loss
        return final_loss


class TwoLayerRio(nn.Module):
    """
    The Rio-GNN model in one layer
    """

    def __init__(self, num_classes, inter1, inter2, lambda_1, last_label_scores):
        """
        Initialize the Rio-GNN model
        :param num_classes: number of classes (2 in our paper)
        :param inter1: the inter-relation aggregator that output the final embedding
        """
        super(TwoLayerRio, self).__init__()
        self.inter1 = inter1
        self.inter2 = inter2
        self.xent = nn.CrossEntropyLoss()

        # the parameter to transform the final embedding
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, inter2.embed_dim))
        init.xavier_uniform_(self.weight)
        self.lambda_1 = lambda_1
        self.last_label_scores = last_label_scores

    def forward(self, nodes, labels, train_flag=True):
        label_scores_one = self.last_label_scores
        embeds2, label_scores_two = self.inter2(nodes, labels, train_flag)
        scores2 = self.weight.mm(embeds2)
        return scores2.t(), label_scores_one, label_scores_two

    def to_prob(self, nodes, labels, train_flag=True):
        gnn_logits2, label_logits_one, label_logits_two = self.forward(nodes, labels, train_flag)
        gnn_scores2 = torch.sigmoid(gnn_logits2)
        label_scores_one = torch.sigmoid(label_logits_one)
        label_scores_two = torch.sigmoid(label_logits_two)
        return gnn_scores2, label_scores_one, label_scores_two

    def loss(self, nodes, labels, train_flag=True):
        gnn_scores2, label_scores_one, label_scores_two = self.forward(nodes, labels, train_flag)
        # Simi loss, Eq. (4) in the paper
        label_loss_one = self.xent(label_scores_one, labels.squeeze())
        label_loss_two = self.xent(label_scores_two, labels.squeeze())
        # GNN loss, Eq. (10) in the paper
        gnn_loss2 = self.xent(gnn_scores2, labels.squeeze())
        # the loss function of Rio-GNN, Eq. (11) in the paper
        final_loss = gnn_loss2 + self.lambda_1 * label_loss_one
        #final_loss = gnn_loss2 + (label_loss_one + label_loss_two)

        return final_loss


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

    def build_toy_rio(inter="GNN"):
        features = nn.Embedding(8, 4)
        with torch.no_grad():
            features.weight.copy_(torch.tensor([
                [1.0, 0.0, 0.5, 0.2],
                [0.9, 0.1, 0.4, 0.1],
                [0.2, 0.8, 0.1, 0.5],
                [0.1, 0.9, 0.2, 0.4],
                [0.8, 0.2, 0.3, 0.3],
                [0.3, 0.7, 0.6, 0.1],
                [0.4, 0.6, 0.1, 0.7],
                [0.7, 0.3, 0.5, 0.2],
            ]))
        adj_lists = [
            {0: {1, 2, 4}, 1: {0, 2, 5}, 2: {0, 1, 3}, 3: {2, 4, 6},
             4: {0, 3, 5}, 5: {1, 4, 7}, 6: {2, 3, 7}, 7: {5, 6}},
            {0: {3, 5, 6}, 1: {2, 4, 7}, 2: {1, 5, 6}, 3: {0, 4, 7},
             4: {1, 3, 6}, 5: {0, 2, 7}, 6: {0, 2, 4}, 7: {1, 3, 5}},
        ]
        intra_aggs = [IntraAgg(features, 4, cuda=False) for _ in adj_lists]
        inter_agg = InterAgg(
            width_rl=[3, 3],
            height_rl=[2, 2],
            device="cpu",
            LR=0.01,
            GAMMA=0.95,
            stop_num=2,
            features=features,
            feature_dim=4,
            embed_dim=5,
            adj_lists=adj_lists,
            intra_aggs=intra_aggs,
            inter=inter,
            cuda=False,
        )
        inter_agg.RL = False
        return inter_agg

    print("=" * 70)
    print("RioGNN reproduction benchmark: reinforced multi-relation aggregation")
    print("Automated Test Suite - 7 ablated targets")
    print("=" * 70)
    print()

    print("-" * 60)
    print("[Test 1/5] filter_neighs_ada_threshold")
    try:
        center_scores = torch.tensor([[0.10, 0.90], [0.80, 0.20]], dtype=torch.float32)
        neigh_scores = [
            torch.tensor([[0.12, 0.88], [0.60, 0.40], [0.20, 0.80]], dtype=torch.float32),
            torch.tensor([[0.70, 0.30], [0.10, 0.90], [0.82, 0.18]], dtype=torch.float32),
        ]
        neighs_list = [[10, 11, 12], [20, 21, 22]]
        sample_list = [1, 2]
        sampled_neighs, sampled_scores = filter_neighs_ada_threshold(center_scores, neigh_scores, neighs_list, sample_list)
        check("filter output not None", sampled_neighs is not None and sampled_scores is not None)
        if sampled_neighs is not None and sampled_scores is not None:
            check("filter batch length", len(sampled_neighs) == 2 and len(sampled_scores) == 2)
            check("filter applies adaptive sample count", len(sampled_neighs[0]) == 1 and len(sampled_neighs[1]) == 3)
            check("filter selects nearest first node", sampled_neighs[0] == {10}, f"got {sampled_neighs[0]}")
            check("filter keeps all when under threshold", sampled_neighs[1] == {20, 21, 22}, f"got {sampled_neighs[1]}")
            check("filter score list finite", all(np.isfinite(float(v)) for row in sampled_scores for v in row))
        else:
            skip_checks(5, "filter returned None")
    except Exception as exc:
        skip_checks(6, f"filter raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 2/5] IntraAgg.forward and threshold_inter_agg")
    try:
        features = nn.Embedding(6, 3)
        with torch.no_grad():
            features.weight.copy_(torch.tensor([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
            ]))
        intra = IntraAgg(features, 3, cuda=False)
        nodes = [0, 1]
        to_neighs_list = [[1, 2, 3], [2, 4, 5]]
        batch_scores = torch.tensor([[0.10, 0.90], [0.80, 0.20]], dtype=torch.float32)
        neigh_scores = [
            torch.tensor([[0.11, 0.89], [0.30, 0.70], [0.60, 0.40]], dtype=torch.float32),
            torch.tensor([[0.82, 0.18], [0.20, 0.80], [0.77, 0.23]], dtype=torch.float32),
        ]
        to_feats, samp_scores = intra(nodes, to_neighs_list, batch_scores, neigh_scores, [2, 2])
        check("intra output not None", to_feats is not None and samp_scores is not None)
        if to_feats is not None and samp_scores is not None:
            check("intra output shape", to_feats.shape == (2, 3), f"got {tuple(to_feats.shape)}")
            check("intra output finite", torch.isfinite(to_feats).all().item())
            check("intra relu nonnegative", torch.all(to_feats >= 0).item())
            check("intra scores per node", len(samp_scores) == 2 and all(len(s) == 3 for s in samp_scores))
        else:
            skip_checks(4, "intra aggregation returned None")

        self_feats = torch.randn(2, 3)
        neigh_feats = torch.randn(4, 3)
        weight = torch.randn(4, 3)
        combined = threshold_inter_agg(2, self_feats, neigh_feats, 4, weight, [0.25, 0.75], 2, False)
        check("threshold inter output not None", combined is not None)
        if combined is not None:
            check("threshold inter output shape", combined.shape == (4, 2), f"got {tuple(combined.shape)}")
            check("threshold inter output finite", torch.isfinite(combined).all().item())
            check("threshold inter relu nonnegative", torch.all(combined >= 0).item())
        else:
            skip_checks(3, "threshold inter returned None")
    except Exception as exc:
        skip_checks(9, f"intra/inter raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 3/5] RLForest state, action, and threshold update")
    try:
        forest_checks_started = False
        labels = torch.tensor([1, 0, 1], dtype=torch.long)
        relation_scores = [
            [[0.2, 0.4], [0.9], [0.1, 0.3]],
            [[0.5], [0.7, 0.8], [0.2, 0.6]],
        ]
        state_scores = get_scores(relation_scores, labels)
        check("get_scores output not None", state_scores is not None)
        if state_scores is not None:
            check("get_scores relation count", len(state_scores) == 2)
            check("get_scores first relation average", abs(state_scores[0] - 0.25) < 1e-6, f"got {state_scores[0]}")
            check("get_scores second relation average", abs(state_scores[1] - (1.3 / 3.0)) < 1e-6, f"got {state_scores[1]}")
        else:
            skip_checks(3, "get_scores returned None")

        forest_checks_started = True
        forest = RLForest([3, 3], [2, 2], "cpu", 0.01, 0.95, 1, 2)
        action, threshold = forest.get_action([np.array([0.2]), np.array([0.4])], 0)
        check("get_action output valid", action is not None and threshold is not None)
        if action is not None and threshold is not None:
            check("get_action action range", 0 <= action < 3, f"got {action}")
            check("get_action threshold bounds", 0 < threshold <= 1, f"got {threshold}")
        else:
            skip_checks(2, "get_action returned None")

        forest.scores_log = [[0.2, 0.3]]
        thresholds, keep_running = forest.get_threshold(relation_scores, labels, [0.5, 0.5], batch_num=1, auc=0.6)
        check("get_threshold output valid", thresholds is not None and keep_running is not None)
        if thresholds is not None and keep_running is not None:
            check("get_threshold relation count", len(thresholds) == 2)
            check("get_threshold bounds", all(0 < t <= 1 for t in thresholds), f"got {thresholds}")
            check("get_threshold updates logs", len(forest.states_log) == 1 and len(forest.actions_log) == 1)
            check("get_threshold keep flag bool", isinstance(keep_running, bool))
        else:
            skip_checks(4, "get_threshold returned None")
    except Exception as exc:
        skip_checks(8 if forest_checks_started else 12, f"RLForest raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 4/5] InterAgg.forward")
    try:
        inter_agg = build_toy_rio(inter="GNN")
        nodes = [0, 1, 2]
        labels = torch.tensor([1, 0, 1], dtype=torch.long)
        combined, center_scores = inter_agg(nodes, labels, train_flag=False)
        check("InterAgg output not None", combined is not None and center_scores is not None)
        if combined is not None and center_scores is not None:
            check("InterAgg combined shape", combined.shape == (5, 3), f"got {tuple(combined.shape)}")
            check("InterAgg center score shape", center_scores.shape == (3, 2), f"got {tuple(center_scores.shape)}")
            check("InterAgg combined finite", torch.isfinite(combined).all().item())
            check("InterAgg label scores finite", torch.isfinite(center_scores).all().item())
            check("InterAgg keeps RL disabled in eval", inter_agg.RL is False)
        else:
            skip_checks(5, "InterAgg returned None")
    except Exception as exc:
        skip_checks(6, f"InterAgg raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 5/5] OneLayerRio.forward/to_prob/loss")
    try:
        inter_agg = build_toy_rio(inter="GNN")
        model = OneLayerRio(2, inter_agg, lambda_1=2.0)
        nodes = [0, 1, 2]
        labels = torch.tensor([1, 0, 1], dtype=torch.long)
        logits, label_scores = model(nodes, labels, train_flag=False)
        check("OneLayerRio forward output not None", logits is not None and label_scores is not None)
        if logits is not None and label_scores is not None:
            check("OneLayerRio logits shape", logits.shape == (3, 2), f"got {tuple(logits.shape)}")
            check("OneLayerRio label shape", label_scores.shape == (3, 2), f"got {tuple(label_scores.shape)}")
            check("OneLayerRio logits finite", torch.isfinite(logits).all().item())
            probs, label_probs = model.to_prob(nodes, labels, train_flag=False)
            check("OneLayerRio prob range", torch.all((probs >= 0) & (probs <= 1)).item())
            loss = model.loss(nodes, labels, train_flag=False)
            check("OneLayerRio loss scalar", loss.dim() == 0)
            check("OneLayerRio loss finite", torch.isfinite(loss).item())
            loss.backward()
            check("OneLayerRio classifier gradient", model.weight.grad is not None and torch.isfinite(model.weight.grad).all().item())
        else:
            skip_checks(7, "OneLayerRio returned None")
    except Exception as exc:
        skip_checks(8, f"OneLayerRio raised {type(exc).__name__}: {exc}")
    print()

    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed == 0:
        print("All tests PASSED! The model code is complete and correct.")
    else:
        print(f"{failed} check(s) FAILED - some [TODO] functions may not be implemented correctly.")
        raise SystemExit(1)
