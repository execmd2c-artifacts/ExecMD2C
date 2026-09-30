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
        [TODO] Update relation-wise neighbor filtering thresholds with the RL forest.

        Input:
            scores: list[num_relations][batch] - neighbor distance lists produced by intra-relation filtering.
            labels: (batch,) - node labels used to identify positive nodes when constructing relation states.
            previous_thresholds: list[num_relations] - thresholds used by the previous epoch or batch.
            batch_num: int - number of batches that define one epoch for RL updates.
            auc: float - validation/training AUC associated with the previous thresholds.

        Output:
            new_thresholds: list[num_relations] - updated filtering thresholds for all relations.
            rl_flag: bool - True while at least one relation tree should keep exploring.

"""
        pass

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
        [TODO] Map an actor-selected discrete action to the current relation threshold interval.

        Input:
            new_states: list[num_relations] of (1,) arrays - current RL states for every relation.
            r_num: int - relation index whose actor should choose an action.

        Output:
            new_actions: int - selected action index within the current depth action space.
            new_thresholds: float - threshold value produced for this relation.

"""
        pass

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
    [TODO] Build RL state scores from positive nodes only.

    Input:
        scores: list[num_relations][batch] - each batch entry is a scalar or list of sampled neighbor distances.
        labels: (batch,) - binary labels for the current batch.

    Output:
        relation_scores: list[num_relations] - average sampled-neighbor distance for positive nodes in each relation.

"""
    pass


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
        [TODO] Run RioGNN inter-relation aggregation for a batch of nodes.

        Input:
            nodes: list[batch] - node ids in the current mini-batch.
            labels: (batch,) - node labels used by the RL threshold module during training.
            train_flag: bool - whether to allow logging and RL threshold updates.

        Output:
            combined: (embed_dim, batch) - final multi-relation node embeddings.
            center_scores: (batch, 2) - label-aware logits for the center nodes.

"""
        pass


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
        [TODO] Aggregate one relation's sampled neighbors after adaptive filtering.

        Input:
            nodes: list[batch] - center node ids.
            to_neighs_list: list[batch][num_neighbors] - candidate neighbor ids for one relation.
            batch_scores: (batch, 2) - label-aware logits for center nodes.
            neigh_scores: list[batch] of (num_neighbors, 2) - label-aware logits for candidate neighbors.
            sample_list: list[batch] - number of neighbors to keep per center node.

        Output:
            to_feats: (batch, feat_dim) - averaged and activated neighbor features.
            samp_scores: list[batch] - selected neighbor distance values used by the RL state.

"""
        pass


def filter_neighs_ada_threshold(center_scores, neigh_scores, neighs_list, sample_list):
    """
    [TODO] Select neighbors with the smallest label-aware distance under adaptive thresholds.

    Input:
        center_scores: (batch, 2) - label-aware logits for center nodes.
        neigh_scores: list[batch] of (num_neighbors, 2) - label-aware logits for each candidate neighbor.
        neighs_list: list[batch][num_neighbors] - candidate neighbor ids aligned with neigh_scores.
        sample_list: list[batch] - number of neighbors to retain for each center node.

    Output:
        samp_neighs: list[batch] of sets - selected neighbor ids for each center node.
        samp_scores: list[batch] - selected L1 distance values used by the RL state.

"""
    pass


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
    [TODO] Aggregate relation-specific neighbor embeddings using RioGNN thresholds as weights.

    Input:
        num_relations: int - number of relation-specific neighbor embedding blocks.
        self_feats: (batch, feat_dim) - center node feature vectors.
        neigh_feats: (num_relations * batch, feat_dim) - concatenated intra-relation neighbor embeddings.
        embed_dim: int - output embedding dimension.
        weight: (embed_dim, feat_dim) - shared projection for center and neighbor features.
        threshold: list[num_relations] - relation filtering thresholds reused as aggregation weights.
        n: int - batch size.
        cuda: bool - whether to allocate intermediate tensors on GPU.

    Output:
        combined: (embed_dim, batch) - activated center-plus-threshold-weighted relation embedding.

"""
    pass


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
