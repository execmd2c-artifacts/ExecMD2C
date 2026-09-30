# ============================================================
# ground_truth.py - MESA Core Meta-Sampler Components
# Source:
#   Miscellaneous/mesa-master/utils.py
#   Miscellaneous/mesa-master/environment.py
#   Miscellaneous/mesa-master/sac_src/model.py
#   Miscellaneous/mesa-master/sac_src/utils.py
#   Miscellaneous/mesa-master/sac_src/replay_memory.py
#   Miscellaneous/mesa-master/sac_src/sac.py
#
# Contains ONLY core algorithm/model components and direct dependencies.
# No CLI, dataset loading, experiment baselines, checkpoint I/O, or training scripts.
# ============================================================

# -*- coding: utf-8 -*-

import math
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score,
    average_precision_score,
    matthews_corrcoef,
)
from sklearn.model_selection import train_test_split
from torch.distributions import Normal
from torch.optim import Adam, lr_scheduler


# --- [Original file: utils.py] ---
class Rater():
    """Rater for evaluate classifiers performance on class imabalanced data.

    Parameters
    ----------
    metric :    {'aucprc', 'mcc', 'fscore'}, optional (default='aucprc')
        Specify the performance metric used for evaluation.
        If 'aucprc' then use Area Under Precision-Recall Curve.
        If 'mcc' then use Matthews Correlation Coefficient.
        If 'fscore' then use F1-score, also known as balanced F-score or F-measure.
        Passing other values raises an exception.

    threshold : float, optional (default=0.5)
        The threshold used for binarizing the predicted probability.
        It does not affect the AUCPRC score
    """
    def __init__(self, metric='aucprc', threshold=0.5):

        if metric not in ['aucprc', 'mcc', 'fscore', 'bacc']:
            raise ValueError(f'Metric {metric} is not supported.\
                \nSupport metrics: [aucprc, mcc, fscore].')

        self.metric_ = metric
        self.threshold_ = threshold
        
    def score(self, y_true, y_pred):
        """Score function."""
        if self.metric_ == 'aucprc':
            return average_precision_score(y_true , y_pred)
        elif self.metric_ == 'mcc':
            y_pred_b = y_pred.copy()
            y_pred_b[y_pred_b < self.threshold_] = 0
            y_pred_b[y_pred_b >= self.threshold_] = 1
            return matthews_corrcoef(y_true, y_pred_b)
        elif self.metric_ == 'fscore':
            y_pred_b = y_pred.copy()
            y_pred_b[y_pred_b < self.threshold_] = 0
            y_pred_b[y_pred_b >= self.threshold_] = 1
            return f1_score(y_true, y_pred_b)


def histogram_error_distribution(y_true, y_pred, bins):
    """Util function that compute the error histogram."""
    error = np.absolute(y_true - y_pred)
    hist, _ = np.histogram(error, bins=bins)
    return hist


def gaussian_prob(x, mu, sigma):
    """The Gaussian function."""
    return (1 / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*np.power((x-mu)/sigma, 2))


def meta_sampling(y_pred, y_true, X, n_under_samples, mu, sigma, random_state=None):
    """
    [TODO] Perform MESA's action-conditioned majority-class meta-sampling.

    Input:
        y_pred: array-like with shape (n_samples,), containing current ensemble
            predicted probabilities for candidate majority-class samples.
        y_true: array-like with shape (n_samples,), containing ground-truth labels
            for the same candidates.
        X: array-like or DataFrame with shape (n_samples, n_features), containing
            candidate majority-class features.
        n_under_samples: integer number of samples to draw.
        mu: scalar action in [0, 1] selected by the meta-sampler.
        sigma: positive scalar controlling the sampling concentration around mu.
        random_state: optional sampling seed.

    Output:
        X_subset: pandas DataFrame with shape (n_under_samples, n_features).

"""
    pass


def imbalance_train_test_split(X, y, test_size, random_state=None):
    '''Train/Test split that guarantee same class distribution between split datasets.'''
    classes = np.unique(y)
    X_trains, y_trains, X_tests, y_tests = [], [], [], []
    for label in classes:
        inds = (y==label)
        X_label, y_label = X[inds], y[inds]
        X_train, X_test, y_train, y_test = train_test_split(
            X_label, y_label, test_size=test_size, random_state=random_state)
        X_trains.append(X_train)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_tests.append(y_test)
    X_train = np.concatenate(X_trains)
    X_test = np.concatenate(X_tests)
    y_train = np.concatenate(y_trains)
    y_test = np.concatenate(y_tests)
    return  X_train, X_test, y_train, y_test


def state_scale(state, scale):
    '''Scale up the meta-states.'''
    return state / state.sum() * 2 * scale


def memory_init_fulfill(args, memory):
    '''Initialize the memory.'''
    num_bins = args.num_bins
    memory_size = args.replay_size
    error_in_bins = np.linspace(0, 1, num_bins)
    mu = 0.3
    unfitted, midfitted, fitted = \
        gaussian_prob(error_in_bins, 1, mu), \
        gaussian_prob(error_in_bins, 0.5, mu), \
        gaussian_prob(error_in_bins, 0, mu)
    underfitting_state = state_scale(np.concatenate([unfitted, unfitted]), num_bins)
    learning_state = state_scale(np.concatenate([midfitted, midfitted]), num_bins)
    overfitting_state = state_scale(np.concatenate([fitted, midfitted]), num_bins)
    noise_scale = 0.5
    num_per_transitions = int(memory_size/3)
    for i in range(num_per_transitions):
        state = underfitting_state + np.random.rand(num_bins*2) * noise_scale
        next_state = underfitting_state + np.random.rand(num_bins*2) * noise_scale
        memory.push(state, [0.9], args.reward_coefficient * 0.05, next_state, 0)
    for i in range(num_per_transitions):
        state = learning_state + np.random.rand(num_bins*2) * noise_scale
        next_state = learning_state + np.random.rand(num_bins*2) * noise_scale
        memory.push(state, [0.5], args.reward_coefficient * 0.05, next_state, 0)
    for i in range(num_per_transitions):
        state = overfitting_state + np.random.rand(num_bins*2) * noise_scale
        next_state = overfitting_state + np.random.rand(num_bins*2) * noise_scale
        memory.push(state, [0.1], args.reward_coefficient * 0.05, next_state, 0)
    return memory


# --- [Original file: environment.py] ---
class Ensemble():
    """A basic ensemble learning framework."""
    def __init__(self, base_estimator):
        self.estimators_ = []
        if not sklearn.base.is_classifier(base_estimator):
            raise TypeError(f'Base estimator {base_estimator} is not a sklearn classifier.')
        self.base_estimator_ = base_estimator

    def fit_step(self, X, y):
        """Bulid a new base classifier from the training set (X, y)."""
        self.estimators_.append(
            sklearn.base.clone(self.base_estimator_).fit(X, y)
            )
        return self

    def predict_proba(self, X):
        """Predict class probabilities for X."""
        y_pred = np.array(
            [model.predict_proba(X)[:, 1] for model in self.estimators_]
            ).mean(axis=0)
        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]
        if y_pred.shape[1] == 1:
            y_pred = np.append(1-y_pred, y_pred, axis=1)
        return y_pred
    
    def predict(self, X):
        """Predict classes for X."""
        y_pred_binarized = sklearn.preprocessing.binarize(
            self.predict_proba(X)[:,1].reshape(1,-1), threshold=0.5)[0]
        return y_pred_binarized
    
    def score(self, X, y):
        """Return area under precision recall curve (AUCPRC) scores for X, y."""
        yield sklearn.metrics.average_precision_score(
            y, self.predict_proba(X)[:, 1])


class EnsembleTrainingEnv(Ensemble):
    """The ensemble training environment in MESA."""
    def __init__(self, args, base_estimator):

        super(EnsembleTrainingEnv, self).__init__(
            base_estimator=base_estimator)

        self.base_estimator_ = base_estimator
        self.args = args
        self.rater = Rater(metric=args.metric)
    
    def load_data(self, X_train, y_train, X_valid, y_valid, X_test=None, y_test=None, train_ratio=1):
        """Load and preprocess the train/valid/test data into the environment."""
        self.flag_use_test_set = False if X_test is None or y_test is None else True
        if train_ratio < 1:
            print ('Using {:.2%} random subset for meta-training.'.format(train_ratio))
            _, X_train, _, y_train = imbalance_train_test_split(X_train, y_train, test_size=train_ratio)
        self.X_train, self.y_train = pd.DataFrame(X_train), pd.Series(y_train)
        self.X_valid, self.y_valid = pd.DataFrame(X_valid), pd.Series(y_valid)
        self.X_test,  self.y_test  = pd.DataFrame(X_test),  pd.Series(y_test)
        self.mask_maj_train, self.mask_min_train = (y_train==0), (y_train==1)
        self.mask_maj_valid, self.mask_min_valid = (y_valid==0), (y_valid==1)
        self.n_min_samples = self.mask_min_train.sum()
        n_samples = int(self.n_min_samples*self.args.train_ir)
        if n_samples > self.mask_maj_train.sum():
            raise ValueError(f"\
                Argument 'train_ir' should be smaller than imbalance ratio,\n \
                Please set this parameter to < {self.mask_maj_train.sum()/self.mask_min_train.sum()}.\
                ")
        self.n_samples = n_samples

    def init(self):
        """Reset the environment."""
        self.estimators_ = []
        # buffer the predict probabilities for better efficiency
        # initialize 
        self.y_pred_train_buffer = np.zeros_like(self.y_train)
        self.y_pred_valid_buffer = np.zeros_like(self.y_valid)
        if self.flag_use_test_set:
            self.y_pred_test_buffer  = np.zeros_like(self.y_test)
        self._warm_up()
    
    def get_state(self):
        """
        [TODO] Build the MESA meta-state from train and validation error distributions.

        Input:
            Uses the environment's buffered predictions and labels:
            - majority-class training labels/predictions with shape
              (n_majority_train,).
            - majority-class validation labels/predictions with shape
              (n_majority_valid,).

        Output:
            state: numpy array with shape (2 * num_bins,), containing the
                normalized training error histogram followed by the normalized
                validation error histogram.

"""
        pass
    
    def step(self, action, verbose=False):
        """
        [TODO] Execute one MESA environment transition for an action-selected sampler.

        Input:
            action: scalar float in [0, 1], used as the meta-sampling center.
            verbose: bool controlling whether a progress string is returned.

        Output:
            next_state: numpy array with shape (2 * num_bins,).
            reward: scalar validation-score improvement from before to after
                fitting the new base estimator.
            done: bool indicating whether the ensemble reached max_estimators.
            info: string with optional progress information.

"""
        pass
        
    def update_all_pred_buffer(self):
        """Update all buffered predict probabilities."""
        n_clf = len(self.estimators_)
        self.y_pred_train_buffer = self._update_pred_buffer(n_clf, self.X_train, self.y_pred_train_buffer)
        self.y_pred_valid_buffer = self._update_pred_buffer(n_clf, self.X_valid, self.y_pred_valid_buffer)
        if self.flag_use_test_set:
            self.y_pred_test_buffer  = self._update_pred_buffer(n_clf, self.X_test,  self.y_pred_test_buffer)
        return

    def _update_pred_buffer(self, n_clf, X, y_pred_buffer):
        """Update buffered predict probabilities."""
        y_pred_last_clf = self.estimators_[-1].predict_proba(X)[:, 1]
        y_pred_buffer_updated = (y_pred_buffer * (n_clf-1) + y_pred_last_clf) / n_clf
        return y_pred_buffer_updated
    
    def _warm_up(self):
        """Train the first base classifier with random under-sampling."""
        X_maj = self.X_train[self.mask_maj_train]
        X_min = self.X_train[self.mask_min_train]
        X_maj_rus = X_maj.sample(n=self.n_samples, random_state=self.args.random_state)
        # X_maj_rus = X_maj
        X_train_rus = pd.concat([X_maj_rus, X_min]).values
        y_train_rus = np.concatenate([np.zeros(X_maj_rus.shape[0]), np.ones(X_min.shape[0])])
        self.fit_step(X_train_rus, y_train_rus)
        self.update_all_pred_buffer()
        return


# --- [Original file: sac_src/utils.py] ---
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


# --- [Original file: sac_src/replay_memory.py] ---
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# --- [Original file: sac_src/model.py] ---
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        # self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear4 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        # x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))
        # x = F.relu(self.linear4(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        """
        [TODO] Sample a bounded SAC action from the Gaussian meta-sampler policy.

        Input:
            state: Tensor with shape (batch, num_inputs), containing MESA
                meta-states.

        Output:
            action: Tensor with shape (batch, num_actions), rescaled to the
                action-space bounds.
            log_prob: Tensor with shape (batch, 1), containing corrected log
                probabilities for the sampled bounded actions.
            mean: Tensor with shape (batch, num_actions), containing the bounded
                deterministic policy mean.

"""
        pass

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)


# --- [Original file: sac_src/sac.py] ---
class SAC(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.action_space = action_space
        self.learning_rate = args.lr

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu") 

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = -dim(A) as given in the paper
            if self.automatic_entropy_tuning == True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        self.policy_scheduler = lr_scheduler.StepLR(self.critic_optim, step_size=args.lr_decay_steps, gamma=args.lr_decay_gamma)

    def learning_rate_decay(self, decay_ratio=0.5):
        self.learning_rate = self.learning_rate * decay_ratio
        self.critic_optim = Adam(self.critic.parameters(), lr=self.learning_rate)
        self.policy_optim = Adam(self.policy.parameters(), lr=self.learning_rate)

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        self.policy_scheduler.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()


# ============================================================
# __main__: Automated test suite for 4 ablated functions
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

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

    print("=" * 70)
    print("MESA: Boost Ensemble Imbalanced Learning with Meta-Sampler")
    print("Automated reproduction benchmark - 4 ablated functions")
    print("=" * 70)
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    # ==============================================================
    # Test 1/4: meta_sampling
    # ==============================================================
    print("-" * 60)
    print("[Test 1/4] meta_sampling - Gaussian error-centered majority under-sampling")
    try:
        X = pd.DataFrame({"f0": np.arange(6), "f1": np.arange(10, 16)})
        y_true = pd.Series(np.zeros(6))
        y_pred = pd.Series(np.array([0.02, 0.18, 0.31, 0.49, 0.73, 0.95]))
        subset = meta_sampling(y_pred=y_pred, y_true=y_true, X=X, n_under_samples=3, mu=0.5, sigma=0.12, random_state=7)
        check("meta_sampling output not None", subset is not None)
        if subset is not None:
            weights = gaussian_prob(np.absolute(y_true - y_pred), 0.5, 0.12)
            top_weight_index = int(np.argmax(weights))
            check("meta_sampling output shape", tuple(subset.shape) == (3, 2), f"expected (3, 2), got {tuple(subset.shape)}")
            check("meta_sampling returns pandas DataFrame", isinstance(subset, pd.DataFrame))
            check("meta_sampling uses rows from original X", set(subset.index).issubset(set(X.index)))
            check("meta_sampling favors errors near action mu", top_weight_index in set(subset.index))
        else:
            skip_checks(4, "meta_sampling returned None")
    except Exception as e:
        skip_checks(5, f"meta_sampling raised {type(e).__name__}: {e}")
    print()

    class ThresholdEstimator(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
        def fit(self, X, y):
            X = np.asarray(X)
            y = np.asarray(y)
            self.threshold_ = X[y == 1, 0].mean() if np.any(y == 1) else X[:, 0].mean()
            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, X):
            X = np.asarray(X)
            prob = 1.0 / (1.0 + np.exp(-(X[:, 0] - self.threshold_)))
            return np.vstack([1.0 - prob, prob]).T

    def build_env():
        args = SimpleNamespace(
            metric="aucprc",
            num_bins=4,
            train_ir=1,
            sigma=0.2,
            random_state=3,
            max_estimators=2,
            meta_verbose=0,
        )
        X_train = np.array([[0.0], [0.2], [0.4], [0.6], [1.2], [1.4]], dtype=float)
        y_train = np.array([0, 0, 0, 0, 1, 1])
        X_valid = np.array([[0.1], [0.3], [0.7], [1.1], [1.5]], dtype=float)
        y_valid = np.array([0, 0, 0, 1, 1])
        env = EnsembleTrainingEnv(args, ThresholdEstimator())
        env.load_data(X_train, y_train, X_valid, y_valid)
        env.init()
        return env

    # ==============================================================
    # Test 2/4: EnsembleTrainingEnv.get_state
    # ==============================================================
    print("-" * 60)
    print("[Test 2/4] EnsembleTrainingEnv.get_state - train/valid error-distribution state")
    try:
        env = build_env()
        state = env.get_state()
        check("get_state output not None", state is not None)
        if state is not None:
            train_hist = state[:env.args.num_bins]
            valid_hist = state[env.args.num_bins:]
            check("get_state output shape", tuple(state.shape) == (8,), f"expected (8,), got {tuple(state.shape)}")
            check("get_state output finite", np.isfinite(state).all())
            check("get_state train histogram normalized to num_bins", np.isclose(train_hist.sum(), env.args.num_bins))
            check("get_state valid histogram normalized to num_bins", np.isclose(valid_hist.sum(), env.args.num_bins))
        else:
            skip_checks(4, "EnsembleTrainingEnv.get_state returned None")
    except Exception as e:
        skip_checks(5, f"EnsembleTrainingEnv.get_state raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 3/4: EnsembleTrainingEnv.step
    # ==============================================================
    print("-" * 60)
    print("[Test 3/4] EnsembleTrainingEnv.step - action-driven ensemble transition")
    try:
        env = build_env()
        before_estimators = len(env.estimators_)
        before_valid = env.rater.score(env.y_valid, env.y_pred_valid_buffer)
        result = env.step(0.5, verbose=False)
        check("step output not None", result is not None)
        if result is not None:
            next_state, reward, done, info = result
            after_valid = env.rater.score(env.y_valid, env.y_pred_valid_buffer)
            check("step next_state shape", tuple(next_state.shape) == (8,), f"expected (8,), got {tuple(next_state.shape)}")
            check("step reward equals validation score delta", np.isclose(reward, after_valid - before_valid))
            check("step adds exactly one base estimator", len(env.estimators_) == before_estimators + 1)
            check("step done follows max_estimators", done is True)
            check("step returns empty info when not verbose", info == "")
        else:
            skip_checks(5, "EnsembleTrainingEnv.step returned None")
    except Exception as e:
        skip_checks(6, f"EnsembleTrainingEnv.step raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 4/4: GaussianPolicy.sample
    # ==============================================================
    print("-" * 60)
    print("[Test 4/4] GaussianPolicy.sample - squashed SAC action with log-prob correction")
    try:
        action_space = SimpleNamespace(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            shape=(1,),
        )
        policy = GaussianPolicy(num_inputs=8, num_actions=1, hidden_dim=16, action_space=action_space).to(device)
        state = torch.randn(4, 8, device=device, requires_grad=True)
        action, log_prob, mean = policy.sample(state)
        check("GaussianPolicy.sample output not None", action is not None and log_prob is not None and mean is not None)
        if action is not None and log_prob is not None and mean is not None:
            bounds_ok = action.min().item() >= -1e-6 and action.max().item() <= 1.0 + 1e-6
            check("GaussianPolicy.sample action shape", tuple(action.shape) == (4, 1), f"expected (4, 1), got {tuple(action.shape)}")
            check("GaussianPolicy.sample log_prob shape", tuple(log_prob.shape) == (4, 1), f"expected (4, 1), got {tuple(log_prob.shape)}")
            check("GaussianPolicy.sample outputs finite", torch.isfinite(action).all().item() and torch.isfinite(log_prob).all().item() and torch.isfinite(mean).all().item())
            check("GaussianPolicy.sample respects action bounds", bounds_ok)
            (action.sum() + log_prob.sum()).backward()
            grad_ok = policy.linear1.weight.grad is not None and policy.linear1.weight.grad.abs().sum().item() > 0
            check("GaussianPolicy.sample keeps reparameterized gradient path", grad_ok)
        else:
            skip_checks(5, "GaussianPolicy.sample returned None")
    except Exception as e:
        skip_checks(6, f"GaussianPolicy.sample raised {type(e).__name__}: {e}")
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
