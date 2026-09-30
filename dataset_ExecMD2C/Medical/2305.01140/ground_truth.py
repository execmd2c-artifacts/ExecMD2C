"""
ground_truth.py for GeoLDM core model components.

Source-consolidated from:
- equivariant_diffusion/utils.py
- egnn/egnn_new.py
- egnn/models.py
- equivariant_diffusion/en_diffusion.py

Only the geometric equivariant neural network components, QM9 dynamics /
autoencoder wrappers, noise schedules, and core diffusion formulas are included.
Experiment drivers, data preparation, evaluation scripts, plotting utilities,
property predictors, chemistry toolkit utilities, saved-state I/O, and CLI code are
intentionally excluded.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- [Original file: equivariant_diffusion/utils.py] ---
def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(dim=-1)


def remove_mean(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    x = x - mean
    return x


def remove_mean_with_mask(x, node_mask):
    masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x


def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
    assert_correctly_masked(x, node_mask)
    largest_value = x.abs().max().item()
    error = torch.sum(x, dim=1, keepdim=True).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f'Mean is not zero, relative_error {rel_error}'


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().max().item() < 1e-4, \
        'Variables not masked properly.'


def sample_center_gravity_zero_gaussian_with_mask(size, device, node_mask):
    assert len(size) == 3
    x = torch.randn(size, device=device)

    x_masked = x * node_mask

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean_with_mask(x_masked, node_mask)
    return x_projected


def sample_gaussian_with_mask(size, device, node_mask):
    x = torch.randn(size, device=device)
    x_masked = x * node_mask
    return x_masked


# --- [Original file: egnn/egnn_new.py] ---
class GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(), attention=False):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, edge_attr, edge_mask):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        return out, agg

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h, mij


class EquivariantUpdate(nn.Module):
    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=1, act_fn=nn.SiLU(), tanh=False, coords_range=10.0):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, edge_mask):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        if self.tanh:
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
        if edge_mask is not None:
            trans = trans * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        coord = coord + agg
        return coord

    def forward(self, h, coord, edge_index, coord_diff, edge_attr=None, node_mask=None, edge_mask=None):
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask)
        if node_mask is not None:
            coord = coord * node_mask
        return coord


class EquivariantBlock(nn.Module):
    def __init__(self, hidden_nf, edge_feat_nf=2, device='cpu', act_fn=nn.SiLU(), n_layers=2, attention=True,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum'):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edge_feat_nf,
                                              act_fn=act_fn, attention=attention,
                                              normalization_factor=self.normalization_factor,
                                              aggregation_method=self.aggregation_method))
        self.add_module("gcl_equiv", EquivariantUpdate(hidden_nf, edges_in_d=edge_feat_nf, act_fn=nn.SiLU(), tanh=tanh,
                                                       coords_range=self.coords_range_layer,
                                                       normalization_factor=self.normalization_factor,
                                                       aggregation_method=self.aggregation_method))
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, edge_attr=None):
        # Edit Emiel: Remove velocity as input
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        edge_attr = torch.cat([distances, edge_attr], dim=1)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        x = self._modules["gcl_equiv"](h, x, edge_index, coord_diff, edge_attr, node_mask, edge_mask)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x


class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=3, attention=False,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 sin_embedding=False, normalization_factor=100, aggregation_method='sum'):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range/n_layers) if n_layers > 0 else float(coords_range)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("e_block_%d" % i, EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf, device=device,
                                                               act_fn=act_fn, n_layers=inv_sublayers,
                                                               attention=attention, norm_diff=norm_diff, tanh=tanh,
                                                               coords_range=coords_range, norm_constant=norm_constant,
                                                               sin_embedding=self.sin_embedding,
                                                               normalization_factor=self.normalization_factor,
                                                               aggregation_method=self.aggregation_method))
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        distances, _ = coord2diff(x, edge_index)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x = self._modules["e_block_%d" % i](h, x, edge_index, node_mask=node_mask, edge_mask=edge_mask, edge_attr=distances)

        # Important, the bias of the last linear might be non-zero
        h = self.embedding_out(h)
        if node_mask is not None:
            h = h * node_mask
        return h, x


class GNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, aggregation_method='sum', device='cpu',
                 act_fn=nn.SiLU(), n_layers=4, attention=False,
                 normalization_factor=1, out_node_nf=None):
        super(GNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        ### Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                edges_in_d=in_edge_nf, act_fn=act_fn,
                attention=attention))
        self.to(self.device)

    def forward(self, h, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h


class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies)/max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result


# --- [Original file: egnn/models.py] ---
class EGNN_dynamics_QM9(nn.Module):
    def __init__(self, in_node_nf, context_node_nf,
                 n_dims, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics', norm_constant=0,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum'):
        super().__init__()
        self.mode = mode
        if mode == 'egnn_dynamics':
            self.egnn = EGNN(
                in_node_nf=in_node_nf + context_node_nf, in_edge_nf=1,
                hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
                inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method)
            self.in_node_nf = in_node_nf
        elif mode == 'gnn_dynamics':
            self.gnn = GNN(
                in_node_nf=in_node_nf + context_node_nf + 3, in_edge_nf=0,
                hidden_nf=hidden_nf, out_node_nf=3 + in_node_nf, device=device,
                act_fn=act_fn, n_layers=n_layers, attention=attention,
                normalization_factor=normalization_factor, aggregation_method=aggregation_method)

        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims
        self._edges_dict = {}
        self.condition_time = condition_time

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)
        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, t, xh, node_mask, edge_mask, context):
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]
        node_mask = node_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)
        xh = xh.view(bs*n_nodes, -1).clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()
        if h_dims == 0:
            h = torch.ones(bs*n_nodes, 1).to(self.device)
        else:
            h = xh[:, self.n_dims:].clone()

        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            h = torch.cat([h, h_time], dim=1)

        if context is not None:
            # We're conditioning, awesome!
            context = context.view(bs*n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)

        if self.mode == 'egnn_dynamics':
            h_final, x_final = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
            vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case
        elif self.mode == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]

        else:
            raise Exception("Wrong mode %s" % self.mode)

        if context is not None:
            # Slice off context size:
            h_final = h_final[:, :-self.context_node_nf]

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        if h_dims == 0:
            return vel
        else:
            h_final = h_final.view(bs, n_nodes, -1)
            return torch.cat([vel, h_final], dim=2)

    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(device),
                         torch.LongTensor(cols).to(device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)


class EGNN_encoder_QM9(nn.Module):
    def __init__(self, in_node_nf, context_node_nf, out_node_nf,
                 n_dims, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 tanh=False, mode='egnn_dynamics', norm_constant=0,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum',
                 include_charges=True):
        '''
        :param in_node_nf: Number of invariant features for input nodes.'''
        super().__init__()

        include_charges = int(include_charges)
        num_classes = in_node_nf - include_charges

        self.mode = mode
        if mode == 'egnn_dynamics':
            self.egnn = EGNN(
                in_node_nf=in_node_nf + context_node_nf, out_node_nf=hidden_nf, 
                in_edge_nf=1, hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
                inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method)
            self.in_node_nf = in_node_nf
        elif mode == 'gnn_dynamics':
            self.gnn = GNN(
                in_node_nf=in_node_nf + context_node_nf + 3, out_node_nf=hidden_nf + 3, 
                in_edge_nf=0, hidden_nf=hidden_nf, device=device,
                act_fn=act_fn, n_layers=n_layers, attention=attention,
                normalization_factor=normalization_factor, aggregation_method=aggregation_method)
        
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, out_node_nf * 2 + 1))

        self.num_classes = num_classes
        self.include_charges = include_charges
        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims
        self._edges_dict = {}
        # self.condition_time = condition_time

        self.out_node_nf = out_node_nf

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)
        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, xh, node_mask, edge_mask, context):      
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]
        node_mask = node_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)
        xh = xh.view(bs*n_nodes, -1).clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()
        if h_dims == 0:
            h = torch.ones(bs*n_nodes, 1).to(self.device)
        else:
            h = xh[:, self.n_dims:].clone()

        if context is not None:
            # We're conditioning, awesome!
            context = context.view(bs*n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)

        if self.mode == 'egnn_dynamics':
            h_final, x_final = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
            vel = x_final * node_mask  # This masking operation is redundant but just in case
        elif self.mode == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]

        else:
            raise Exception("Wrong mode %s" % self.mode)

        vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        h_final = self.final_mlp(h_final)
        h_final = h_final * node_mask if node_mask is not None else h_final
        h_final = h_final.view(bs, n_nodes, -1)

        vel_mean = vel
        vel_std = h_final[:, :, :1].sum(dim=1, keepdim=True).expand(-1, n_nodes, -1)
        vel_std = torch.exp(0.5 * vel_std)

        h_mean = h_final[:, :, 1:1 + self.out_node_nf]
        h_std = torch.exp(0.5 * h_final[:, :, 1 + self.out_node_nf:])

        if torch.any(torch.isnan(vel_std)):
            print('Warning: detected nan in vel_std, resetting to one.')
            vel_std = torch.ones_like(vel_std)
        
        if torch.any(torch.isnan(h_std)):
            print('Warning: detected nan in h_std, resetting to one.')
            h_std = torch.ones_like(h_std)
        
        # Note: only vel_mean and h_mean are correctly masked
        # vel_std and h_std are not masked, but that's fine:

        # For calculating KL: vel_std will be squeezed to 1D
        # h_std will be masked

        # For sampling: both stds will be masked in reparameterization

        return vel_mean, vel_std, h_mean, h_std
    
    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(device),
                         torch.LongTensor(cols).to(device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)


class EGNN_decoder_QM9(nn.Module):
    def __init__(self, in_node_nf, context_node_nf, out_node_nf,
                 n_dims, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 tanh=False, mode='egnn_dynamics', norm_constant=0,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum',
                 include_charges=True):
        super().__init__()

        include_charges = int(include_charges)
        num_classes = out_node_nf - include_charges

        self.mode = mode
        if mode == 'egnn_dynamics':
            self.egnn = EGNN(
                in_node_nf=in_node_nf + context_node_nf, out_node_nf=out_node_nf, 
                in_edge_nf=1, hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
                inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method)
            self.in_node_nf = in_node_nf
        elif mode == 'gnn_dynamics':
            self.gnn = GNN(
                in_node_nf=in_node_nf + context_node_nf + 3, out_node_nf=out_node_nf + 3, 
                in_edge_nf=0, hidden_nf=hidden_nf, device=device,
                act_fn=act_fn, n_layers=n_layers, attention=attention,
                normalization_factor=normalization_factor, aggregation_method=aggregation_method)

        self.num_classes = num_classes
        self.include_charges = include_charges
        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims
        self._edges_dict = {}
        # self.condition_time = condition_time

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)
        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, xh, node_mask, edge_mask, context):
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]
        node_mask = node_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)
        xh = xh.view(bs*n_nodes, -1).clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()
        if h_dims == 0:
            h = torch.ones(bs*n_nodes, 1).to(self.device)
        else:
            h = xh[:, self.n_dims:].clone()

        if context is not None:
            # We're conditioning, awesome!
            context = context.view(bs*n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)

        if self.mode == 'egnn_dynamics':
            h_final, x_final = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
            vel = x_final * node_mask  # This masking operation is redundant but just in case
        elif self.mode == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]

        else:
            raise Exception("Wrong mode %s" % self.mode)

        vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        if node_mask is not None:
            h_final = h_final * node_mask
        h_final = h_final.view(bs, n_nodes, -1)

        return vel, h_final
    
    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(device),
                         torch.LongTensor(cols).to(device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)


# --- [Original file: equivariant_diffusion/en_diffusion.py] ---
def expm1(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x)


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power))**2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def gaussian_KL(q_mu, q_sigma, p_mu, p_sigma, node_mask):
    """Computes the KL distance between two normal distributions.

        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    return sum_except_batch(
            (
                torch.log(p_sigma / (q_sigma + 1e-8) + 1e-8)
                + 0.5 * (q_sigma**2 + (q_mu - p_mu)**2) / (p_sigma**2)
                - 0.5
            ) * node_mask
        )


def gaussian_KL_for_dimension(q_mu, q_sigma, p_mu, p_sigma, d):
    """Computes the KL distance between two normal distributions.

        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    mu_norm2 = sum_except_batch((q_mu - p_mu)**2)
    assert len(q_sigma.size()) == 1
    assert len(p_sigma.size()) == 1
    return (d * torch.log(p_sigma / (q_sigma + 1e-8) + 1e-8) 
            + 0.5 * (d * q_sigma**2 + mu_norm2) / (p_sigma**2) 
            - 0.5 * d
            )


class PositiveLinear(torch.nn.Module):
    """Linear layer with weights forced to be positive."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_init_offset: int = -2):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        positive_weight = softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.squeeze() * 1000
        assert len(x.shape) == 1
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """
    def __init__(self, noise_schedule, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            alphas2 = cosine_beta_schedule(timesteps)
        elif 'polynomial' in noise_schedule:
            splits = noise_schedule.split('_')
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        else:
            raise ValueError(noise_schedule)

        print('alphas2', alphas2)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        print('gamma', -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]


class GammaNetwork(torch.nn.Module):
    """The gamma network models a monotonic increasing function. Construction as in the VDM paper."""
    def __init__(self):
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        self.gamma_0 = torch.nn.Parameter(torch.tensor([-5.]))
        self.gamma_1 = torch.nn.Parameter(torch.tensor([10.]))

    def gamma_tilde(self, t):
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    def forward(self, t):
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        # Not super efficient.
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # Normalize to [0, 1]
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (
                gamma_tilde_1 - gamma_tilde_0)

        # Rescale to [gamma_0, gamma_1]
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma

        return gamma


def cdf_standard_gaussian(x):
    return 0.5 * (1. + torch.erf(x / math.sqrt(2)))


class EnVariationalDiffusion(torch.nn.Module):
    """
    The E(n) Diffusion Module.
    """
    def __init__(
            self,
            dynamics: EGNN_dynamics_QM9, in_node_nf: int, n_dims: int,
            timesteps: int = 1000, parametrization='eps', noise_schedule='learned',
            noise_precision=1e-4, loss_type='vlb', norm_values=(1., 1., 1.),
            norm_biases=(None, 0., 0.), include_charges=True):
        super().__init__()

        assert loss_type in {'vlb', 'l2'}
        self.loss_type = loss_type
        self.include_charges = include_charges
        if noise_schedule == 'learned':
            assert loss_type == 'vlb', 'A noise schedule can only be learned' \
                                       ' with a vlb objective.'

        # Only supported parametrization.
        assert parametrization == 'eps'

        if noise_schedule == 'learned':
            self.gamma = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps,
                                                 precision=noise_precision)

        # The network that will predict the denoising.
        self.dynamics = dynamics

        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.num_classes = self.in_node_nf - self.include_charges

        self.T = timesteps
        self.parametrization = parametrization

        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.register_buffer('buffer', torch.zeros(1))

    def phi(self, x, t, node_mask, edge_mask, context):
        net_out = self.dynamics._forward(t, x, node_mask, edge_mask, context)

        return net_out

    def inflate_batch_array(self, array, target):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
        axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)

    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)

    def SNR(self, gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)

    def subspace_dimensionality(self, node_mask):
        """Compute the dimensionality on translation-invariant linear subspace where distributions on x are defined."""
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        return (number_of_nodes - 1) * self.n_dims

    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_tensor: torch.Tensor):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = self.inflate_batch_array(
            -expm1(softplus(gamma_s) - softplus(gamma_t)), target_tensor
        )

        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(
            alpha_t_given_s, target_tensor)

        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    def compute_x_pred(self, net_out, zt, gamma_t):
        """Commputes x_pred, i.e. the most likely prediction of x."""
        if self.parametrization == 'x':
            x_pred = net_out
        elif self.parametrization == 'eps':
            sigma_t = self.sigma(gamma_t, target_tensor=net_out)
            alpha_t = self.alpha(gamma_t, target_tensor=net_out)
            eps_t = net_out
            x_pred = 1. / alpha_t * (zt - sigma_t * eps_t)
        else:
            raise ValueError(self.parametrization)

        return x_pred

    def compute_error(self, net_out, gamma_t, eps):
        """Computes error, i.e. the most likely prediction of x."""
        eps_t = net_out
        if self.training and self.loss_type == 'l2':
            denom = (self.n_dims + self.in_node_nf) * eps_t.shape[1]
            error = sum_except_batch((eps - eps_t) ** 2) / denom
        else:
            error = sum_except_batch((eps - eps_t) ** 2)
        return error

    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims), device=node_mask.device,
            node_mask=node_mask)
        z_h = sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.in_node_nf), device=node_mask.device,
            node_mask=node_mask)
        z = torch.cat([z_x, z_h], dim=2)
        return z


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

    def fully_connected_edges(batch_size, n_nodes, device="cpu"):
        rows, cols = [], []
        for b in range(batch_size):
            for i in range(n_nodes):
                for j in range(n_nodes):
                    rows.append(i + b * n_nodes)
                    cols.append(j + b * n_nodes)
        return [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]

    print("=" * 70)
    print("GeoLDM core model component benchmark")
    print("=" * 70)

    # ==========================================================
    # Test 1: coord2diff and unsorted_segment_sum
    # ==========================================================
    print("[Test 1/8] geometric differences and segment aggregation")
    try:
        x = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0]], dtype=torch.float)
        edges = [torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0])]
        radial, coord_diff = coord2diff(x, edges, norm_constant=1)
        data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        segment_ids = torch.tensor([0, 0, 1])
        agg_sum = unsorted_segment_sum(data, segment_ids, 2, normalization_factor=2, aggregation_method="sum")
        agg_mean = unsorted_segment_sum(data, segment_ids, 2, normalization_factor=1, aggregation_method="mean")
        check("radial shape", tuple(radial.shape) == (3, 1), f"got {tuple(radial.shape)}")
        check("coord_diff shape", tuple(coord_diff.shape) == (3, 3), f"got {tuple(coord_diff.shape)}")
        check("radial nonnegative", torch.all(radial >= 0).item())
        check("coord diff finite", torch.isfinite(coord_diff).all().item())
        check("sum aggregation shape", tuple(agg_sum.shape) == (2, 2), f"got {tuple(agg_sum.shape)}")
        check("sum aggregation normalized", torch.allclose(agg_sum[0], torch.tensor([2.0, 3.0])))
        check("mean aggregation value", torch.allclose(agg_mean[0], torch.tensor([2.0, 3.0])))
    except Exception as exc:
        skip_checks(7, f"geometry utilities raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 2: GCL message passing
    # ==========================================================
    print("[Test 2/8] GCL edge and node updates")
    try:
        gcl = GCL(input_nf=4, output_nf=4, hidden_nf=8, normalization_factor=1, aggregation_method="sum",
                  edges_in_d=1, attention=True)
        h = torch.randn(4, 4)
        edges = [torch.tensor([0, 0, 1, 2]), torch.tensor([1, 2, 2, 3])]
        edge_attr = torch.randn(4, 1)
        node_mask = torch.tensor([[1.0], [1.0], [1.0], [0.0]])
        edge_mask = torch.tensor([[1.0], [1.0], [1.0], [0.0]])
        edge_feat, mij = gcl.edge_model(h[edges[0]], h[edges[1]], edge_attr, edge_mask)
        out, mij_forward = gcl(h, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        check("edge model output not None", edge_feat is not None)
        check("edge model shape", tuple(edge_feat.shape) == (4, 8), f"got {tuple(edge_feat.shape)}")
        check("raw message shape", tuple(mij.shape) == (4, 8), f"got {tuple(mij.shape)}")
        check("masked edge zeroed", torch.allclose(edge_feat[3], torch.zeros_like(edge_feat[3]), atol=1e-6))
        check("GCL output shape", tuple(out.shape) == (4, 4), f"got {tuple(out.shape)}")
        check("GCL masked node zeroed", torch.allclose(out[3], torch.zeros_like(out[3]), atol=1e-6))
        check("GCL output finite", torch.isfinite(out).all().item())
    except Exception as exc:
        skip_checks(7, f"GCL raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 3: EquivariantUpdate and EquivariantBlock
    # ==========================================================
    print("[Test 3/8] equivariant coordinate update")
    try:
        h = torch.randn(4, 8)
        coord = torch.randn(4, 3)
        edges = [torch.tensor([0, 0, 1, 2]), torch.tensor([1, 2, 2, 3])]
        radial, coord_diff = coord2diff(coord, edges, norm_constant=1)
        edge_attr = torch.randn(4, 2)
        node_mask = torch.tensor([[1.0], [1.0], [1.0], [0.0]])
        edge_mask = torch.tensor([[1.0], [1.0], [1.0], [0.0]])
        updater = EquivariantUpdate(hidden_nf=8, normalization_factor=1, aggregation_method="sum", edges_in_d=2)
        coord_out = updater(h, coord, edges, coord_diff, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        block = EquivariantBlock(hidden_nf=8, edge_feat_nf=2, n_layers=1, attention=True,
                                 normalization_factor=1, aggregation_method="sum")
        h_out, x_out = block(h, coord, edges, node_mask=node_mask, edge_mask=edge_mask, edge_attr=radial)
        check("coord update shape", tuple(coord_out.shape) == (4, 3), f"got {tuple(coord_out.shape)}")
        check("coord update masked node zeroed", torch.allclose(coord_out[3], torch.zeros_like(coord_out[3]), atol=1e-6))
        check("block feature shape", tuple(h_out.shape) == (4, 8), f"got {tuple(h_out.shape)}")
        check("block coordinate shape", tuple(x_out.shape) == (4, 3), f"got {tuple(x_out.shape)}")
        check("block masked feature zeroed", torch.allclose(h_out[3], torch.zeros_like(h_out[3]), atol=1e-6))
        check("block output finite", torch.isfinite(h_out).all().item() and torch.isfinite(x_out).all().item())
    except Exception as exc:
        skip_checks(6, f"EquivariantUpdate/Block raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 4: EGNN forward
    # ==========================================================
    print("[Test 4/8] EGNN full equivariant stack")
    try:
        egnn = EGNN(in_node_nf=5, in_edge_nf=1, hidden_nf=8, out_node_nf=6, n_layers=1,
                    inv_sublayers=1, attention=True, normalization_factor=1, aggregation_method="sum")
        bs, n_nodes = 2, 3
        h = torch.randn(bs * n_nodes, 5)
        coord = torch.randn(bs * n_nodes, 3)
        edges = fully_connected_edges(bs, n_nodes)
        node_mask = torch.tensor([[[1.0], [1.0], [0.0]], [[1.0], [1.0], [1.0]]]).view(bs*n_nodes, 1)
        edge_mask = (node_mask[edges[0]] * node_mask[edges[1]]).view(bs*n_nodes*n_nodes, 1)
        h_out, x_out = egnn(h, coord, edges, node_mask=node_mask, edge_mask=edge_mask)
        check("EGNN feature output shape", tuple(h_out.shape) == (6, 6), f"got {tuple(h_out.shape)}")
        check("EGNN coordinate output shape", tuple(x_out.shape) == (6, 3), f"got {tuple(x_out.shape)}")
        check("EGNN masked feature zeroed", torch.allclose(h_out[2], torch.zeros_like(h_out[2]), atol=1e-6))
        check("EGNN output finite", torch.isfinite(h_out).all().item() and torch.isfinite(x_out).all().item())
    except Exception as exc:
        skip_checks(4, f"EGNN raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 5: EGNN_dynamics_QM9
    # ==========================================================
    print("[Test 5/8] QM9 denoising dynamics wrapper")
    try:
        bs, n_nodes, in_nf = 2, 3, 2
        dynamics = EGNN_dynamics_QM9(in_node_nf=in_nf + 1, context_node_nf=1, n_dims=3,
                                     hidden_nf=8, n_layers=1, inv_sublayers=1, attention=True,
                                     normalization_factor=1, aggregation_method="sum", device="cpu")
        xh = torch.randn(bs, n_nodes, 3 + in_nf)
        node_mask = torch.tensor([[[1.0], [1.0], [0.0]], [[1.0], [1.0], [1.0]]])
        xh = xh * node_mask
        edge_mask = (node_mask.view(bs, n_nodes, 1) * node_mask.view(bs, 1, n_nodes)).view(bs*n_nodes*n_nodes, 1)
        context = torch.randn(bs, n_nodes, 1) * node_mask
        t = torch.tensor([[0.25], [0.75]])
        out = dynamics._forward(t, xh, node_mask, edge_mask, context)
        check("dynamics output not None", out is not None)
        check("dynamics output shape", tuple(out.shape) == (bs, n_nodes, 3 + in_nf), f"got {tuple(out.shape)}")
        check("dynamics masked output zeroed", torch.allclose(out[0, 2], torch.zeros_like(out[0, 2]), atol=1e-5))
        check("dynamics velocity mean zero", torch.allclose((out[:, :, :3] * node_mask).sum(dim=1), torch.zeros(bs, 3), atol=1e-4))
        check("adjacency cache shape", dynamics.get_adj_matrix(n_nodes, bs, "cpu")[0].numel() == bs*n_nodes*n_nodes)
    except Exception as exc:
        skip_checks(5, f"EGNN_dynamics_QM9 raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 6: EGNN encoder and decoder wrappers
    # ==========================================================
    print("[Test 6/8] QM9 encoder and decoder wrappers")
    try:
        bs, n_nodes, in_nf, latent_nf = 2, 3, 4, 2
        node_mask = torch.tensor([[[1.0], [1.0], [0.0]], [[1.0], [1.0], [1.0]]])
        edge_mask = (node_mask.view(bs, n_nodes, 1) * node_mask.view(bs, 1, n_nodes)).view(bs*n_nodes*n_nodes, 1)
        xh = torch.randn(bs, n_nodes, 3 + in_nf) * node_mask
        encoder = EGNN_encoder_QM9(in_node_nf=in_nf, context_node_nf=0, out_node_nf=latent_nf,
                                   n_dims=3, hidden_nf=8, n_layers=1, inv_sublayers=1,
                                   normalization_factor=1, aggregation_method="sum", device="cpu")
        decoder = EGNN_decoder_QM9(in_node_nf=latent_nf, context_node_nf=0, out_node_nf=in_nf,
                                   n_dims=3, hidden_nf=8, n_layers=1, inv_sublayers=1,
                                   normalization_factor=1, aggregation_method="sum", device="cpu")
        vel_mean, vel_std, h_mean, h_std = encoder._forward(xh, node_mask, edge_mask, context=None)
        z_xh = torch.cat([vel_mean, h_mean], dim=2)
        dec_x, dec_h = decoder._forward(z_xh, node_mask, edge_mask, context=None)
        check("encoder vel mean shape", tuple(vel_mean.shape) == (bs, n_nodes, 3), f"got {tuple(vel_mean.shape)}")
        check("encoder vel std shape", tuple(vel_std.shape) == (bs, n_nodes, 1), f"got {tuple(vel_std.shape)}")
        check("encoder h mean shape", tuple(h_mean.shape) == (bs, n_nodes, latent_nf), f"got {tuple(h_mean.shape)}")
        check("encoder h std positive", torch.all(h_std > 0).item())
        check("decoder x shape", tuple(dec_x.shape) == (bs, n_nodes, 3), f"got {tuple(dec_x.shape)}")
        check("decoder h shape", tuple(dec_h.shape) == (bs, n_nodes, in_nf), f"got {tuple(dec_h.shape)}")
        check("decoder masked output zeroed", torch.allclose(dec_x[0, 2], torch.zeros_like(dec_x[0, 2]), atol=1e-5))
    except Exception as exc:
        skip_checks(7, f"encoder/decoder wrappers raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 7: noise schedules and gamma utilities
    # ==========================================================
    print("[Test 7/8] diffusion noise schedules")
    try:
        poly = polynomial_schedule(8, s=1e-4, power=2.0)
        cosine = cosine_beta_schedule(8)
        sched = PredefinedNoiseSchedule("polynomial_2", timesteps=8, precision=1e-4)
        t = torch.tensor([[0.0], [0.5], [1.0]])
        gamma = sched(t)
        pos = PositiveLinear(3, 2)
        pos_out = pos(torch.ones(4, 3))
        emb = SinusoidalPosEmb(6)(torch.tensor([0.1, 0.2]))
        check("polynomial length", poly.shape == (9,), f"got {poly.shape}")
        check("cosine length", cosine.shape == (9,), f"got {cosine.shape}")
        check("schedule gamma shape", tuple(gamma.shape) == (3, 1), f"got {tuple(gamma.shape)}")
        check("schedule gamma finite", torch.isfinite(gamma).all().item())
        check("positive linear output shape", tuple(pos_out.shape) == (4, 2), f"got {tuple(pos_out.shape)}")
        check("positive effective weights", torch.all(softplus(pos.weight) > 0).item())
        check("sinusoidal position shape", tuple(emb.shape) == (2, 6), f"got {tuple(emb.shape)}")
    except Exception as exc:
        skip_checks(7, f"noise schedule utilities raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 8: EnVariationalDiffusion formulas and noise sampling
    # ==========================================================
    print("[Test 8/8] diffusion posterior formulas and masked noise")
    try:
        dynamics = EGNN_dynamics_QM9(in_node_nf=3, context_node_nf=0, n_dims=3,
                                     hidden_nf=8, n_layers=1, inv_sublayers=1,
                                     normalization_factor=1, aggregation_method="sum", device="cpu")
        diffusion = EnVariationalDiffusion(dynamics=dynamics, in_node_nf=2, n_dims=3,
                                           timesteps=8, noise_schedule="polynomial_2",
                                           noise_precision=1e-4, loss_type="l2",
                                           norm_values=(1., 1., 1.), include_charges=False)
        bs, n_nodes = 2, 3
        target = torch.randn(bs, n_nodes, 5)
        gamma_s = torch.tensor([[0.1], [0.2]])
        gamma_t = torch.tensor([[0.5], [0.8]])
        sigma2, sigma, alpha = diffusion.sigma_and_alpha_t_given_s(gamma_t, gamma_s, target)
        net_out = torch.randn_like(target)
        zt = torch.randn_like(target)
        x_pred = diffusion.compute_x_pred(net_out, zt, gamma_t)
        err = diffusion.compute_error(net_out, gamma_t, torch.zeros_like(net_out))
        node_mask = torch.tensor([[[1.0], [1.0], [0.0]], [[1.0], [1.0], [1.0]]])
        noise = diffusion.sample_combined_position_feature_noise(bs, n_nodes, node_mask)
        check("sigma2 shape", tuple(sigma2.shape) == (bs, 1, 1), f"got {tuple(sigma2.shape)}")
        check("posterior sigma finite", torch.isfinite(sigma).all().item())
        check("posterior alpha finite", torch.isfinite(alpha).all().item())
        check("x prediction shape", tuple(x_pred.shape) == (bs, n_nodes, 5), f"got {tuple(x_pred.shape)}")
        check("error vector shape", tuple(err.shape) == (bs,), f"got {tuple(err.shape)}")
        check("combined noise shape", tuple(noise.shape) == (bs, n_nodes, 5), f"got {tuple(noise.shape)}")
        check("position noise mean zero", torch.allclose((noise[:, :, :3] * node_mask).sum(dim=1), torch.zeros(bs, 3), atol=1e-4))
        check("masked noise zeroed", torch.allclose(noise[0, 2], torch.zeros_like(noise[0, 2]), atol=1e-6))
    except Exception as exc:
        skip_checks(8, f"diffusion formulas raised {type(exc).__name__}: {exc}")
    print()

    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed == 0:
        print("All tests PASSED! The model code is complete and correct.")
    else:
        print(f"{failed} check(s) FAILED - some TODO functions may be incomplete.")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
