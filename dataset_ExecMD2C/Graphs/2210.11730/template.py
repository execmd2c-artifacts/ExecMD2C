import types

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch_geometric.nn.conv.gated_graph_conv import GatedGraphConv
from torch_geometric.nn.dense.dense_gcn_conv import DenseGCNConv
from torch_geometric.nn.dense.dense_gin_conv import DenseGINConv
from torch_geometric.nn.dense.dense_sage_conv import DenseSAGEConv
from torch_geometric.utils import dense_to_sparse


class DenseGGNN(nn.Module):
    def __init__(self, out_channels, num_layers=1):
        super(DenseGGNN, self).__init__()
        self.model = GatedGraphConv(out_channels=out_channels, num_layers=num_layers)
    
    def forward(self, x, adj, **kwargs):
        B = x.size()[0]
        N = x.size()[1]
        D = x.size()[2]
        indices = []
        for i in range(B):
            edge_index = dense_to_sparse(adj[i])
            indices.append(edge_index[0] + i * N)
        edge_index = torch.cat(indices, dim=1)
        x = x.reshape(-1, D)
        output = self.model(x, edge_index)
        return output.reshape(B, N, -1)


class MessageExtraction(torch.nn.Module):
    def __init__(self, n_queries, n_heads, hidden_size, inner_dim=1024, layer_norm_eps=1e-5):
        super().__init__()

        self.queries = nn.Parameter(torch.rand(n_queries, hidden_size))
        self.mhattn = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
        self.ffn_dropout = nn.Dropout(0.1)
        self.ffn_dropout2 = nn.Dropout(0.1)
        self.linear1 = nn.Linear(hidden_size, inner_dim)
        self.linear2 = nn.Linear(inner_dim, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def _ff_block(self, x):
        x = self.linear2(self.ffn_dropout(functional.relu(self.linear1(x))))
        return self.ffn_dropout2(x)

    def forward(self, x, outside_queries=None):
        """
        [TODO] Extract graph-level messages with learnable or external queries.

        Input:
            x: (batch, num_nodes, hidden_size) node embeddings.
            outside_queries: optional (batch, num_queries, hidden_size) query tensor
                supplied by the paired graph.

        Output: (batch, query_count, hidden_size) message embeddings, where
            query_count is self.queries length when outside_queries is None and
            outside_queries length otherwise.

"""
        pass


class PPGM(torch.nn.Module):
    @staticmethod
    def add_model_configs(parser):
        parser.add_argument("--filters", type=str, default='100_100_100', help="filters (neurons) for graph neural networks")
        parser.add_argument("--conv", type=str, default='gcn', help="one kind of graph neural networks")
        parser.add_argument("--hidden_size", type=int, default=100, help='hidden size for the graph-level embedding')
        parser.add_argument("--n_queries", type=int, default=8, help='number of learnable queries')
        parser.add_argument("--n_heads", type=int, default=4, help='number of heads')

        # global-level information
        parser.add_argument("--global_flag", type=lambda x: (str(x).lower() == 'true'), default='True', help="Whether use global info ")
        parser.add_argument("--global_agg", type=str, default='lstm', help="aggregation function for global level gcn ")
        return parser

    @staticmethod
    def log_name(args):
        return f'{args.global_agg}_{args.n_queries}_{args.n_heads}'

    def __init__(self, node_init_dims, arguments, device):
        super(PPGM, self).__init__()
        
        self.node_init_dims = node_init_dims
        self.args = arguments
        self.device = device
        self.hidden_size = self.args.hidden_size
        
        self.dropout = arguments.dropout
        
        # ---------- Node Embedding Layer ----------
        filters = self.args.filters.split('_')
        self.gcn_filters = [int(n_filter) for n_filter in filters]  # GCNs' filter sizes
        self.gcn_numbers = len(self.gcn_filters)
        self.gcn_last_filter = self.gcn_filters[-1]  # last filter size of node embedding layer
        
        gcn_parameters = [dict(in_channels=self.gcn_filters[i - 1], out_channels=self.gcn_filters[i], bias=True) for i in range(1, self.gcn_numbers)]
        gcn_parameters.insert(0, dict(in_channels=node_init_dims, out_channels=self.gcn_filters[0], bias=True))
        
        gin_parameters = [dict(nn=nn.Linear(in_features=self.gcn_filters[i - 1], out_features=self.gcn_filters[i])) for i in range(1, self.gcn_numbers)]
        gin_parameters.insert(0, {'nn': nn.Linear(in_features=node_init_dims, out_features=self.gcn_filters[0])})
        
        ggnn_parameters = [dict(out_channels=self.gcn_filters[i]) for i in range(self.gcn_numbers)]
        
        conv_layer_constructor = {
            'gcn': dict(constructor=DenseGCNConv, kwargs=gcn_parameters),
            'graphsage': dict(constructor=DenseSAGEConv, kwargs=gcn_parameters),
            'gin': dict(constructor=DenseGINConv, kwargs=gin_parameters),
            'ggnn': dict(constructor=DenseGGNN, kwargs=ggnn_parameters)
        }
        
        conv = conv_layer_constructor[self.args.conv]
        constructor = conv['constructor']
        # build GCN layers
        setattr(self, 'gc{}'.format(1), constructor(**conv['kwargs'][0]))
        for i in range(1, self.gcn_numbers):
            setattr(self, 'gc{}'.format(i + 1), constructor(**conv['kwargs'][i]))
        
        # Learnable queries
        self.n_queries = self.args.n_queries
        self.message_extractor = MessageExtraction(
            n_queries=self.n_queries,
            n_heads=self.args.n_heads,
            hidden_size=self.gcn_last_filter
        )

        self.indevice_message_extractor = MessageExtraction(
            n_queries=self.n_queries,
            n_heads=self.args.n_heads,
            hidden_size=self.gcn_last_filter
        )

        # global aggregation
        self.global_flag = self.args.global_flag
        if self.global_flag is True:
            self.global_agg = self.args.global_agg
            if self.global_agg.lower() == 'max_pool':
                print("Only Max Pooling")
            elif self.global_agg.lower() == 'fc_max_pool':
                self.global_fc_agg = nn.Linear(2 * self.gcn_last_filter, self.gcn_last_filter)
            elif self.global_agg.lower() == 'mean_pool':
                print("Only Mean Pooling")
            elif self.global_agg.lower() == 'fc_mean_pool':
                self.global_fc_agg = nn.Linear(2 * self.gcn_last_filter, self.gcn_last_filter)
            elif self.global_agg.lower() == 'lstm':
                self.global_lstm_agg = nn.LSTM(input_size=2 * self.gcn_last_filter, hidden_size=self.gcn_last_filter, num_layers=1, bidirectional=True, batch_first=True)
            else:
                raise NotImplementedError
        
        # ---------- Prediction Layer ----------
        if self.args.task.lower() == 'regression':
            factor = 2
            self.predict_fc1 = nn.Linear(int(self.hidden_size * 2 * factor), int(self.hidden_size * factor))
            self.predict_fc2 = nn.Linear(int(self.hidden_size * factor), int((self.hidden_size * factor) / 2))
            self.predict_fc3 = nn.Linear(int((self.hidden_size * factor) / 2), int((self.hidden_size * factor) / 4))
            self.predict_fc4 = nn.Linear(int((self.hidden_size * factor) / 4), 1)
        elif self.args.task.lower() in ['classification', 'cls_attr_inf']:
            print("classification task")
        else:
            raise NotImplementedError

    def global_aggregation_info(self, v, agg_func_name):
        """
        [TODO] Aggregate query-level paired graph information into one graph vector.

        Input:
            v: (batch, num_queries, 2 * hidden_size) tensor containing in-device
                features concatenated with paired-graph messages.
            agg_func_name: string selecting the aggregation branch.

        Output:
            (batch, 2 * hidden_size) for the recurrent branch, max-pool branch, and
            mean-pool branch used by the benchmark; projection branches may reduce
            after their learned transform as configured by the model.

"""
        pass

    @staticmethod
    def div_with_small_value(n, d, eps=1e-8):
        # too small values are replaced by 1e-8 to prevent it from exploding.
        d = d * (d > eps).float() + eps * (d <= eps).float()
        return n / d

    def cosine_attention(self, v1, v2):
        """
        :param v1: (batch, len1, dim)
        :param v2: (batch, len2, dim)
        :return:  (batch, len1, len2)
        """
        # (batch, len1, len2)
        a = torch.bmm(v1, v2.permute(0, 2, 1))
        
        v1_norm = v1.norm(p=2, dim=2, keepdim=True)  # (batch, len1, 1)
        v2_norm = v2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1)  # (batch, len2, 1)
        d = v1_norm * v2_norm
        return self.div_with_small_value(a, d)

    def forward_dense_gcn_layers(self, feat, adj):
        feat_in = feat
        for i in range(1, self.gcn_numbers + 1):
            feat_out = functional.relu(getattr(self, 'gc{}'.format(i))(x=feat_in, adj=adj, mask=None, add_loop=False), inplace=True)
            feat_out = functional.dropout(feat_out, p=self.dropout, training=self.training)
            feat_in = feat_out
        return feat_out

    def forward(self, batch_x_p, batch_x_h, batch_adj_p, batch_adj_h, return_mes=False):
        """
        [TODO] Run PPGM on a pair of graphs and produce similarity or messages.

        Input:
            batch_x_p: (batch, num_nodes_p, node_init_dims) first graph features.
            batch_x_h: (batch, num_nodes_h, node_init_dims) second graph features.
            batch_adj_p: (batch, num_nodes_p, num_nodes_p) dense adjacency matrices.
            batch_adj_h: (batch, num_nodes_h, num_nodes_h) dense adjacency matrices.
            return_mes: boolean flag for exposing intermediate privacy messages.

        Output:
            If return_mes is False and task is classification: (batch,) cosine
            similarity scores clamped to [-1, 1].
            If return_mes is False and task is regression: (batch,) sigmoid scores.
            If return_mes is True: a two-item list containing the aggregated first
            graph representation and the first graph message tensor.

"""
        pass


if __name__ == "__main__":
    torch.manual_seed(42)
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

    def make_args(task="classification", global_agg="lstm"):
        return types.SimpleNamespace(
            filters="8_8",
            conv="gcn",
            hidden_size=8,
            n_queries=3,
            n_heads=2,
            global_flag=True,
            global_agg=global_agg,
            task=task,
            dropout=0.0,
        )

    def make_graph_pair(batch=2, len_p=4, len_h=5, init_dim=6):
        batch_x_p = torch.randn(batch, len_p, init_dim).numpy()
        batch_x_h = torch.randn(batch, len_h, init_dim).numpy()
        batch_adj_p = torch.eye(len_p).unsqueeze(0).repeat(batch, 1, 1).numpy()
        batch_adj_h = torch.eye(len_h).unsqueeze(0).repeat(batch, 1, 1).numpy()
        batch_adj_p[:, 0, 1] = 1.0
        batch_adj_p[:, 1, 0] = 1.0
        batch_adj_h[:, 0, 1] = 1.0
        batch_adj_h[:, 1, 0] = 1.0
        return batch_x_p, batch_x_h, batch_adj_p, batch_adj_h

    # Test group 1: MessageExtraction.forward
    try:
        extractor = MessageExtraction(n_queries=3, n_heads=2, hidden_size=8, inner_dim=16)
        extractor.eval()
        x = torch.randn(2, 5, 8, requires_grad=True)
        message = extractor(x)
        check("MessageExtraction output not None", message is not None)
        if message is None:
            skip_checks(4, "MessageExtraction.forward returned None")
        else:
            check("MessageExtraction output shape", message.shape == (2, 3, 8), f"got {tuple(message.shape)}")
            check("MessageExtraction output finite", torch.isfinite(message).all().item())
            message.sum().backward()
            check("MessageExtraction query gradients", extractor.queries.grad is not None and torch.isfinite(extractor.queries.grad).all().item())
            outside = torch.randn(2, 4, 8)
            outside_message = extractor(x.detach(), outside_queries=outside)
            check("MessageExtraction outside query length", outside_message.shape == (2, 4, 8), f"got {tuple(outside_message.shape)}")
    except Exception as exc:
        skip_checks(5, f"MessageExtraction.forward raised {type(exc).__name__}: {exc}")

    # Test group 2: PPGM.global_aggregation_info
    try:
        args = make_args(global_agg="lstm")
        model = PPGM(node_init_dims=6, arguments=args, device=torch.device("cpu"))
        model.eval()
        v = torch.randn(2, 3, 16)
        lstm_out = model.global_aggregation_info(v, "lstm")
        check("global aggregation output not None", lstm_out is not None)
        if lstm_out is None:
            skip_checks(4, "global_aggregation_info returned None")
        else:
            check("global aggregation lstm shape", lstm_out.shape == (2, 16), f"got {tuple(lstm_out.shape)}")
            check("global aggregation lstm finite", torch.isfinite(lstm_out).all().item())
            mean_out = model.global_aggregation_info(v, "mean_pool")
            max_out = model.global_aggregation_info(v, "max_pool")
            check("global aggregation mean branch", mean_out.shape == (2, 16) and torch.allclose(mean_out, torch.mean(v, dim=1)))
            check("global aggregation max branch", max_out.shape == (2, 16) and torch.allclose(max_out, torch.max(v, 1)[0]))
    except Exception as exc:
        skip_checks(5, f"PPGM.global_aggregation_info raised {type(exc).__name__}: {exc}")

    # Test group 3: PPGM.forward
    try:
        args = make_args(task="classification", global_agg="lstm")
        model = PPGM(node_init_dims=6, arguments=args, device=torch.device("cpu"))
        model.eval()
        batch_x_p, batch_x_h, batch_adj_p, batch_adj_h = make_graph_pair()
        output = model(batch_x_p, batch_x_h, batch_adj_p, batch_adj_h)
        check("PPGM.forward output not None", output is not None)
        if output is None:
            skip_checks(5, "PPGM.forward returned None")
        else:
            check("PPGM.forward classification shape", output.shape == (2,), f"got {tuple(output.shape)}")
            check("PPGM.forward classification finite", torch.isfinite(output).all().item())
            check("PPGM.forward cosine range", torch.ge(output, -1).all().item() and torch.le(output, 1).all().item())
            mes_list = model(batch_x_p, batch_x_h, batch_adj_p, batch_adj_h, return_mes=True)
            check("PPGM.forward return_mes length", isinstance(mes_list, list) and len(mes_list) == 2)
            check("PPGM.forward return_mes shapes", mes_list[0].shape == (2, 16) and mes_list[1].shape == (2, 3, 8))
    except Exception as exc:
        skip_checks(6, f"PPGM.forward raised {type(exc).__name__}: {exc}")

    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    if failed != 0:
        raise SystemExit(1)
