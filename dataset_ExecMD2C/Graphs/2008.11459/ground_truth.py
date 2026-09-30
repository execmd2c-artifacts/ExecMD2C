import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, cuda=0, idx=None, xyz=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if xyz:
            idx = knn(x[:, :3, :], k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda:' + str(cuda)) # 'cuda'

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature


class AttentionModule(torch.nn.Module):
    """
    SimGNN Attention Module to make a pass on graph.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(AttentionModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.filters_3, self.args.filters_3))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embedding: Result of the GCN.
        :return representation: A graph level representation vector.
        """
        batch_size = embedding.shape[0]
        global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=1) # 0 # nxf -> f  bxnxf->bxf
        transformed_global = torch.tanh(global_context) # f  bxf
        sigmoid_scores = torch.sigmoid(torch.matmul(embedding, transformed_global.view(batch_size, -1, 1)))   #weights      nxf fx1  bxnxf bxfx1 bxnx1
        representation = torch.matmul(embedding.permute(0, 2, 1), sigmoid_scores)    # bxnxf bxfxn bxnx1 bxfx1
        return representation, sigmoid_scores


class TenorNetworkModule(torch.nn.Module):
    """
    SimGNN Tensor Network module to calculate similarity vector.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(TenorNetworkModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.filters_3, self.args.filters_3, self.args.tensor_neurons))
        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 2 * self.args.filters_3))
        self.bias = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 1))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.    bxfx1
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """
        batch_size = embedding_1.shape[0]
        scoring = torch.matmul(embedding_1.permute(0, 2, 1), self.weight_matrix.view(self.args.filters_3, -1)).view(batch_size, self.args.filters_3, self.args.tensor_neurons)
        scoring = torch.matmul(scoring.permute(0, 2, 1), embedding_2) # bxfx1
        combined_representation = torch.cat((embedding_1, embedding_2), dim=1)  # bx2fx1
        block_scoring = torch.matmul(self.weight_matrix_block, combined_representation) # bxtensor_neuronsx1
        scores = torch.nn.functional.relu(scoring + block_scoring + self.bias)
        return scores


class SG(torch.nn.Module):
    """
    SimGNN: A Neural Network Approach to Fast Graph Similarity Computation
    https://arxiv.org/abs/1808.05689
    """

    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(SG, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        self.attention = AttentionModule(self.args)
        self.tensor_network = TenorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)
        bias_bool = False # TODO
        self.dgcnn_s_conv1 = nn.Sequential(
            nn.Conv2d(3 * 2, self.args.filters_1, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.args.filters_1),
            nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_f_conv1 = nn.Sequential(
            nn.Conv2d(self.number_labels * 2, self.args.filters_1, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.args.filters_1),
            nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_s_conv2 = nn.Sequential(
            nn.Conv2d(self.args.filters_1 * 2, self.args.filters_2, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.args.filters_2),
            nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_f_conv2 = nn.Sequential(
            nn.Conv2d(self.args.filters_1 * 2, self.args.filters_2, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.args.filters_2),
            nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_s_conv3 = nn.Sequential(
            nn.Conv2d(self.args.filters_2 * 2, self.args.filters_3, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.args.filters_3),
            nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_f_conv3 = nn.Sequential(
            nn.Conv2d(self.args.filters_2 * 2, self.args.filters_3, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.args.filters_3),
            nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv_end = nn.Sequential(nn.Conv1d(self.args.filters_3 * 2,
                                                      self.args.filters_3, kernel_size=1, bias=bias_bool),
                                            nn.BatchNorm1d(self.args.filters_3), nn.LeakyReLU(negative_slope=0.2))

    def dgcnn_conv_pass(self, x):
        self.k = self.args.K
        xyz = x[:, :3, :] # Bx3xN
        sem = x[:, 3:, :]   # BxfxN

        xyz = get_graph_feature(xyz, k=self.k, cuda=self.args.cuda)    #Bx6xNxk
        xyz = self.dgcnn_s_conv1(xyz)
        xyz1 = xyz.max(dim=-1, keepdim=False)[0]
        xyz = get_graph_feature(xyz1, k=self.k, cuda=self.args.cuda)
        xyz = self.dgcnn_s_conv2(xyz)
        xyz2 = xyz.max(dim=-1, keepdim=False)[0]
        xyz = get_graph_feature(xyz2, k=self.k, cuda=self.args.cuda)
        xyz = self.dgcnn_s_conv3(xyz)
        xyz3 = xyz.max(dim=-1, keepdim=False)[0]

        sem = get_graph_feature(sem, k=self.k, cuda=self.args.cuda)  # Bx2fxNxk
        sem = self.dgcnn_f_conv1(sem)
        sem1 = sem.max(dim=-1, keepdim=False)[0]
        sem = get_graph_feature(sem1, k=self.k, cuda=self.args.cuda)
        sem = self.dgcnn_f_conv2(sem)
        sem2 = sem.max(dim=-1, keepdim=False)[0]
        sem = get_graph_feature(sem2, k=self.k, cuda=self.args.cuda)
        sem = self.dgcnn_f_conv3(sem)
        sem3 = sem.max(dim=-1, keepdim=False)[0]

        x = torch.cat((xyz3, sem3), dim=1)
        # x = self.dgcnn_conv_all(x)
        x = self.dgcnn_conv_end(x)
        # print(x.shape)

        x = x.permute(0, 2, 1)  # [node_num, 32]
        return x

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :return score: Similarity score.
        """

        features_1 = data["features_1"].cuda(self.args.gpu)
        features_2 = data["features_2"].cuda(self.args.gpu)

        # features B x (3+label_num) x node_num
        abstract_features_1 = self.dgcnn_conv_pass(features_1) # node_num x feature_size(filters-3)
        abstract_features_2 = self.dgcnn_conv_pass(features_2)  #BXNXF
        # print("abstract feature: ", abstract_features_1.shape)
        pooled_features_1, attention_scores_1 = self.attention(abstract_features_1) # bxfx1
        pooled_features_2, attention_scores_2 = self.attention(abstract_features_2)
        # print("pooled_features_1: ", pooled_features_1.shape)
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        # print("scores: ", scores.shape)
        scores = scores.permute(0, 2, 1) # bx1xf
        # print("scores: ", scores.shape)

        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        # print("scores: ", scores.shape)
        score = torch.sigmoid(self.scoring_layer(scores)).reshape(-1)
        # print("scores: ", score.shape)
        return score, attention_scores_1, attention_scores_2


if __name__ == "__main__":
    import types

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

    def make_args():
        return types.SimpleNamespace(
            filters_1=8,
            filters_2=8,
            filters_3=6,
            tensor_neurons=4,
            bottle_neck_neurons=5,
            K=3,
            cuda=0,
            gpu=0,
        )

    args = make_args()

    # Test group 1: AttentionModule.forward
    try:
        attention = AttentionModule(args)
        embedding = torch.randn(2, 7, args.filters_3, requires_grad=True)
        output = attention(embedding)
        check("AttentionModule.forward output not None", output is not None)
        if output is None:
            skip_checks(4, "AttentionModule.forward returned None")
        else:
            representation, sigmoid_scores = output
            check("AttentionModule representation shape", representation.shape == (2, args.filters_3, 1), f"got {tuple(representation.shape)}")
            check("AttentionModule scores shape", sigmoid_scores.shape == (2, 7, 1), f"got {tuple(sigmoid_scores.shape)}")
            check("AttentionModule finite outputs", torch.isfinite(representation).all().item() and torch.isfinite(sigmoid_scores).all().item())
            representation.sum().backward()
            check("AttentionModule weight gradient", attention.weight_matrix.grad is not None and torch.isfinite(attention.weight_matrix.grad).all().item())
    except Exception as exc:
        skip_checks(5, f"AttentionModule.forward raised {type(exc).__name__}: {exc}")

    # Test group 2: TenorNetworkModule.forward
    try:
        tensor_network = TenorNetworkModule(args)
        embedding_1 = torch.randn(2, args.filters_3, 1, requires_grad=True)
        embedding_2 = torch.randn(2, args.filters_3, 1, requires_grad=True)
        scores = tensor_network(embedding_1, embedding_2)
        check("TenorNetworkModule.forward output not None", scores is not None)
        if scores is None:
            skip_checks(4, "TenorNetworkModule.forward returned None")
        else:
            check("TenorNetworkModule score shape", scores.shape == (2, args.tensor_neurons, 1), f"got {tuple(scores.shape)}")
            check("TenorNetworkModule finite scores", torch.isfinite(scores).all().item())
            check("TenorNetworkModule relu scores", torch.ge(scores, 0).all().item())
            scores.sum().backward()
            check("TenorNetworkModule parameter gradients", tensor_network.weight_matrix.grad is not None and tensor_network.weight_matrix_block.grad is not None and tensor_network.bias.grad is not None)
    except Exception as exc:
        skip_checks(5, f"TenorNetworkModule.forward raised {type(exc).__name__}: {exc}")

    # Test group 3: SG.dgcnn_conv_pass
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = SG(args, number_of_labels=4).to(device)
        model.train()
        features = torch.randn(2, 7, 8, device=device)
        encoded = model.dgcnn_conv_pass(features)
        check("SG.dgcnn_conv_pass output not None", encoded is not None)
        if encoded is None:
            skip_checks(4, "SG.dgcnn_conv_pass returned None")
        else:
            check("SG.dgcnn_conv_pass output shape", encoded.shape == (2, 8, args.filters_3), f"got {tuple(encoded.shape)}")
            check("SG.dgcnn_conv_pass finite output", torch.isfinite(encoded).all().item())
            check("SG.dgcnn_conv_pass preserves node axis", encoded.shape[1] == features.shape[2])
            check("SG.dgcnn_conv_pass fuses spatial and semantic branches", model.dgcnn_conv_end[0].in_channels == args.filters_3 * 2)
    except Exception as exc:
        skip_checks(5, f"SG.dgcnn_conv_pass raised {type(exc).__name__}: {exc}")

    # Test group 4: SG.forward
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = SG(args, number_of_labels=4).to(device)
        model.eval()
        data = {
            "features_1": torch.randn(2, 7, 8, device=device),
            "features_2": torch.randn(2, 7, 8, device=device),
        }
        prediction = model(data)
        check("SG.forward output not None", prediction is not None)
        if prediction is None:
            skip_checks(5, "SG.forward returned None")
        else:
            score, attention_scores_1, attention_scores_2 = prediction
            check("SG.forward score shape", score.shape == (2,), f"got {tuple(score.shape)}")
            check("SG.forward score finite", torch.isfinite(score).all().item())
            check("SG.forward score bounded", torch.ge(score, 0).all().item() and torch.le(score, 1).all().item())
            check("SG.forward first attention shape", attention_scores_1.shape == (2, 8, 1), f"got {tuple(attention_scores_1.shape)}")
            check("SG.forward second attention shape", attention_scores_2.shape == (2, 8, 1), f"got {tuple(attention_scores_2.shape)}")
    except Exception as exc:
        skip_checks(6, f"SG.forward raised {type(exc).__name__}: {exc}")

    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    if failed != 0:
        raise SystemExit(1)
