import types

import torch
from torch_geometric.nn import SAGEConv


class AttentionModule(torch.nn.Module):
    def __init__(self, args):
        super(AttentionModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        self.weight_matrix = torch.nn.Parameter(
            torch.Tensor(self.args.filters_3, self.args.filters_3)
        )

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding):
        global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=0)
        transformed_global = torch.tanh(global_context)
        sigmoid_scores = torch.sigmoid(torch.mm(embedding, transformed_global.view(-1, 1)))
        representation = torch.mm(torch.t(embedding), sigmoid_scores)
        return representation


class TenorNetworkModule(torch.nn.Module):
    def __init__(self, args):
        super(TenorNetworkModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        self.weight_matrix = torch.nn.Parameter(
            torch.Tensor(
                self.args.filters_3,
                self.args.filters_3,
                self.args.tensor_neurons,
            )
        )
        self.weight_matrix_block = torch.nn.Parameter(
            torch.Tensor(self.args.tensor_neurons, 2 * self.args.filters_3)
        )
        self.bias = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 1))

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        scoring = torch.mm(
            torch.t(embedding_1),
            self.weight_matrix.view(self.args.filters_3, -1),
        )
        scoring = scoring.view(self.args.filters_3, self.args.tensor_neurons)
        scoring = torch.mm(torch.t(scoring), embedding_2)
        combined_representation = torch.cat((embedding_1, embedding_2))
        block_scoring = torch.mm(self.weight_matrix_block, combined_representation)
        scores = torch.nn.functional.relu(scoring + block_scoring + self.bias)
        return scores


class funcGNN(torch.nn.Module):
    def __init__(self, args, number_of_labels):
        super(funcGNN, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()

    def calculate_bottleneck_features(self):
        if self.args.histogram is True:
            self.feature_count = self.args.tensor_neurons + self.args.bins
        else:
            self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        self.calculate_bottleneck_features()
        self.convolution_1 = SAGEConv(
            self.number_labels,
            self.args.filters_1,
            normalize=True,
        )
        self.convolution_2 = SAGEConv(
            self.args.filters_1,
            self.args.filters_2,
            normalize=True,
        )
        self.convolution_3 = SAGEConv(
            self.args.filters_2,
            self.args.filters_3,
            normalize=True,
        )
        self.attention = AttentionModule(self.args)
        self.tensor_network = TenorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(
            self.feature_count,
            self.args.bottle_neck_neurons,
        )
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)

    def calculate_histogram(self, abstract_features_1, abstract_features_2):
        scores = torch.mm(abstract_features_1, abstract_features_2).detach()
        scores = scores.view(-1, 1)
        hist = torch.histc(scores, bins=self.args.bins)
        hist = hist / torch.sum(hist)
        hist = hist.view(1, -1)
        return hist

    def convolutional_pass(self, edge_index, features):
        features = self.convolution_1(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(
            features,
            p=self.args.dropout,
            training=self.training,
        )
        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(
            features,
            p=self.args.dropout,
            training=self.training,
        )
        features = self.convolution_3(features, edge_index)
        return features

    def forward(self, data):
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]
        features_1 = data["features_1"]
        features_2 = data["features_2"]

        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)

        if self.args.histogram is True:
            hist = self.calculate_histogram(
                abstract_features_1,
                torch.t(abstract_features_2),
            )

        pooled_features_1 = self.attention(abstract_features_1)
        pooled_features_2 = self.attention(abstract_features_2)
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)

        if self.args.histogram is True:
            scores = torch.cat((scores, hist), dim=1).view(1, -1)

        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        score = torch.sigmoid(self.scoring_layer(scores))
        return score


if __name__ == "__main__":
    torch.manual_seed(7)
    results = {"passed": 0, "failed": 0}

    def check(name, condition, detail=""):
        if bool(condition):
            results["passed"] += 1
            print(f"PASS: {name}")
        else:
            results["failed"] += 1
            suffix = f" - {detail}" if detail else ""
            print(f"FAIL: {name}{suffix}")

    def skip_checks(label, exc, count):
        for index in range(count):
            check(
                f"{label} check {index + 1}",
                False,
                f"setup raised {type(exc).__name__}: {exc}",
            )

    args = types.SimpleNamespace(
        filters_1=5,
        filters_2=6,
        filters_3=4,
        tensor_neurons=3,
        bottle_neck_neurons=5,
        bins=4,
        dropout=0.0,
        histogram=True,
    )

    # Group 1: AttentionModule.forward
    try:
        attention = AttentionModule(args)
        embedding = torch.randn(5, args.filters_3, requires_grad=True)
        representation = attention(embedding)
        attention_loss = representation.sum()
        attention_loss.backward()

        check("attention output shape", representation.shape == (args.filters_3, 1))
        check("attention output is finite", torch.isfinite(representation).all().item())
        check(
            "attention weight receives gradient",
            attention.weight_matrix.grad is not None
            and torch.isfinite(attention.weight_matrix.grad).all().item(),
        )
        check(
            "attention pools all nodes into one graph vector",
            representation.numel() == args.filters_3,
        )
    except Exception as exc:
        skip_checks("AttentionModule.forward", exc, 4)

    # Group 2: TenorNetworkModule.forward
    try:
        tensor_network = TenorNetworkModule(args)
        embedding_1 = torch.randn(args.filters_3, 1, requires_grad=True)
        embedding_2 = torch.randn(args.filters_3, 1, requires_grad=True)
        tensor_scores = tensor_network(embedding_1, embedding_2)
        tensor_loss = tensor_scores.sum()
        tensor_loss.backward()

        check("tensor network output shape", tensor_scores.shape == (args.tensor_neurons, 1))
        check("tensor network output is finite", torch.isfinite(tensor_scores).all().item())
        check("tensor network applies relu", torch.ge(tensor_scores, 0).all().item())
        check(
            "tensor network parameters receive gradients",
            tensor_network.weight_matrix.grad is not None
            and tensor_network.weight_matrix_block.grad is not None
            and tensor_network.bias.grad is not None,
        )
    except Exception as exc:
        skip_checks("TenorNetworkModule.forward", exc, 4)

    # Group 3: funcGNN.forward
    try:
        model = funcGNN(args, number_of_labels=3)
        model.eval()
        data = {
            "edge_index_1": torch.tensor(
                [[0, 1, 1, 2], [1, 0, 2, 1]],
                dtype=torch.long,
            ),
            "edge_index_2": torch.tensor(
                [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]],
                dtype=torch.long,
            ),
            "features_1": torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.2, 0.3, 0.5]],
                dtype=torch.float32,
            ),
            "features_2": torch.tensor(
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.4, 0.4, 0.2],
                ],
                dtype=torch.float32,
            ),
        }
        prediction = model(data)
        check("model output shape", prediction.shape == (1, 1))
        check("model output is finite", torch.isfinite(prediction).all().item())
        check(
            "model output is sigmoid bounded",
            torch.ge(prediction, 0).all().item() and torch.le(prediction, 1).all().item(),
        )
        check(
            "histogram branch expands bottleneck features",
            model.feature_count == args.tensor_neurons + args.bins,
        )
        check(
            "shared convolution stack produces graph embeddings",
            model.convolutional_pass(data["edge_index_1"], data["features_1"]).shape
            == (3, args.filters_3),
        )
    except Exception as exc:
        skip_checks("funcGNN.forward", exc, 5)

    print(f"Summary: {results['passed']} passed, {results['failed']} failed")
    if results["failed"]:
        raise SystemExit(1)
