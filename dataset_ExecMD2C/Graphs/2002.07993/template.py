import copy
import numpy as np, sys, math, os

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


eps = 1e-7
inf = 1e31
save_dir = "run_time/save"


class Object(object):
    def __init__(self, **kwargs):
        for name in kwargs:
            setattr(self, name, kwargs[name])

    def vars(self):
        return self.__dict__

    def keys(self):
        return sorted(self.__dict__.keys())

    def values(self, keys=None):
        if keys is None:
            keys = self.keys()
        return [self.__dict__[k] for k in keys]

    def copy(self):
        return copy.deepcopy(self)

    def get(self, name, d=None):
        return self.__dict__.get(name, d)

    def set(self, name, value):
        self.__dict__[name] = value

    def update(self, **args):
        self.__dict__.update(args)
        return self

    def setdefault(self, **args):
        for k, v in args.items():
            if k not in self.__dict__:
                self.__dict__[k] = v
        return self


args = Object()


class Base:
    deep = True
    args = Object()

    # feature: [type..], [tags..], mid
    def __init__(self, data):
        self.raw_adjs = data.adjs

        # self.save_name = f'{utils.save_dir}/{args.run_name}/model.ckpt'
        self.save_dir = f'{save_dir}/{args.run_name}'
        self.tb_name = f'{save_dir}/{args.run_name}'

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(args.seed)
            self.compile()
        self.fit_step = 0

    def compile(self):
        self.make_io()
        self.make_model()
        if args.run_tb:
            self.all_summary = tf.summary.merge_all()
            self.tbfw = tf.summary.FileWriter(self.tb_name, self.sess.graph)

    def placeholder(self, dtype, shape, name, to_list):
        ph = tf.placeholder(dtype, shape, name)
        self.placeholder_dict[name] = ph
        to_list.append(ph)
        return ph

    def make_io(self):
        self.placeholder_dict = {}
        self.inputs = []
        L = args.seq_length
        self.placeholder(tf.int32, [None, L], 'share_seq', self.inputs)
        self.placeholder(tf.int32, [None, L], 'click_seq', self.inputs)

        self.placeholder(tf.int32, [None], 'pos', self.inputs)
        self.placeholder(tf.int32, [None, None], 'neg', self.inputs)

        self.adjs = [tf.constant(adj, dtype=tf.int32) for adj in self.raw_adjs]  # [N, M] * 4, in_0, out_0, in_1, out_1

    def get_data_map(self, data):
        data_map = dict(zip(self.inputs, data))
        return data_map

    def make_model(self):
        with tf.variable_scope('Graph', reuse=tf.AUTO_REUSE, regularizer=self.l2_loss('all')) as self.graph_scope:
            n = args.nb_nodes
            k = args.dim_k
            self.embedding_matrix = tf.get_variable(name='emb_w', shape=[n, k])
            with tf.variable_scope('graph_agg', reuse=tf.AUTO_REUSE) as self.graph_agg_scope:
                pass

        with tf.variable_scope('Network', reuse=tf.AUTO_REUSE, regularizer=self.l2_loss('all')):
            score, label = self.forward(*self.inputs)
            seq_loss = tf.losses.softmax_cross_entropy(label, score)
            tf.summary.scalar('seq_loss', seq_loss)

        self.loss = seq_loss
        self.loss += tf.losses.get_regularization_loss()

        opt = tf.train.AdamOptimizer(learning_rate=args.lr)
        self.minimizer = opt.minimize(self.loss)
        tf.summary.scalar('loss', self.loss)

        graph_var_list = tf.trainable_variables(scope='^Graph/')
        network_var_list = tf.trainable_variables(scope='^Network/')
        for v in graph_var_list:
            print('graph', v)
        for v in network_var_list:
            print('network', v)

        self.saver = tf.train.Saver()
        self.sess = self.get_session()
        self.sess.run(tf.global_variables_initializer())

    def get_session(self):
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=1,
            visible_device_list=args.gpu,
            allow_growth=True,
        )
        config = tf.ConfigProto(gpu_options=gpu_options)
        session = tf.Session(config=config)
        return session

    def l2_loss(self, name):
        alpha = args.get(f'l2_{name}', 0)
        if alpha < 1e-7:
            return None
        return lambda x: alpha * tf.nn.l2_loss(x)

    def Mean(self, seq, seq_length=None, mask=None, name=None):
        # seq: (None, L, k), seq_length: (None, ), mask: (None, L)
        # ret: (None, k)
        if seq_length is None and mask is None:
            with tf.variable_scope('Mean'):
                return tf.reduce_sum(seq, -2)

        with tf.variable_scope('MaskMean'):
            if mask is None:
                mask = tf.sequence_mask(seq_length, maxlen=tf.shape(seq)[1], dtype=tf.float32)
            mask = tf.expand_dims(mask, -1)  # (None, L, 1)
            seq = seq * mask
            seq = tf.reduce_sum(seq, -2)  # (None, k)
            seq = seq / (tf.reduce_sum(mask, -2) + eps)
        return seq

    def MLP(self, x, fc, activation, name):
        with tf.variable_scope(f'MLP_{name}'):
            for i in range(len(fc)):
                x = tf.layers.dense(x, fc[i], activation=activation, name=f'dense_{i}')
        return x

    def gate(self, a, b, name):
        with tf.variable_scope(name):
            alpha = tf.layers.dense(tf.concat([a, b], -1), 1, activation=tf.nn.sigmoid, name='gateW')
            ret = alpha * a + (1 - alpha) * b
        return ret

    def Embedding(self, node, name='node', mask_zero=False):
        # node: [BS]
        with tf.variable_scope(f'Emb_{name}'):
            emb_w = self.embedding_matrix
            t = tf.gather(emb_w, node)
            if mask_zero:
                mask = tf.not_equal(node, 0)
                mask = tf.cast(mask, tf.float32)
            else:
                mask = None
        return t, mask

    def forward(self, share_seq, click_seq, pos, neg):
        """
        [TODO] Build the session target-behavior prediction scores.

        Input:
            share_seq: (batch, seq_length) integer item ids for share behavior history.
            click_seq: (batch, seq_length) integer item ids for click behavior history.
            pos: (batch,) integer ids for the positive next item.
            neg: (batch, num_negative) integer ids for sampled negative items.

        Output: tuple of two tensors:
            - scores with shape (batch, num_negative + 1), one candidate score per
              positive/negative item.
            - labels with shape (batch, num_negative + 1), with the positive item in
              the first column and all negatives marked zero.

"""
        pass

    def node_embedding(self, node):
        # node: [BS, L]
        embs, mask = self.Embedding(node, mask_zero=True)
        return embs, mask

    def merge_seq(self, share_seq, click_seq):
        with tf.variable_scope(f'merge_seq', reuse=tf.AUTO_REUSE):
            share_seq_embs, share_mask = self.node_embedding(share_seq)
            share_emb = self.seq_embedding(share_seq_embs, share_mask, 'share')
            click_seq_embs, click_mask = self.node_embedding(click_seq)
            click_emb = self.seq_embedding(click_seq_embs, click_mask, 'click')

            emb = self.gate(share_emb, click_emb, 'merge_share_and_click_seq')
            return emb

    def seq_embedding(self, seq, mask, name):
        # seq: [BS, L, k]
        with tf.variable_scope(f'seq_embedding_{name}', reuse=tf.AUTO_REUSE):
            seq_emb = self.Mean(seq, mask=mask)
        return seq_emb


class GNN(Base):
    args = Base.args.copy().update()

    def node_embedding(self, node):
        # node: [BS, L]
        with tf.variable_scope(self.graph_scope):
            embs, mask = self.node_list_aggregate(node, depth=args.gnnd, mask_zero=True)  # [BS, L, k]
        return embs, mask

    def node_list_aggregate(self, nodes, depth, mask_zero, name='share'):
        """
        [TODO] Aggregate graph-enhanced embeddings for a batch of node lists.

        Input:
            nodes: (batch, list_length) integer item ids.
            depth: integer recursion depth for graph aggregation.
            mask_zero: boolean indicating whether item id zero should be masked.
            name: string scope suffix.

        Output: tuple of two tensors:
            - embeddings with shape (batch, list_length, dim_k).
            - mask with shape (batch, list_length), or None when masking is disabled.

"""
        pass

    def _agg(self, cur, nxt, mask, name):
        return self.Mean(nxt, mask=mask)

    def _merge(self, cur, nxt):
        return cur + nxt

    def single_node_aggregate(self, node, depth, mask_zero, name='share'):
        """
        [TODO] Recursively aggregate one batch of item nodes over four relations.

        Input:
            node: (batch,) integer item ids.
            depth: integer recursion depth.
            mask_zero: boolean indicating whether item id zero should be masked.
            name: string scope suffix.

        Output: tuple of two tensors:
            - embeddings with shape (batch, dim_k).
            - mask with shape (batch,), or None when masking is disabled.

"""
        pass

    def _aggregate4(self, cur, nxt_in_0, nxt_in_0_mask, nxt_out_0, nxt_out_0_mask, nxt_in_1, nxt_in_1_mask, nxt_out_1, nxt_out_1_mask, name):
        """
        [TODO] Merge four relation-specific neighbor representations.

        Input:
            cur: (batch, dim_k) current node representation.
            nxt_in_0: (batch, neighbors, dim_k) first incoming-relation neighbors.
            nxt_in_0_mask: (batch, neighbors) mask for first incoming relation.
            nxt_out_0: (batch, neighbors, dim_k) first outgoing-relation neighbors.
            nxt_out_0_mask: (batch, neighbors) mask for first outgoing relation.
            nxt_in_1: (batch, neighbors, dim_k) second incoming-relation neighbors.
            nxt_in_1_mask: (batch, neighbors) mask for second incoming relation.
            nxt_out_1: (batch, neighbors, dim_k) second outgoing-relation neighbors.
            nxt_out_1_mask: (batch, neighbors) mask for second outgoing relation.
            name: string scope suffix.

        Output: (batch, dim_k) tensor containing the merged node representation.

"""
        pass


if __name__ == "__main__":
    args.update(
        nb_nodes=600,
        dim_k=4,
        seq_length=3,
        gnnd=1,
        seed=123,
        lr=1e-3,
        gpu="",
        run_tb=False,
        run_name="benchmark",
        l2_all=0,
    )

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

    def make_fake_adjs():
        return [
            np.array([[0, 0], [2, 3], [1, 4], [2, 5], [3, 1], [4, 2], [5, 1], [1, 3]], dtype=np.int32),
            np.array([[0, 0], [3, 2], [4, 1], [5, 2], [1, 3], [2, 4], [1, 5], [3, 1]], dtype=np.int32),
            np.array([[0, 0], [4, 5], [3, 1], [2, 4], [1, 2], [5, 3], [2, 1], [4, 2]], dtype=np.int32),
            np.array([[0, 0], [5, 4], [1, 3], [4, 2], [2, 1], [3, 5], [1, 2], [2, 4]], dtype=np.int32),
        ]

    def make_bare_model(graph, nb_nodes=8):
        model = object.__new__(GNN)
        model.raw_adjs = make_fake_adjs()
        model.adjs = [tf.constant(adj, dtype=tf.int32) for adj in model.raw_adjs]
        with tf.variable_scope("Graph", reuse=tf.AUTO_REUSE) as model.graph_scope:
            model.embedding_matrix = tf.get_variable(name="emb_w", shape=[nb_nodes, args.dim_k])
            with tf.variable_scope("graph_agg", reuse=tf.AUTO_REUSE) as model.graph_agg_scope:
                pass
        return model

    def run_session(graph, fetches, feed_dict=None):
        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(fetches, feed_dict=feed_dict)

    # Test group 1: Base.forward
    try:
        graph = tf.Graph()
        with graph.as_default():
            tf.set_random_seed(11)
            args.update(nb_nodes=600)
            model = make_bare_model(graph, nb_nodes=args.nb_nodes)
            share_seq = tf.constant([[1, 2, 0], [3, 4, 5]], dtype=tf.int32)
            click_seq = tf.constant([[2, 3, 0], [4, 5, 6]], dtype=tf.int32)
            pos = tf.constant([3, 4], dtype=tf.int32)
            neg = tf.constant([[5, 6], [1, 2]], dtype=tf.int32)
            output = Base.forward(model, share_seq, click_seq, pos, neg)
            check("Base.forward output not None", output is not None)
            if output is None:
                skip_checks(4, "Base.forward returned None")
            else:
                score, label = output
                score_v, label_v, topk_v = run_session(graph, [score, label, model.topkI])
                check("Base.forward score shape", score_v.shape == (2, 3), f"got {score_v.shape}")
                check("Base.forward label shape", label_v.shape == (2, 3), f"got {label_v.shape}")
                check("Base.forward score finite", np.isfinite(score_v).all())
                check("Base.forward labels mark one positive", np.array_equal(label_v, np.array([[1, 0, 0], [1, 0, 0]], dtype=np.int32)))
                check("Base.forward keeps recommendation topk", topk_v.shape == (2, 500), f"got {topk_v.shape}")
    except Exception as exc:
        skip_checks(6, f"Base.forward raised {type(exc).__name__}: {exc}")

    # Test group 2: GNN.node_list_aggregate
    try:
        graph = tf.Graph()
        with graph.as_default():
            tf.set_random_seed(13)
            args.update(nb_nodes=8)
            model = make_bare_model(graph)
            nodes = tf.constant([[1, 0], [2, 3]], dtype=tf.int32)
            output = GNN.node_list_aggregate(model, nodes, depth=1, mask_zero=True)
            check("node_list_aggregate output not None", output is not None)
            if output is None:
                skip_checks(3, "node_list_aggregate returned None")
            else:
                emb, mask = output
                emb_v, mask_v = run_session(graph, [emb, mask])
                check("node_list_aggregate embedding shape", emb_v.shape == (2, 2, args.dim_k), f"got {emb_v.shape}")
                check("node_list_aggregate mask shape", mask_v.shape == (2, 2), f"got {mask_v.shape}")
                check("node_list_aggregate finite embeddings", np.isfinite(emb_v).all())
                check("node_list_aggregate preserves zero mask", mask_v[0, 1] == 0.0 and mask_v[0, 0] == 1.0)
    except Exception as exc:
        skip_checks(5, f"GNN.node_list_aggregate raised {type(exc).__name__}: {exc}")

    # Test group 3: GNN.single_node_aggregate
    try:
        graph = tf.Graph()
        with graph.as_default():
            tf.set_random_seed(17)
            args.update(nb_nodes=8)
            model = make_bare_model(graph)
            node = tf.constant([1, 2, 0], dtype=tf.int32)
            output = GNN.single_node_aggregate(model, node, depth=1, mask_zero=True)
            check("single_node_aggregate output not None", output is not None)
            if output is None:
                skip_checks(4, "single_node_aggregate returned None")
            else:
                emb, mask = output
                base_emb, _ = GNN.single_node_aggregate(model, node, depth=0, mask_zero=True)
                emb_v, base_v, mask_v = run_session(graph, [emb, base_emb, mask])
                check("single_node_aggregate embedding shape", emb_v.shape == (3, args.dim_k), f"got {emb_v.shape}")
                check("single_node_aggregate mask shape", mask_v.shape == (3,), f"got {mask_v.shape}")
                check("single_node_aggregate finite embeddings", np.isfinite(emb_v).all())
                check("single_node_aggregate preserves input mask", np.array_equal(mask_v, np.array([1.0, 1.0, 0.0], dtype=np.float32)))
                check("single_node_aggregate uses neighbors at depth", not np.allclose(emb_v[:2], base_v[:2]))
    except Exception as exc:
        skip_checks(6, f"GNN.single_node_aggregate raised {type(exc).__name__}: {exc}")

    # Test group 4: GNN._aggregate4
    try:
        graph = tf.Graph()
        with graph.as_default():
            tf.set_random_seed(19)
            args.update(nb_nodes=8)
            model = make_bare_model(graph)
            cur = tf.constant([[1.0, 2.0, 3.0, 4.0], [0.5, 0.25, 0.75, 1.0]], dtype=tf.float32)
            zero_neighbors = tf.zeros([2, 2, args.dim_k], dtype=tf.float32)
            zero_mask = tf.zeros([2, 2], dtype=tf.float32)
            output = GNN._aggregate4(model, cur, zero_neighbors, zero_mask, zero_neighbors, zero_mask, zero_neighbors, zero_mask, zero_neighbors, zero_mask, "zero_case")
            check("_aggregate4 output not None", output is not None)
            if output is None:
                skip_checks(3, "_aggregate4 returned None")
            else:
                out_v, cur_v = run_session(graph, [output, cur])
                check("_aggregate4 output shape", out_v.shape == (2, args.dim_k), f"got {out_v.shape}")
                check("_aggregate4 output finite", np.isfinite(out_v).all())
                check("_aggregate4 empty neighbors preserve current", np.allclose(out_v, cur_v))
                check("_aggregate4 merges four relation directions", len(make_fake_adjs()) == 4)
    except Exception as exc:
        skip_checks(5, f"GNN._aggregate4 raised {type(exc).__name__}: {exc}")

    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    if failed != 0:
        raise SystemExit(1)
