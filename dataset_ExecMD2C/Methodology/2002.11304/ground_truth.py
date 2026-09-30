"""
ground_truth.py for PaDGAN core model components.

Source-consolidated from:
- synthetic/models.py
- airfoil/bezier_gan.py

Only the GAN architecture components and performance-augmented diversity
objectives are included. Experiment drivers, parameter-update procedures,
file I/O wrappers, plotting utilities, external simulators, and saved-graph
handling are intentionally excluded.
"""

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


EPSILON = 1e-7


# --- [Original file: synthetic/models.py] ---
class BaseModel(object):
    
    def __init__(self, noise_dim, data_dim, lambda0=1., lambda1=0.01):

        self.noise_dim = noise_dim
        self.data_dim = data_dim
        self.lambda0 = lambda0
        self.lambda1 = lambda1
    
    def compute_diversity_loss(self, x, equation, val_scale):
            
#        x_norm = tf.linalg.l2_normalize(x, 1)
#        S = tf.tensordot(x_norm, tf.transpose(x_norm), 1) # similarity matrix (cosine)
#        D = None
        
        r = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
        D = r - 2*tf.matmul(x, tf.transpose(x)) + tf.transpose(r)
        S = tf.exp(-0.5*tf.square(D)) # similarity matrix (rbf)
#        S = 1/(1+D)
        
        y = val_scale * equation(x)
        
        if self.lambda0 == 'inf':
            
            eig_val, _ = tf.self_adjoint_eig(S)
            loss = -10*tf.reduce_mean(y)
            
            Q = None
            L = None
        
        elif self.lambda0 == 'naive':
            
            eig_val, _ = tf.self_adjoint_eig(S)
            loss = -tf.reduce_mean(tf.log(tf.maximum(eig_val, EPSILON)))-10*tf.reduce_mean(y)
            
            Q = None
            L = None
            
        else:
            
            Q = tf.tensordot(tf.expand_dims(y, 1), tf.expand_dims(y, 0), 1) # quality matrix
            if self.lambda0 == 0.:
                L = S
            else:
                L = S * tf.pow(Q, self.lambda0)
            
            eig_val, _ = tf.self_adjoint_eig(L)
            loss = -tf.reduce_mean(tf.log(tf.maximum(eig_val, EPSILON)))
        
        return loss, D, S, Q, L, y


class GAN(BaseModel):
        
    def generator(self, z, reuse=tf.AUTO_REUSE):
        
        with tf.variable_scope('Generator', reuse=reuse):
            
            h = tf.layers.dense(z, 128)
            h = tf.nn.leaky_relu(h, alpha=0.2)
#            h = tf.nn.tanh(h)
            
            h = tf.layers.dense(h, 128)
            h = tf.nn.leaky_relu(h, alpha=0.2)
#            h = tf.nn.tanh(h)
            
            h = tf.layers.dense(h, 128)
            h = tf.nn.leaky_relu(h, alpha=0.2)
#            h = tf.nn.tanh(h)
            
            h = tf.layers.dense(h, self.data_dim)
#            x = tf.identity(h, name='gen')
            x = tf.nn.tanh(h, name='gen')
            
            return x
        
    def discriminator(self, x, reuse=tf.AUTO_REUSE):
        
        with tf.variable_scope('Discriminator', reuse=reuse):
            
            h = tf.layers.dense(x, 128)
            h = tf.nn.leaky_relu(h, alpha=0.2)
#            h = tf.nn.tanh(h)
            
            h = tf.layers.dense(h, 128)
            h = tf.nn.leaky_relu(h, alpha=0.2)
#            h = tf.nn.tanh(h)
            
            log_d = tf.layers.dense(h, 1)
            
            return log_d


# --- [Original file: airfoil/bezier_gan.py] ---
class BezierGAN(object):
    
    def __init__(self, latent_dim=5, noise_dim=100, n_points=64, bezier_degree=16, bounds=(0.0, 1.0), 
                 lambda0=1., lambda1=0.01):

        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        
        self.n_points = n_points
        self.X_shape = (n_points, 2, 1)
        self.bezier_degree = bezier_degree
        self.bounds = bounds
        
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        
    def generator(self, c, z, reuse=tf.AUTO_REUSE, training=True):
        
        depth_cpw = 32*8
        dim_cpw = int((self.bezier_degree+1)/8)
        kernel_size = (4,3)
#        noise_std = 0.01
        
        with tf.variable_scope('Generator', reuse=reuse):
                
            if self.noise_dim == 0:
                cz = c
            else:
                cz = tf.concat([c, z], axis=-1)
            
            cpw = tf.layers.dense(cz, 1024)
            cpw = tf.layers.batch_normalization(cpw, momentum=0.9)#, training=training)
            cpw = tf.nn.leaky_relu(cpw, alpha=0.2)
    
            cpw = tf.layers.dense(cpw, dim_cpw*3*depth_cpw)
            cpw = tf.layers.batch_normalization(cpw, momentum=0.9)#, training=training)
            cpw = tf.nn.leaky_relu(cpw, alpha=0.2)
            cpw = tf.reshape(cpw, (-1, dim_cpw, 3, depth_cpw))
    
            cpw = tf.layers.conv2d_transpose(cpw, int(depth_cpw/2), kernel_size, strides=(2,1), padding='same')
            cpw = tf.layers.batch_normalization(cpw, momentum=0.9)#, training=training)
            cpw = tf.nn.leaky_relu(cpw, alpha=0.2)
#            cpw += tf.random_normal(shape=tf.shape(cpw), stddev=noise_std)
            
            cpw = tf.layers.conv2d_transpose(cpw, int(depth_cpw/4), kernel_size, strides=(2,1), padding='same')
            cpw = tf.layers.batch_normalization(cpw, momentum=0.9)#, training=training)
            cpw = tf.nn.leaky_relu(cpw, alpha=0.2)
#            cpw += tf.random_normal(shape=tf.shape(cpw), stddev=noise_std)
            
            cpw = tf.layers.conv2d_transpose(cpw, int(depth_cpw/8), kernel_size, strides=(2,1), padding='same')
            cpw = tf.layers.batch_normalization(cpw, momentum=0.9)#, training=training)
            cpw = tf.nn.leaky_relu(cpw, alpha=0.2)
#            cpw += tf.random_normal(shape=tf.shape(cpw), stddev=noise_std)
            
            # Control points
            cp = tf.layers.conv2d(cpw, 1, (1,2), padding='valid') # batch_size x (bezier_degree+1) x 2 x 1
            cp = tf.nn.tanh(cp)
            cp = tf.squeeze(cp, axis=-1, name='control_point') # batch_size x (bezier_degree+1) x 2
            
            # Weights
            w = tf.layers.conv2d(cpw, 1, (1,3), padding='valid')
            w = tf.nn.sigmoid(w) # batch_size x (bezier_degree+1) x 1 x 1
            w = tf.squeeze(w, axis=-1, name='weight') # batch_size x (bezier_degree+1) x 1
            
            # Parameters at data points
            db = tf.layers.dense(cz, 1024)
            db = tf.layers.batch_normalization(db, momentum=0.9)#, training=training)
            db = tf.nn.leaky_relu(db, alpha=0.2)
            
            db = tf.layers.dense(db, 256)
            db = tf.layers.batch_normalization(db, momentum=0.9)#, training=training)
            db = tf.nn.leaky_relu(db, alpha=0.2)
            
            db = tf.layers.dense(db, self.X_shape[0]-1)
            db = tf.nn.softmax(db) # batch_size x (n_data_points-1)
            
#            db = tf.random_gamma([tf.shape(cz)[0], self.X_shape[0]-1], alpha=100, beta=100)
#            db = tf.nn.softmax(db) # batch_size x (n_data_points-1)
            
            ub = tf.pad(db, [[0,0],[1,0]], constant_values=0) # batch_size x n_data_points
            ub = tf.cumsum(ub, axis=1)
            ub = tf.minimum(ub, 1)
            ub = tf.expand_dims(ub, axis=-1) # 1 x n_data_points x 1
            
            # Bezier layer
            # Compute values of basis functions at data points
            num_control_points = self.bezier_degree + 1
            lbs = tf.tile(ub, [1, 1, num_control_points]) # batch_size x n_data_points x n_control_points
            pw1 = tf.range(0, num_control_points, dtype=tf.float32)
            pw1 = tf.reshape(pw1, [1, 1, -1]) # 1 x 1 x n_control_points
            pw2 = tf.reverse(pw1, axis=[-1])
            lbs = tf.add(tf.multiply(pw1, tf.log(lbs+EPSILON)), tf.multiply(pw2, tf.log(1-lbs+EPSILON))) # batch_size x n_data_points x n_control_points
            lc = tf.add(tf.lgamma(pw1+1), tf.lgamma(pw2+1))
            lc = tf.subtract(tf.lgamma(tf.cast(num_control_points, dtype=tf.float32)), lc) # 1 x 1 x n_control_points
            lbs = tf.add(lbs, lc) # batch_size x n_data_points x n_control_points
            bs = tf.exp(lbs)
            # Compute data points
            cp_w = tf.multiply(cp, w)
            dp = tf.matmul(bs, cp_w) # batch_size x n_data_points x 2
            bs_w = tf.matmul(bs, w) # batch_size x n_data_points x 1
            dp = tf.div(dp, bs_w) # batch_size x n_data_points x 2
            dp = tf.expand_dims(dp, axis=-1, name='fake_image') # batch_size x n_data_points x 2 x 1
            
            return dp, cp, w, ub, db
        
    def discriminator(self, x, reuse=tf.AUTO_REUSE, training=True):
        
        depth = 64
        dropout = 0.4
        kernel_size = (4,2)
        
        with tf.variable_scope('Discriminator', reuse=reuse):
        
            x = tf.layers.conv2d(x, depth*1, kernel_size, strides=(2,1), padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)#, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, dropout, training=training)
            
            x = tf.layers.conv2d(x, depth*2, kernel_size, strides=(2,1), padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)#, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, dropout, training=training)
            
            x = tf.layers.conv2d(x, depth*4, kernel_size, strides=(2,1), padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)#, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, dropout, training=training)
            
            x = tf.layers.conv2d(x, depth*8, kernel_size, strides=(2,1), padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)#, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, dropout, training=training)
            
            x = tf.layers.conv2d(x, depth*16, kernel_size, strides=(2,1), padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)#, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, dropout, training=training)
            
            x = tf.layers.conv2d(x, depth*32, kernel_size, strides=(2,1), padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)#, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, dropout, training=training)
            
            x = tf.layers.flatten(x)
            x = tf.layers.dense(x, 1024)
            x = tf.layers.batch_normalization(x, momentum=0.9)#, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            
            d = tf.layers.dense(x, 1)
            
            q = tf.layers.dense(x, 128)
#            q = tf.layers.batch_normalization(q, momentum=0.9)#, training=training)
            q = tf.nn.leaky_relu(q, alpha=0.2)
            q_mean = tf.layers.dense(q, self.latent_dim)
            q_logstd = tf.layers.dense(q, self.latent_dim)
            q_logstd = tf.maximum(q_logstd, -16)
            # Reshape to batch_size x 1 x latent_dim
            q_mean = tf.reshape(q_mean, (-1, 1, self.latent_dim))
            q_logstd = tf.reshape(q_logstd, (-1, 1, self.latent_dim))
            q = tf.concat([q_mean, q_logstd], axis=1, name='predicted_latent') # batch_size x 2 x latent_dim
            
            return d, q
    
    def compute_diversity_loss(self, x, y):
            
        x = tf.layers.flatten(x)
        y = tf.squeeze(y)
        
        r = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
        D = r - 2*tf.matmul(x, tf.transpose(x)) + tf.transpose(r)
        S = tf.exp(-0.5*tf.square(D)) # similarity matrix (rbf)
#        S = 1/(1+D)
        
        if self.lambda0 == 'naive':
            
            eig_val, _ = tf.self_adjoint_eig(S)
            loss = -tf.reduce_mean(tf.log(tf.maximum(eig_val, EPSILON)))-10*tf.reduce_mean(y)
            
            Q = None
            L = None
            
        else:
            
            Q = tf.tensordot(tf.expand_dims(y, 1), tf.expand_dims(y, 0), 1) # quality matrix
            if self.lambda0 == 0.:
                L = S
            else:
                L = S * tf.pow(Q, self.lambda0)
            
            eig_val, _ = tf.self_adjoint_eig(L)
            loss = -tf.reduce_mean(tf.log(tf.maximum(eig_val, EPSILON)))
        
        return loss, D, S, Q, L


if __name__ == "__main__":
    np.random.seed(42)
    if hasattr(tf, "set_random_seed"):
        tf.set_random_seed(42)

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

    def reset_graph():
        if hasattr(tf, "reset_default_graph"):
            tf.reset_default_graph()

    def tensor_shape_list(tensor):
        shape = tensor.get_shape().as_list()
        return tuple(shape)

    print("=" * 70)
    print("PaDGAN core model component benchmark")
    print("=" * 70)

    # ==========================================================
    # Test 1: synthetic BaseModel.compute_diversity_loss
    # ==========================================================
    print("[Test 1/6] synthetic performance-augmented DPP loss")
    try:
        reset_graph()
        model = BaseModel(noise_dim=3, data_dim=2, lambda0=1.0, lambda1=0.01)
        x = tf.constant([[0.0, 0.0], [0.5, 0.25], [1.0, -0.5]], dtype=tf.float32)

        def equation(v):
            return tf.reduce_sum(tf.square(v), axis=1)

        loss, D, S, Q, L, y = model.compute_diversity_loss(x, equation, val_scale=2.0)
        check("synthetic loss not None", loss is not None)
        if loss is not None:
            check("synthetic distance shape", tensor_shape_list(D) == (3, 3), f"got {tensor_shape_list(D)}")
            check("synthetic similarity shape", tensor_shape_list(S) == (3, 3), f"got {tensor_shape_list(S)}")
            check("synthetic quality matrix shape", tensor_shape_list(Q) == (3, 3), f"got {tensor_shape_list(Q)}")
            check("synthetic kernel matrix shape", tensor_shape_list(L) == (3, 3), f"got {tensor_shape_list(L)}")
            check("synthetic value vector shape", tensor_shape_list(y) == (3,), f"got {tensor_shape_list(y)}")
        else:
            skip_checks(5, "synthetic diversity loss returned None")
    except Exception as exc:
        skip_checks(6, f"synthetic diversity loss raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 2: synthetic lambda0 branches
    # ==========================================================
    print("[Test 2/6] synthetic DPP branch behavior")
    try:
        reset_graph()
        x = tf.constant([[0.0, 0.0], [0.5, 0.25], [1.0, -0.5]], dtype=tf.float32)

        def equation(v):
            return tf.reduce_sum(tf.square(v), axis=1)

        naive = BaseModel(noise_dim=3, data_dim=2, lambda0="naive")
        _, _, _, Q_naive, L_naive, _ = naive.compute_diversity_loss(x, equation, 1.0)
        zero = BaseModel(noise_dim=3, data_dim=2, lambda0=0.0)
        _, _, S_zero, Q_zero, L_zero, _ = zero.compute_diversity_loss(x, equation, 1.0)
        inf_model = BaseModel(noise_dim=3, data_dim=2, lambda0="inf")
        _, _, _, Q_inf, L_inf, _ = inf_model.compute_diversity_loss(x, equation, 1.0)
        check("naive omits quality matrix", Q_naive is None)
        check("naive omits kernel matrix", L_naive is None)
        check("inf omits quality matrix", Q_inf is None)
        check("zero branch keeps kernel shape", tensor_shape_list(L_zero) == (3, 3), f"got {tensor_shape_list(L_zero)}")
        check("zero branch has quality matrix", tensor_shape_list(Q_zero) == (3, 3), f"got {tensor_shape_list(Q_zero)}")
        check("zero branch similarity shape", tensor_shape_list(S_zero) == (3, 3), f"got {tensor_shape_list(S_zero)}")
    except Exception as exc:
        skip_checks(6, f"synthetic branch behavior raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 3: synthetic GAN generator/discriminator
    # ==========================================================
    print("[Test 3/6] synthetic GAN graph shapes")
    try:
        reset_graph()
        gan = GAN(noise_dim=4, data_dim=2, lambda0=1.0, lambda1=0.01)
        z = tf.placeholder(tf.float32, shape=[None, 4], name="z")
        fake = gan.generator(z)
        logit_fake = gan.discriminator(fake)
        logit_real = gan.discriminator(tf.placeholder(tf.float32, shape=[None, 2], name="real"))
        check("synthetic generator not None", fake is not None)
        if fake is not None:
            check("synthetic generator shape", tensor_shape_list(fake) == (None, 2), f"got {tensor_shape_list(fake)}")
            check("synthetic generator output named", fake.name.endswith("gen:0"), fake.name)
        else:
            skip_checks(2, "synthetic generator returned None")
        check("synthetic fake discriminator not None", logit_fake is not None)
        if logit_fake is not None:
            check("synthetic fake discriminator shape", tensor_shape_list(logit_fake) == (None, 1), f"got {tensor_shape_list(logit_fake)}")
        else:
            skip_checks(1, "synthetic fake discriminator returned None")
        check("synthetic real discriminator shape", tensor_shape_list(logit_real) == (None, 1), f"got {tensor_shape_list(logit_real)}")
    except Exception as exc:
        skip_checks(6, f"synthetic GAN graph raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 4: BezierGAN generator
    # ==========================================================
    print("[Test 4/6] BezierGAN generator Bezier outputs")
    try:
        reset_graph()
        bezier = BezierGAN(latent_dim=3, noise_dim=4, n_points=16, bezier_degree=7)
        c = tf.placeholder(tf.float32, shape=[None, 3], name="latent")
        z = tf.placeholder(tf.float32, shape=[None, 4], name="noise")
        dp, cp, w, ub, db = bezier.generator(c, z, training=False)
        check("Bezier generator output not None", dp is not None)
        if dp is not None:
            check("Bezier fake image shape", tensor_shape_list(dp) == (None, 16, 2, 1), f"got {tensor_shape_list(dp)}")
            check("Bezier control point shape", tensor_shape_list(cp) == (None, 8, 2), f"got {tensor_shape_list(cp)}")
            check("Bezier weight shape", tensor_shape_list(w) == (None, 8, 1), f"got {tensor_shape_list(w)}")
            check("Bezier parameter shape", tensor_shape_list(ub) == (None, 16, 1), f"got {tensor_shape_list(ub)}")
            check("Bezier spacing distribution shape", tensor_shape_list(db) == (None, 15), f"got {tensor_shape_list(db)}")
            check("Bezier fake image tensor named", dp.name.endswith("fake_image:0"), dp.name)
        else:
            skip_checks(6, "Bezier generator returned None")
    except Exception as exc:
        skip_checks(7, f"Bezier generator raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 5: BezierGAN discriminator
    # ==========================================================
    print("[Test 5/6] BezierGAN discriminator outputs")
    try:
        reset_graph()
        bezier = BezierGAN(latent_dim=3, noise_dim=4, n_points=16, bezier_degree=8)
        x = tf.placeholder(tf.float32, shape=[None, 16, 2, 1], name="airfoil")
        d, q = bezier.discriminator(x, training=False)
        check("Bezier discriminator logit not None", d is not None)
        if d is not None:
            check("Bezier discriminator logit shape", tensor_shape_list(d) == (None, 1), f"got {tensor_shape_list(d)}")
        else:
            skip_checks(1, "Bezier discriminator logit returned None")
        check("Bezier latent head not None", q is not None)
        if q is not None:
            check("Bezier latent head shape", tensor_shape_list(q) == (None, 2, 3), f"got {tensor_shape_list(q)}")
            check("Bezier latent head named", q.name.endswith("predicted_latent:0"), q.name)
        else:
            skip_checks(2, "Bezier discriminator latent head returned None")
    except Exception as exc:
        skip_checks(5, f"Bezier discriminator raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 6: BezierGAN.compute_diversity_loss
    # ==========================================================
    print("[Test 6/6] BezierGAN performance-augmented DPP loss")
    try:
        reset_graph()
        bezier = BezierGAN(latent_dim=2, noise_dim=0, n_points=4, bezier_degree=4, lambda0=1.0)
        x = tf.constant(
            [
                [[[0.0], [0.0]], [[0.2], [0.1]], [[0.4], [0.0]], [[0.6], [-0.1]]],
                [[[0.1], [0.0]], [[0.3], [0.2]], [[0.5], [0.1]], [[0.7], [0.0]]],
                [[[0.2], [0.1]], [[0.4], [0.3]], [[0.6], [0.2]], [[0.8], [0.1]]],
            ],
            dtype=tf.float32,
        )
        y = tf.constant([[0.8], [0.9], [1.1]], dtype=tf.float32)
        loss, D, S, Q, L = bezier.compute_diversity_loss(x, y)
        check("Bezier diversity loss not None", loss is not None)
        if loss is not None:
            check("Bezier distance shape", tensor_shape_list(D) == (3, 3), f"got {tensor_shape_list(D)}")
            check("Bezier similarity shape", tensor_shape_list(S) == (3, 3), f"got {tensor_shape_list(S)}")
            check("Bezier quality matrix shape", tensor_shape_list(Q) == (3, 3), f"got {tensor_shape_list(Q)}")
            check("Bezier kernel matrix shape", tensor_shape_list(L) == (3, 3), f"got {tensor_shape_list(L)}")
        else:
            skip_checks(4, "Bezier diversity loss returned None")
    except Exception as exc:
        skip_checks(5, f"Bezier diversity loss raised {type(exc).__name__}: {exc}")
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
