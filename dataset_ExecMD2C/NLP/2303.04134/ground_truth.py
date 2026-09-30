# ============================================================
# ground_truth.py - A Hybrid Architecture for OOD Intent Detection
# Source: NLP/A-hybrid-architecture-for-out-of-domain-intent-detection-and-intent-discovery-main
#
# Contains ONLY the core model architecture definitions from model/vae.py.
# No training, dataset loading, evaluation loop, saved-weight I/O, or remote model code.
# ============================================================

# --- Third-party imports ---
import tensorflow as tf
from tensorflow_probability import distributions as tfd


# --- [Original file: model/vae.py] ---
def vae_cost(y_true, y_pred, mu, sigma, z_sample, analytic_kl=True, kl_weight=1):
    # compute cross entropy loss for each dimension of every datapoint
    raw_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,
                        logits=y_pred)  

    # compute cross entropy loss for all instances in mini-batch
    neg_log_likelihood = tf.math.reduce_sum(raw_cross_entropy)
    # compute reverse KL divergence, either analytically 
    # or through MC approximation with one sample
    if analytic_kl:
        kl_divergence = - 0.5 * tf.math.reduce_sum(
            1 + tf.math.log(tf.math.square(sigma)) - tf.math.square(mu) - tf.math.square(sigma),
            axis=1)  
    else:
        logpz = tfd.Normal(loc=0., scale=1.).prob(z_sample)
        logqz_x = tfd.Normal(loc=mu, scale=tf.math.square(sigma)).prob(z_sample) 
        kl_divergence = logqz_x - logpz
    elbo = tf.math.reduce_mean(-kl_weight * kl_divergence - neg_log_likelihood)  
    return -elbo


class Sampling(tf.keras.layers.Layer):
  def call(sellf, inputs):
    mu, sigma = inputs
    batch = tf.shape(mu)[0]
    dim = tf.shape(mu)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return mu + tf.exp(0.5 * sigma) * epsilon

def encoder_layers(inputs, latent_dim, dims:list, activation:str):
    encoder = tf.keras.layers.Dense(units=dims[0], activation=activation)(inputs)
    encoder = tf.keras.layers.Dropout(0.2)(encoder)

    for dim in dims:
        encoder = tf.keras.layers.Dense(units=dim, activation=activation)(encoder)
        encoder = tf.keras.layers.Dropout(0.2)(encoder)
 
    mu = tf.keras.layers.Dense(units=latent_dim, activation=activation)(encoder)
    sigma = tf.keras.layers.Dense(units=latent_dim, activation=activation)(encoder)
    return mu, sigma, encoder

def encoder_model(input_shape, latent_dim, dims, activation='tanh'):
  input = tf.keras.layers.Input(shape=input_shape, name='encoder_model_input')
  mu, sigma, encoder = encoder_layers(inputs=input, latent_dim=latent_dim, dims=dims, activation=activation)
  z = Sampling()((mu, sigma))
  model = tf.keras.Model(inputs=input, outputs=[mu, sigma, z])
  model._name = 'Encoder'
  return model

def decoder_layers(inputs, dims, activation):
    dec = tf.keras.layers.Dense(units=dims[0], activation=activation)(inputs)
    dec = tf.keras.layers.Dropout(0.2)(dec)
    
    for dim in dims[1:-1]:
        dec = tf.keras.layers.Dense(units=dim, activation=activation)(dec)
        dec = tf.keras.layers.Dropout(0.2)(dec)

    dec = tf.keras.layers.Dense(units=dims[-1], activation=activation)(dec)

    return dec

def decoder_model(input_shape, dims, activation='tanh'):
  inputs = tf.keras.layers.Input(shape=input_shape)
  outputs = decoder_layers(inputs, dims, activation)
  model = tf.keras.Model(inputs, outputs)
  model._name = 'Decoder'
  return model

def vae(encoder, decoder, bert, input_shape):
 input_ids = tf.keras.layers.Input(shape=input_shape, name='input_ids', dtype=tf.int32)
 attention_mask = tf.keras.layers.Input(shape=input_shape, name='attention_mask', dtype=tf.int32)
 token_type_ids = tf.keras.layers.Input(shape=input_shape, name='token_type_ids', dtype=tf.int32)
 embeddings = bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0][:, 0]

 mu, sigma, z = encoder(embeddings)
 reconstructed = decoder(z)

 model = tf.keras.Model(inputs=[input_ids, attention_mask, token_type_ids], outputs=reconstructed)

 loss = vae_cost(embeddings, reconstructed, mu, sigma, z)
 model.add_loss(loss)
 model._name = 'VAE'
 return model


# ============================================================
# __main__: Automated test suite for 5 ablated functions
# ============================================================

if __name__ == "__main__":
    tf.random.set_seed(42)

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
    print("A Hybrid Architecture for OOD Intent Detection - benchmark suite")
    print("=" * 70)

    # ------------------------------------------------------------
    # Test 1/5: vae_cost
    # ------------------------------------------------------------
    try:
        y_true = tf.constant([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]], dtype=tf.float32)
        y_pred = tf.Variable([[0.2, -0.3, 0.1, 0.5], [-0.1, 0.4, -0.2, 0.3]], dtype=tf.float32)
        mu = tf.Variable([[0.1, -0.2, 0.3], [0.2, 0.1, -0.1]], dtype=tf.float32)
        sigma = tf.Variable([[1.1, 0.9, 1.2], [1.0, 1.3, 0.8]], dtype=tf.float32)
        z_sample = tf.Variable([[0.0, 0.2, -0.2], [0.1, -0.1, 0.3]], dtype=tf.float32)
        with tf.GradientTape() as tape:
            loss = vae_cost(y_true, y_pred, mu, sigma, z_sample, analytic_kl=True, kl_weight=0.7)
        grads = tape.gradient(loss, [y_pred, mu, sigma])
        check("vae_cost output not None", loss is not None)
        if loss is not None:
            check("vae_cost scalar shape", tuple(loss.shape) == (), str(tuple(loss.shape)))
            check("vae_cost finite", bool(tf.reduce_all(tf.math.is_finite(loss)).numpy()))
            check("vae_cost nonnegative analytic objective", float(loss.numpy()) >= 0.0)
            check("vae_cost gradients reach reconstruction and latent stats", all(g is not None for g in grads))
        else:
            skip_checks(4, "vae_cost returned None")
    except Exception as exc:
        skip_checks(5, f"vae_cost raised {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------
    # Test 2/5: Sampling.call
    # ------------------------------------------------------------
    try:
        sampler = Sampling()
        mu = tf.Variable([[0.1, -0.2, 0.3], [0.2, 0.1, -0.1]], dtype=tf.float32)
        sigma = tf.Variable([[0.0, 0.2, -0.1], [0.1, -0.2, 0.3]], dtype=tf.float32)
        with tf.GradientTape() as tape:
            z = sampler((mu, sigma))
            z_sum = tf.reduce_sum(z)
        grads = tape.gradient(z_sum, [mu, sigma])
        check("Sampling output not None", z is not None)
        if z is not None:
            check("Sampling output shape", tuple(z.shape) == (2, 3), str(tuple(z.shape)))
            check("Sampling output finite", bool(tf.reduce_all(tf.math.is_finite(z)).numpy()))
            check("Sampling preserves latent width", int(tf.shape(z)[1].numpy()) == 3)
            check("Sampling gradients reach mu and sigma", all(g is not None for g in grads))
        else:
            skip_checks(4, "Sampling returned None")
    except Exception as exc:
        skip_checks(5, f"Sampling raised {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------
    # Test 3/5: encoder_layers
    # ------------------------------------------------------------
    try:
        encoder_input = tf.keras.layers.Input(shape=(4,))
        mu_out, sigma_out, hidden_out = encoder_layers(encoder_input, latent_dim=3, dims=[6, 5], activation="tanh")
        enc_probe = tf.keras.Model(encoder_input, [mu_out, sigma_out, hidden_out])
        x = tf.ones((2, 4), dtype=tf.float32)
        mu_value, sigma_value, hidden_value = enc_probe(x, training=False)
        check("encoder_layers outputs not None", mu_value is not None and sigma_value is not None and hidden_value is not None)
        if mu_value is not None and sigma_value is not None and hidden_value is not None:
            check("encoder_layers mu shape", tuple(mu_value.shape) == (2, 3), str(tuple(mu_value.shape)))
            check("encoder_layers sigma shape", tuple(sigma_value.shape) == (2, 3), str(tuple(sigma_value.shape)))
            check("encoder_layers hidden shape follows last dim", tuple(hidden_value.shape) == (2, 5), str(tuple(hidden_value.shape)))
            check("encoder_layers outputs finite", bool(tf.reduce_all(tf.math.is_finite(mu_value)).numpy()) and bool(tf.reduce_all(tf.math.is_finite(sigma_value)).numpy()) and bool(tf.reduce_all(tf.math.is_finite(hidden_value)).numpy()))
        else:
            skip_checks(4, "encoder_layers returned None output")
    except Exception as exc:
        skip_checks(5, f"encoder_layers raised {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------
    # Test 4/5: decoder_layers
    # ------------------------------------------------------------
    try:
        decoder_input = tf.keras.layers.Input(shape=(3,))
        decoded = decoder_layers(decoder_input, dims=[5, 4], activation="tanh")
        dec_probe = tf.keras.Model(decoder_input, decoded)
        latent = tf.ones((2, 3), dtype=tf.float32)
        reconstructed = dec_probe(latent, training=False)
        dense_layers = [layer for layer in dec_probe.layers if isinstance(layer, tf.keras.layers.Dense)]
        check("decoder_layers output not None", reconstructed is not None)
        if reconstructed is not None:
            check("decoder_layers output shape", tuple(reconstructed.shape) == (2, 4), str(tuple(reconstructed.shape)))
            check("decoder_layers output finite", bool(tf.reduce_all(tf.math.is_finite(reconstructed)).numpy()))
            check("decoder_layers final width follows dims[-1]", int(tf.shape(reconstructed)[1].numpy()) == 4)
            check("decoder_layers builds two dense projections", len(dense_layers) == 2, str(len(dense_layers)))
        else:
            skip_checks(4, "decoder_layers returned None")
    except Exception as exc:
        skip_checks(5, f"decoder_layers raised {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------
    # Test 5/5: vae
    # ------------------------------------------------------------
    try:
        class TinyBert(tf.keras.layers.Layer):
            def __init__(self, vocab_size=17, embed_dim=4):
                super().__init__()
                self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)
                self.projection = tf.keras.layers.Dense(embed_dim, activation="tanh")

            def call(self, input_ids, token_type_ids=None, attention_mask=None):
                tokens = self.embedding(input_ids)
                if attention_mask is not None:
                    tokens = tokens * tf.cast(tf.expand_dims(attention_mask, -1), tokens.dtype)
                return (self.projection(tokens),)

        enc = encoder_model((4,), 3, dims=[6, 5], activation="tanh")
        dec = decoder_model((3,), dims=[5, 4], activation="tanh")
        bert = TinyBert()
        model = vae(encoder=enc, decoder=dec, bert=bert, input_shape=(5,))
        ids = tf.constant([[1, 2, 3, 4, 0], [4, 3, 2, 1, 0]], dtype=tf.int32)
        mask = tf.constant([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]], dtype=tf.int32)
        type_ids = tf.zeros((2, 5), dtype=tf.int32)
        output = model([ids, mask, type_ids], training=False)
        check("vae output not None", output is not None)
        if output is not None:
            check("vae reconstructed embedding shape", tuple(output.shape) == (2, 4), str(tuple(output.shape)))
            check("vae output finite", bool(tf.reduce_all(tf.math.is_finite(output)).numpy()))
            check("vae exposes three named inputs", [tensor.name.split(':')[0] for tensor in model.inputs] == ["input_ids", "attention_mask", "token_type_ids"])
            check("vae registers variational loss", len(model.losses) >= 1, str(len(model.losses)))
        else:
            skip_checks(4, "vae returned None")
    except Exception as exc:
        skip_checks(5, f"vae raised {type(exc).__name__}: {exc}")

    print("=" * 70)
    print(f"RESULT: passed={passed} failed={failed}")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
