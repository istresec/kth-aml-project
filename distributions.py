import tensorflow as tf

min_epsilon = 1e-5
max_epsilon = 1.-1e-5

def log_Logistic(sample, mean, logscale, average=False, reduce=True, dim=None):
    # https://github.com/openai/iaf/blob/master/tf_utils/distributions.py#L28
    binsize = 1. / 256.
    scale = tf.exp(logscale)
    sample = (tf.floor(sample / binsize) * binsize - mean) / scale
    logp = tf.math.log(tf.sigmoid(sample + binsize / scale) - tf.sigmoid(sample) + 1e-7)
    if reduce:
        if average:
            return tf.reduce_mean(logp, dim)
        else:
            return tf.reduce_sum(logp, [1, 2, 3])
    else:
        return logp

def log_Bernoulli(x, mean, average=False, dim=None):
    probs = tf.clip_by_value(mean, clip_value_min=min_epsilon, clip_value_max=max_epsilon)
    log_bernoulli = x * tf.log(probs) + (1. - x) * tf.log(1. - probs)
    if average:
        return tf.reduce_mean(log_bernoulli, dim)
    else:
        return tf.sum(log_bernoulli, dim)

def log_Normal_diag(x, mean, log_var, average=False, dim=None):
    log_normal = -0.5 * (log_var + tf.pow(x - mean, 2) / tf.exp(log_var))
    if average:
        return tf.reduce_mean(log_normal, dim)
    else:
        return tf.reduce_sum(log_normal, dim)

def log_Normal_standard(x, average=False, dim=None):
    log_normal = -0.5 * tf.pow(x, 2)
    if average:
        return tf.reduce_mean(log_normal, dim)
    else:
        return tf.reduce_sum(log_normal, dim)
