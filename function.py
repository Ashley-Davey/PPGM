import tensorflow as tf
from common import b_mult


class FeedForwardSubNet(tf.keras.Model):
    def __init__(self, config, num_hiddens=[10, 10], dim=1, t=False, name=None):
        super(FeedForwardSubNet, self).__init__()
        self.config = config
        self.t = t
        self.dense_layers = [tf.keras.layers.Dense(num_hiddens[i],
                                                   # kernel_regularizer=tf.keras.regularizers.L1(0.01),
                                                   # bias_regularizer=tf.keras.regularizers.L1(0.01),
                                                   use_bias=True,
                                                   activation=None)
                             for i in range(len(num_hiddens))]
        self.dense_layers.append(tf.keras.layers.Dense(dim,
                                                       # kernel_regularizer=tf.keras.regularizers.L1(0.01),
                                                       # bias_regularizer=tf.keras.regularizers.L1(0.01),
                                                       use_bias=True,
                                                       activation=None))

    def call(self, x, training):
        if self.t:
            t = x[:, :1]
            X = x[:, 1:]
            X = (X - self.config.X[:, 0]) / (self.config.X[:, 1] - self.config.X[:, 0])
            x = tf.concat([t, X], 1)

        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = tf.tanh(x)
        x = self.dense_layers[-1](x)
        return x


class LinearFunction(tf.keras.Model):
    # Linear in X, quadratic in t (where applicable)
    def __init__(self, config, dim=1, t=True):
        super(LinearFunction, self).__init__()
        self.t = t
        self.dim = dim
        self.config = config
        if t:
            self.layer = tf.keras.layers.Dense(2 * config.n * dim, use_bias=True, activation=None)
            self.layer2 = tf.keras.layers.Dense(dim, use_bias=True, activation=None)
        else:
            self.layer = tf.keras.layers.Dense(dim, use_bias=True, activation=None)

    def call(self, x, training):
        if self.t:
            t = x[:, :1]
            X = x[:, 1:]
            X = (X - self.config.X[:, 0]) / (self.config.X[:, 1] - self.config.X[:, 0])
            x_data = tf.concat([X - 0.5, tf.maximum(X - 0.5, 0)], 1)
            t_data = tf.concat([t, tf.pow(t, 2)], 1)
            A = tf.reshape(
                self.layer(t_data),
                shape=tf.stack([tf.shape(x)[0], self.dim, 2 * self.config.n])
            )
            b = self.layer2(t_data)

            return b_mult(A, x_data) + b
        else:
            X = (x - self.config.X[:, 0]) / (self.config.X[:, 1] - self.config.X[:, 0])
            x_data = tf.concat([X - 0.5, tf.maximum(X - 0.5, 0)], 1)

            return self.layer(x_data)
