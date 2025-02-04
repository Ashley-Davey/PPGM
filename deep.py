import tensorflow as tf
import numpy as np
from common import b_mult, b_scale
from function import FeedForwardSubNet


class DeepModel(tf.keras.Model):
    def __init__(self, config):
        super(DeepModel, self).__init__()
        self.config = config
        self.optimiser_control = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate_control)
        self.optimiser_bsde = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate_bsde)
        self.phi = FeedForwardSubNet(config, [10, 10], self.config.m, t=True)
        self.Z = FeedForwardSubNet(config, [10, 10], self.config.n, t=True)
        self.Y0 = FeedForwardSubNet(config, [10, 10], self.config.n)

    def gen(self):
        t = np.ones(shape=np.stack([1, 1]))
        X = np.ones(shape=np.stack([1, self.config.n]))
        state = np.concatenate([t, X], 1)
        self.phi(state)
        self.Y0(X)
        self.Z(state)

    def control(self, t, X):
        ones_vec = np.ones(shape=np.stack([np.shape(X)[0], 1]))
        state = np.concatenate([t * ones_vec, X], 1)
        return self.config.project(self.phi(state))

    def bsde(self, t, X):
        return self.config.project(self.Y0(X))

    def get_sample(self, num_sample):
        paths = np.random.normal(size=[self.config.time_steps, num_sample, 1])
        dw_sample = paths * self.config.sqrt_delta_t

        X = np.random.rand(num_sample, self.config.n)  # 1000, n
        return dw_sample, X * (self.config.X[:, 1] - self.config.X[:, 0]) + self.config.X[:, 0]

    def loss(self):
        with tf.GradientTape() as tape:
            return self(self.get_sample(self.config.batch_size), tape)

    @tf.function
    def bsde_step(self):
        with tf.GradientTape() as tape:
            loss_bsde, loss_control = self(self.get_sample(self.config.batch_size), tape)
        grads = tape.gradient(loss_bsde, self.Z.trainable_variables + self.Y0.trainable_variables)
        self.optimiser_bsde.apply_gradients(
            zip(grads, self.Z.trainable_variables + self.Y0.trainable_variables))
        return loss_bsde, loss_control

    @tf.function
    def control_step(self, phi):
        with tf.GradientTape() as tape:
            loss_bsde, loss_control = self(self.get_sample(self.config.batch_size), tape)
        grads = tape.gradient(
            loss_control, self.phi.trainable_variables)
        self.optimiser_control.apply_gradients(zip(grads, self.phi.trainable_variables))
        return loss_bsde, loss_control

    def call(self, sample_data, tape):
        dw, X0 = sample_data

        ones_vec = tf.ones(shape=tf.stack([tf.shape(dw)[1], 1]), dtype=tf.float64)
        ones_mat = tf.ones(shape=tf.stack([tf.shape(dw)[1], 1, 1]), dtype=tf.float64)

        X = X0  # M x n
        Y = self.Y0(X)  # M x n
        loss_control = 0

        for i in range(self.config.time_steps):
            t = ones_vec * i * self.config.delta_t
            Z = self.Z(tf.concat([t, X], 1))  # M x n
            u = self.config.project(self.phi(tf.concat([t, X], 1)))  # M x m
            drift_X = (  # M x n
                b_mult(ones_mat * self.config.A, X)
                + b_mult(ones_mat * self.config.B, u)
                + ones_vec * self.config.gamma
            )
            diffusion_X = (  # M x n
                b_mult(ones_mat * self.config.C, X)
                + b_mult(ones_mat * self.config.D, u)
                + ones_vec * self.config.sigma
            )

            drift_Y = - (b_mult(ones_mat * self.config.A.T, Y) +
                         b_mult(ones_mat * self.config.C.T, Z) + self.config.f_x(X, u))

            with tape.stop_recording():
                dH = (b_mult(ones_mat * self.config.B.T, Y) +
                      b_mult(ones_mat * self.config.D.T, Z) + self.config.f_u(X, u))

                pre_phi = self.config.project(u - self.config.tau * dH)

            X = X + drift_X * self.config.delta_t + b_scale(dw[i], diffusion_X)
            Y = Y + drift_Y * self.config.delta_t + b_scale(dw[i], Z)

            trap_mult = 1 if i in [0, self.config.time_steps - 1] else 2  # trapezium rule
            loss_control += 0.5 * self.config.delta_t * trap_mult * \
                (tf.reduce_sum(tf.square(u - pre_phi), axis=1))  # + tf.square(u[:, 2:3]))

        loss_bsde = tf.reduce_sum(tf.square(Y - self.config.g_x(X)), axis=1)

        return tf.reduce_mean(loss_bsde), tf.reduce_mean(loss_control)
