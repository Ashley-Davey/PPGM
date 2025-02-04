import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt, rcParams
import time


rcParams['figure.dpi'] = 600
np.set_printoptions(precision=3)

# some general utility functions


def print_mat(A):
    for n, row in enumerate(A):
        print(" & ".join([f'{np.round(r, 3)}' for r in row]), r"\\" if n != len(A) - 1 else "")


def eig_max(A):
    return np.real(np.sqrt(np.max(eig_vals(A))))


def eig_min(A):
    return np.real(np.sqrt(np.min(eig_vals(A))))


def eig_vals(A):
    return np.linalg.eigvals(A.T @ A)


def log(*args, **kwargs):
    now = time.strftime("%H:%M:%S")
    print("[" + now + "] ", end="")
    print(*args, **kwargs)
    with open('log.txt', 'a') as f:
        print("[" + now + "] ", end="", file=f)
        print(*args, **kwargs, file=f)


def b_mult(A, B):  # batch matrix * vector multiplication
    assert len(A.shape) - 1 == len(B.shape) == 2
    try:
        return np.squeeze(A @ np.expand_dims(B, -1), -1)  # @ = matmul
    except NotImplementedError:
        return tf.squeeze(A @ tf.expand_dims(B, -1), -1)  # @ = matmul


def b_dot(A, B):  # batch vector dot product
    assert len(A.shape) == len(B.shape) == 2
    try:
        return np.sum(A * B, 1, keepdims=True)  # * = elementwise mult
    except NotImplementedError:
        return tf.reduce_sum(A * B, 1, keepdims=True)  # * = elementwise mult


def b_scale(a, B):  # batch scalar multiply
    assert len(a.shape) == 2 and a.shape[-1] == 1
    if len(B.shape) == 2:  # vector
        return a * B
    else:
        try:
            return np.expand_dims(a, 1) * B
        except NotImplementedError:
            return tf.expand_dims(a, 1) * B


def b_quad(x, A, y):  # x.T A y
    assert len(A.shape) - 1 == len(x.shape) == len(y.shape) == 2
    return b_dot(x, b_mult(A, y))


def saveplot():
    if not os.path.isdir("plots"):
        os.mkdir('plots')
    n = 0
    fname = f'plots/plot_{time.strftime("%H_%M_%S")}.png'
    while os.path.isfile(fname):
        n += 1
        fname = f'plots/plot_{time.strftime("%H_%M_%S")}({n}).png'
    plt.savefig(fname)
    plt.show()


def inv(A):  # moore-penrose inverse w/ tiny buffer
    if len(A.shape) <= 2:
        A_T = A.T
    else:
        A_T = np.transpose(A, axes=[0, 2, 1])
    return np.linalg.inv(A_T @ A + 1e-8 * np.eye(A.shape[-1])) @ A_T


def H2(config, phi1, phi2=None, A=None):  # compute H2 difference between two feedback controls
    comp = phi2 is not None
    num_sample = 1000
    paths = np.random.normal(size=[config.time_steps, num_sample, 1])
    dw = paths * config.sqrt_delta_t

    ones_vec = tf.ones(shape=tf.stack([tf.shape(dw)[1], 1]), dtype=tf.float64)
    ones_mat = tf.ones(shape=tf.stack([tf.shape(dw)[1], 1, 1]), dtype=tf.float64)

    A = ones_mat * A if A is not None else ones_mat * np.eye(config.m)

    X1 = ones_vec * np.ones(config.n)  # M x n
    if comp:
        X2 = ones_vec * np.ones(config.n)  # M x n

    ret = 0

    for i in range(config.time_steps):
        u1 = phi1(i * config.sim_t, X1)  # M x m
        try:
            u1 = u1.numpy()
        except:
            pass
        drift_X1 = (  # M x n
            b_mult(ones_mat * config.A, X1)
            + b_mult(ones_mat * config.B, u1)
            + ones_vec * config.gamma
        )
        diffusion_X1 = (  # M x n
            b_mult(ones_mat * config.C, X1)
            + b_mult(ones_mat * config.D, u1)
            + ones_vec * config.sigma
        )

        X1 = X1 + drift_X1 * config.delta_t + b_scale(dw[i], diffusion_X1)
        if comp:

            u2 = phi2(i * config.sim_t, X2)  # M x m
            try:
                u2 = u2.numpy()
            except:
                pass
            drift_X2 = (  # M x n
                b_mult(ones_mat * config.A, X2)
                + b_mult(ones_mat * config.B, u2)
                + ones_vec * config.gamma
            )
            diffusion_X2 = (  # M x n
                b_mult(ones_mat * config.C, X2)
                + b_mult(ones_mat * config.D, u2)
                + ones_vec * config.sigma
            )

            X2 = X2 + drift_X2 * config.delta_t + b_scale(dw[i], diffusion_X2)

        trap_mult = 1 if i in [0, config.time_steps - 1] else 2  # trapezium rule
        ret += 0.5 * config.sim_t * trap_mult * (b_quad(u1 - u2, A, u1 - u2) if comp else b_quad(u1, A, u1))

    return tf.reduce_mean(ret)


def sim_value(config, phi, x0, sample=10000):  # get simulation value [J, d] -> [J]
    assert len(x0.shape) == 2
    num_points = x0.shape[0]
    num_sample = sample
    paths = np.random.normal(size=[config.sim_steps, num_sample * num_points, 1])
    dw = paths * config.sqrt_sim_t

    ones_vec = np.ones(shape=np.stack([num_sample * num_points, 1]))
    ones_mat = np.ones(shape=np.stack([num_sample * num_points, 1, 1]))

    X = ones_vec * np.tile(x0, [num_sample, 1])  # M x n

    ret = 0

    for i in range(config.sim_steps):
        u = phi(i * config.sim_t, X)  # M x m
        try:
            u = u.numpy()
        except:
            pass

        trap_mult = 1 if i in [0, config.sim_steps - 1] else 2
        ret += 0.5 * config.sim_t * trap_mult * config.f(X, u)  # trapezium rule
        drift_X = (  # M x n
            b_mult(ones_mat * config.A, X)
            + b_mult(ones_mat * config.B, u)
            + ones_vec * config.gamma
        )

        diffusion_X = (  # M x n
            b_mult(ones_mat * config.C, X)
            + b_mult(ones_mat * config.D, u)
            + ones_vec * config.sigma
        )

        X = X + drift_X * config.sim_t + b_scale(dw[i], diffusion_X)

    ret += config.g(X)
    ret = ret.reshape(num_sample, num_points)

    return np.mean(ret, 0)  # J?


if __name__ == '__main__':
    pass
