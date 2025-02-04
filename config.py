import tensorflow as tf
import numpy as np
from scipy.optimize import minimize
import time
from common import log, eig_max, eig_min, saveplot, sim_value, inv, print_mat, b_dot, b_mult, b_quad
from matplotlib import pyplot as plt, rcParams
from scipy.integrate import odeint

rcParams['figure.dpi'] = 600
np.set_printoptions(precision=3)


class Config(object):
    def __init__(self, **kwargs):
        # default parameters
        self.debug = False

        self.T = 1.0
        self.x_range = [-10.0, 10.0]
        self.n = 5
        self.m = 3
        self.cone = False

        self.time_steps = 20
        self.learning_rate_control = 0.1
        self.learning_rate_bsde = 0.1

        self.iteration_steps = 200  # PPGM iteration steps
        self.display_step = 10000
        self.optimisation_steps_bsde = 100
        self.optimisation_steps_control = 100
        self.batch_size = 100

        self.tau = 0.1
        self.supplement = False
        self.sim_steps = 50
        self.fix_rand = True

        # change parameters with constructor
        for key, value in kwargs.items():
            if hasattr(self, key):
                log(f'Changed {key} to {value}')
                setattr(self, key, value)

        if self.fix_rand:
            np.random.seed(1)

        self.A = (np.random.rand(self.n, self.n) - 0.5) / (self.n)
        self.B = (np.random.rand(self.n, self.m) - 0.5) / (self.n)
        self.C = (np.random.rand(self.n, self.n) - 0.5) / (self.n)
        self.D = (np.random.rand(self.n, self.m) - 0.5) / (self.n)

        self.gamma = np.zeros(self.n)
        self.sigma = np.zeros(self.n)

        self.X = np.array([self.x_range] * self.n)

        self.delta_t = self.T / self.time_steps
        self.sqrt_delta_t = np.sqrt(self.delta_t)

        self.sim_t = self.T / self.sim_steps
        self.sqrt_sim_t = np.sqrt(self.sim_t)

    @property
    def A_cal(self):
        return self.A + self.A.T + self.C.T @ self.C

    @property
    def B_cal(self):
        return self.B + self.C.T @ self.D

    @property
    def D_cal(self):
        return self.D.T @ self.D

    @property
    def bound(self):
        return 2 * np.exp(self.T * (eig_max(self.A_cal) + 4 * 16 * eig_max(self.C) ** 2 + 1)) * ((1 + 4) * eig_max(self.D_cal) + eig_max(self.B_cal) ** 2)

    @ property
    def delta(self):
        if self.A_zero:
            return eig_min(self.D_cal - self.B_cal.T @ self.B)
        else:
            return eig_min(self.D_cal - self.B_cal.T @ inv(self.A_cal) @ self.B)

    def project(self, u):
        if self.cone:
            try:
                return np.maximum(u, 0.0 * u)
            except NotImplementedError:
                return tf.nn.relu(u)
        else:
            return u

    def sanity_check(self):
        if self.value is None or self.phi is None:
            return

        j = 0
        X = np.stack(
            [np.zeros(5)] * j +
            [np.linspace(self.x_range[0], self.x_range[1], 5)]
            + [np.zeros(5)] * (self.n - j - 1), 1
        )

        v1 = self.value(0, X)

        v2 = sim_value(self, self.phi, X)

        for _v1, _v2, x in zip(v1, v2, np.linspace(self.x_range[0], self.x_range[1], 5)):
            if not np.isclose(_v1, _v2, rtol=0.1):
                log(f"value doesn't match at x = {x: .0f},  {_v1:.2f} != {_v2:.2f}")

    def plot(self):
        test_size = 100
        j = 0
        Xs = np.stack(
            [np.zeros(100)] * j +
            [np.linspace(self.x_range[0], self.x_range[1], test_size)]
            + [np.zeros(100)] * (self.n - j - 1), 1
        )
        ones_n = np.stack([np.ones(test_size)] * self.n, 1)
        j = 0
        Us = np.stack(
            [np.zeros(test_size)] * j +
            [np.linspace(self.x_range[0], self.x_range[1], test_size)]
            + [np.zeros(test_size)] * (self.m - j - 1), 1
        )
        ones_m = np.stack([np.ones(test_size)] * self.m, 1)

        if self.phi is not None:

            for j in range(self.m):
                plt.figure()
                plt.grid(0.5)
                plt.plot(Xs[:, 0], self.phi(0, Xs)[:, j])
                plt.title(f'optimal control, dimension {j+1}')
                plt.xlabel(r'x')
                plt.ylabel(rf'$u^*_{j+1}$')
                saveplot()

        if self.value is not None:
            plt.figure()
            plt.grid(0.5)
            plt.plot(Xs[:, 0], self.value(0, Xs))
            plt.title('value function')
            plt.xlabel(r'$x$')
            plt.ylabel(r'$v$')
            saveplot()

        plt.figure()
        plt.grid(0.5)
        plt.plot(Xs[:, 0], self.f(Xs, ones_m)[:, 0])
        plt.title(r'$f_0(x, 1)$')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$f$')

        saveplot()

        plt.figure()
        plt.grid(0.5)
        plt.plot(Us[:, 0], self.f(ones_n, Us)[:, 0])
        plt.title(r'$f_0(1, u)$')
        plt.xlabel(r'$u$')
        plt.ylabel(r'$f$')

        saveplot()

        plt.figure()
        plt.grid(0.5)
        plt.plot(Xs[:, 0], self.g(Xs)[:, 0])
        plt.title(r'$g(x)$')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$g$')

        saveplot()

    def compare(self, Fs, labels, ylabel, title=''):
        try:
            test_size = 25

            for j in range(1):  # self.n):
                X = np.stack([np.stack(
                    [np.zeros(test_size)] * j +
                    [np.linspace(self.x_range[0], self.x_range[1], test_size)]
                    + [np.zeros(test_size)] * (self.n - j - 1), 1)
                    for _ in range(self.time_steps)])

                for i in range(1):
                    plt.figure()
                    plt.grid(0.5)
                    for k, f in enumerate(Fs):
                        linestyle = {'PPGM': 'solid', 'Linear': 'solid',
                                     'LQ-PGM': 'dashed', 'soln': 'dotted'}[labels[k]]
                        linewidth = 1 if labels[k] in ['PPGM', 'Linear'] else 2
                        c = 'r' if labels[k] == 'soln' else f'C{k}'
                        y = np.stack([f(i / self.time_steps, X[i]) for i in range(self.time_steps)])
                        Y = y[i, :, 0] if len(y.shape) == 3 else y[i]

                        plt.plot(X[i, :, j], Y, linewidth=linewidth, label=labels[k], linestyle=linestyle, c=c)

                    plt.title(title + f't = {i * self.delta_t:.2f}, j = {j}')
                    plt.xlabel(r'$X$')
                    plt.ylabel(ylabel)
                    plt.legend()
                    saveplot()
        except Exception as e:
            log('Compare termination due to ', e)

    def f(self, x, u):
        raise NotImplementedError()

    def f_x(self, x, u):
        raise NotImplementedError()

    def f_u(self, x, u):
        raise NotImplementedError()

    def g(self, x):
        raise NotImplementedError()

    def g_x(self, x):
        raise NotImplementedError()

    def value(self, t, X):
        return np.squeeze([sim_value(self, self.phi, x[:, np.newaxis]) for x in X])

    def phi(self, i, x):
        raise NotImplementedError()


class LQConfig(Config):
    def __init__(self, **kwargs):
        self.zero = False
        self.r = None
        super().__init__(**kwargs)

        if self.zero:
            self.D += np.eye(self.n, self.m)
            self.A += np.eye(self.n, self.n) / 3

        self.Q = np.ones((self.n, self.n)) / 5 * (not self.zero)

        if self.r is not None:
            self.R = np.eye(self.m) * self.r * (not self.zero)
        else:
            R_temp = (np.random.rand(self.m, self.m) - 0.5) / 2
            self.R = ((R_temp + R_temp.T) / np.sqrt(self.m) + np.eye(self.m)) * (not self.zero)

        self.S = (np.random.rand(self.m, self.n) - 0.5) * (not self.zero)

        G_temp = (np.random.rand(self.n, self.n) - 0.5) / (2)
        self.G = np.eye(self.n) + (G_temp + G_temp.T)

        self.q = (np.random.rand(self.n) - 0.5) * (not self.cone) * 0
        self.p = (np.random.rand(self.m) - 0.5) * (not self.cone) * 0
        self.g_vec = (np.random.rand(self.n) - 0.5) * (not self.cone) * 0

        time_now = time.time()
        self.phi, self.value, self.bsde = solve_LQ_cone(self) if self.cone else solve_LQ(self)
        self.solve_time = time.time() - time_now

        self.running = eig_min(self.R) > 0
        self.A_zero = np.isclose(eig_min(self.A_cal), 0)

        assert self.running or eig_min(self.G) > 0

        term_string = f"terminal, A {'=' if self.A_zero else '>'} 0"

        log(f'config structure: strong convexity in {"running" if self.running else term_string}')

        self.tau = min(max(1 / (self.L * (1 if self.running else self.delta)), 0.001), 0.8)

        solve_LQ(self)  # get unconstrained solution for lq-pgm
        self.sanity_check()

        if self.debug:

            for name, matrix in [
                    ['A', self.A],
                    ['B', self.B],
                    ['C', self.C],
                    ['D', self.D],
                    [r"\A", self.A_cal],
                    [r"\B", self.B_cal],
                    [r"\D", self.D_cal],
                    [r"\D - \B\B", self.D_cal - self.B_cal.T @ self.B_cal],
                    [r"\D - \B\A^{-1}\B", self.D_cal - self.B_cal.T @ inv(self.A_cal) @ self.B_cal],
            ]:

                log(name)
                print_mat(matrix)

        if self.debug:
            upper = np.concatenate([self.Q, self.S], 0)
            lower = np.concatenate([self.S.T, self.R], 0)
            running = np.concatenate([upper, lower], 1)

            for name, matrix in [
                    ['running', running],
                    ['G', self.G],
            ]:

                log(name)
                print_mat(matrix)

        log(f'mu = {self.mu: .3f}, delta = {self.delta:.3f}, tau = {self.tau:.3f}')

    @ property
    def mu(self):
        if self.running:
            return eig_min(self.R)
        else:
            return eig_min(self.G)

    @ property
    def L(self):
        if self.running:
            return eig_max(self.R)
        else:
            return eig_max(self.G)

    @ property
    def K(self):
        return (
            self.bound ** 2 * eig_max(self.G) * (not self.running)  # I1
            + self.bound * (eig_max(self.Q) * self.bound + eig_max(self.S))  # I2
            + eig_max(self.S) * self.bound  # I3
            + eig_max(self.R) * self.running  # I4
        )

    @ property
    def K2(self):
        if self.running or not self.A_zero:
            return 0
        else:
            return self.bound

    @ property
    def constant(self):
        return self.mu * (self.delta - self.K2) / 4 - self.K

    def f(self, x, u):
        ones_vec = np.ones(shape=np.stack([np.shape(x)[0], 1]), dtype=np.float64)
        ones_mat = np.ones(shape=np.stack([np.shape(x)[0], 1, 1]), dtype=np.float64)
        Q = ones_mat * self.Q
        R = ones_mat * self.R
        S = ones_mat * self.S
        p = ones_vec * self.p
        q = ones_vec * self.q

        return (
            0.5 * b_quad(x, Q, x)
            + b_quad(u, S, x)
            + 0.5 * b_quad(u, R, u)
            + b_dot(x, q)
            + b_dot(u, p)
        )

    def f_x(self, x, u):
        ones_vec = np.ones(shape=np.stack([np.shape(x)[0], 1]), dtype=np.float64)
        ones_mat = np.ones(shape=np.stack([np.shape(x)[0], 1, 1]), dtype=np.float64)
        Q = ones_mat * self.Q
        ST = ones_mat * self.S.T
        q = ones_vec * self.q

        return (
            b_mult(Q, x)
            + b_mult(ST, u)
            + q
        )

    def f_u(self, x, u):
        ones_vec = np.ones(shape=np.stack([np.shape(x)[0], 1]), dtype=np.float64)
        ones_mat = np.ones(shape=np.stack([np.shape(x)[0], 1, 1]), dtype=np.float64)
        R = ones_mat * self.R
        S = ones_mat * self.S
        p = ones_vec * self.p

        return (
            b_mult(S, x)
            + b_mult(R, u)
            + p
        )

    def g(self, x):
        ones_vec = np.ones(shape=np.stack([np.shape(x)[0], 1]), dtype=np.float64)
        ones_mat = np.ones(shape=np.stack([np.shape(x)[0], 1, 1]), dtype=np.float64)
        G = ones_mat * self.G
        g = ones_vec * self.g_vec

        return (
            0.5 * b_quad(x, G, x)
            + b_dot(x, g)
        )

    def g_x(self, x):
        ones_vec = np.ones(shape=np.stack([np.shape(x)[0], 1]), dtype=np.float64)
        ones_mat = np.ones(shape=np.stack([np.shape(x)[0], 1, 1]), dtype=np.float64)
        G = ones_mat * self.G
        g = ones_vec * self.g_vec

        return (
            b_mult(G, x)
            + g
        )

    def interp(self, vec, t):
        # interpolate for vec at t

        pre = max(min(int(t * self.sim_steps / self.T), self.sim_steps - 1), 0)

        # return vec[pre]
        post = min(pre + 1, self.sim_steps - 1)

        c = t * self.sim_steps / self.T - pre

        return vec[pre] * c + vec[post] * (1 - c)


class FConfig(Config):
    def __init__(self, **kwargs):
        self.lam = 0
        super().__init__(**kwargs)

        # self.B = - np.eye(self.n, self.m) / 10
        self.D += 1.5 * np.eye(self.n, self.m)

        self.A = - 0.5 * self.B @ inv(self.D) @ inv(self.D).T @ self.B.T
        self.C = - inv(self.D).T @ self.B.T

        self.A_zero = np.isclose(eig_min(self.A_cal), 0)

        log(f"config structure: strong convexity in terminal, A {'=' if self.A_zero else '>'} 0")
        log(f'mu = {self.mu: .3f}, delta = {self.delta:.3f}, tau = {self.tau:.3f}')

        self.tau = min(max(1 / (self.L * (self.delta)), 0.01), 1.0)

        # assert self.tau > 0
        # assert self.constant > 0

        self.phi = lambda t, X: np.zeros((X.shape[0], self.m))
        self.value = lambda t, X: sim_value(self, self.phi, X)

        self.sanity_check()

        if self.debug:

            for name, matrix in [
                    ['A', self.A],
                    ['B', self.B],
                    ['C', self.C],
                    ['D', self.D],
                    [r"\A", self.A_cal],
                    [r"\B", self.B_cal],
                    [r"\D", self.D_cal],
                    [r"\D - \B\B", self.D_cal - self.B_cal.T @ self.B_cal],
                    [r"\D - \B\A^{-1}\B", self.D_cal - self.B_cal.T @ inv(self.A_cal) @ self.B_cal],
            ]:

                log(name)
                print_mat(matrix)

    @ property
    def mu(self):
        return 1

    @ property
    def L(self):
        return 1

    @ property
    def K(self):
        return (
            1  # I4
        )

    @ property
    def constant(self):
        return self.mu * (self.delta - self.K2) / 4 - self.K

    def f(self, x, u):

        return self.lam * (b_dot(u, u) - 1)

        def f_i(z):
            return tf.nn.relu(z - 1) + tf.nn.relu(- z - 1)

        return tf.reduce_sum(f_i(u), 1, keepdims=True)  # + b_dot(x, x)

    def f_x(self, x, u):
        return 0 * x  # 2 * x

    def f_u(self, x, u):
        return self.lam * 2 * u

        def d_relu(z):
            # return tf.math.sigmoid(10 * z)
            return tf.where(z > 0, tf.ones_like(z), tf.zeros_like(z))

        return ((d_relu(u - 1) + d_relu(-u - 1)))

    def g(self, x):
        return 0.5 * b_dot(x, x)

    def g_x(self, x):
        return x


def solve_LQ(config):

    start_time = time.time()

    def fun(t, a):
        a_temp = a.reshape(config.n, config.n)
        K = config.D.T @ (a_temp @ config.D) + config.R
        L = config.B.T @ a_temp + config.D.T @ (a_temp @ config.C) + config.S

        ret = (
            a_temp @ config.A
            + config.A.T @ a_temp
            + config.C.T @ (a_temp @ config.C)
            + config.Q
            - L.T @ (inv(K) @ L)
        )

        return -1 * ret.reshape(config.n ** 2)

    a_hat = odeint(
        fun,
        config.G.reshape(config.n ** 2),
        np.linspace(config.T, 0, config.sim_steps),
        tfirst=True
    ).reshape(config.sim_steps, config.n, config.n)[::-1]

    K = config.D.T @ (a_hat @ config.D) + config.R
    L = config.B.T @ a_hat + config.D.T @ (a_hat @ config.C) + config.S

    def fun(t, b):
        a = config.interp(a_hat, t)

        K_temp = config.interp(K, t)

        L_temp = config.interp(L, t)

        M = config.B.T @ b + config.D.T @ (a @ config.sigma) + config.p

        ret = (
            config.A.T @ b
            + a @ config.gamma
            + config.C.T @ (a @ config.sigma)
            + config.q
            - L_temp.T @ (inv(K_temp) @ M)
        )

        return -1 * ret.reshape(config.n)

    b_hat = odeint(
        fun,
        config.g_vec,
        np.linspace(config.T, 0, config.sim_steps),
        tfirst=True
    ).reshape(config.sim_steps, config.n)[::-1]

    M1 = b_mult(np.ones((config.sim_steps, 1, 1)) * config.B.T, b_hat)
    M2 = b_mult(np.ones((config.sim_steps, 1, 1)) * config.D.T, a_hat @ config.sigma) + config.p

    M = M1 + M2

    def fun(t, xi):
        a = config.interp(a_hat, t)
        b = config.interp(b_hat, t)

        K_temp = config.interp(K, t)

        M_temp = config.interp(M, t)
        ret = (
            b @ config.gamma
            + 0.5 * config.sigma @ (a @ config.sigma)
            - 0.5 * M_temp.T @ (inv(K_temp) @ M_temp)
        )

        return -1 * ret

    xi_hat = odeint(
        fun,
        0,
        np.linspace(config.T, 0, config.sim_steps),
        tfirst=True
    ).reshape(config.sim_steps)[::-1]

    alpha_hat = - inv(K) @ L

    beta_hat = - b_mult(inv(K), M)

    config.alpha_hat = alpha_hat
    config.beta_hat = beta_hat
    config.a_hat = a_hat
    config.b_hat = b_hat
    config.xi_hat = xi_hat

    log(f'Solve time: {time.time() - start_time:.1f}s')

    def phi(t, X):
        alpha = config.interp(alpha_hat, t)
        beta = config.interp(beta_hat, t)
        ones_vec = np.ones(shape=np.stack([np.shape(X)[0], 1]), dtype=np.float64)
        ones_mat = np.ones(shape=np.stack([np.shape(X)[0], 1, 1]), dtype=np.float64)
        return b_mult(ones_mat * alpha, X) + ones_vec * beta

    def value(t, X):
        a = config.interp(a_hat, t)
        b = config.interp(b_hat, t)
        xi = config.interp(xi_hat, t)
        ones_vec = np.ones(shape=np.stack([np.shape(X)[0], 1]), dtype=np.float64)
        ones_mat = np.ones(shape=np.stack([np.shape(X)[0], 1, 1]), dtype=np.float64)
        return np.squeeze(0.5 * b_quad(X, ones_mat * a, X) + b_dot(ones_vec * b, X) + xi)

    return phi, value


def solve_LQ_cone(config):
    start_time = time.time()

    A = config.A
    B = config.B
    C = config.C
    D = config.D
    Q = 0.5 * config.Q
    R = 0.5 * config.R
    S = 0.5 * config.S
    G = 0.5 * config.G

    def H_plus(t, u_vec, P):
        u = u_vec.reshape(config.m, config.n)
        return np.trace(u.T @ ((D.T @ (P @ D) + R) @ u) + 2 * u.T @  (B.T @ P + D.T @ (P @ C) + S))

    def H_minus(t, u_vec, P):
        u = u_vec.reshape(config.m, config.n)
        return np.trace(u.T @ ((D.T @ (P @ D) + R) @ u) - 2 * u.T @  (B.T @ P + D.T @ (P @ C) + S))

    def arg_plus(t, P):
        return minimize(lambda u: H_plus(t, u, P), np.zeros(config.m * config.n), bounds=[(0 if config.cone else -np.inf, np.inf)]*config.m*config.n).x

    def arg_minus(t, P):
        return minimize(lambda u: H_minus(t, u, P), np.zeros(config.m * config.n), bounds=[(0 if config.cone else -np.inf, np.inf)]*config.m*config.n).x

    def fun_plus(t, P_vec):
        P = P_vec.reshape(config.n, config.n)
        return - (
            P @ A
            + A.T @ P
            + C.T @ (P @ C)
            + Q + H_plus(t, arg_plus(t, P), P)
        ).reshape(config.n ** 2)

    def fun_minus(t, P_vec):
        P = P_vec.reshape(config.n, config.n)
        return - (
            P @ A
            + A.T @ P
            + C.T @ (P @ C)
            + Q + H_minus(t, arg_minus(t, P), P)
        ).reshape(config.n ** 2)

    P_plus = odeint(
        fun_plus,
        G.reshape(config.n ** 2),
        np.linspace(config.T, 0, config.sim_steps),
        tfirst=True
    ).reshape(config.sim_steps, config.n, config.n)[::-1]

    P_minus = odeint(
        fun_minus,
        G.reshape(config.n ** 2),
        np.linspace(config.T, 0, config.sim_steps),
        tfirst=True
    ).reshape(config.sim_steps, config.n, config.n)[::-1]

    T = np.linspace(0, config.T, config.sim_steps)

    alpha_plus = [arg_plus(T[i], P_plus[i]).reshape(config.m, config.n) for i in range(config.sim_steps)]
    alpha_minus = [arg_minus(T[i], P_minus[i]).reshape(config.m, config.n) for i in range(config.sim_steps)]

    log(f'Solve time: {time.time() - start_time:.1f}s')

    def phi(t, X):
        a_p = config.interp(alpha_plus, t)
        a_m = config.interp(alpha_minus, t)

        ones_mat = np.ones(shape=np.stack([np.shape(X)[0], 1, 1]), dtype=np.float64)
        return (
            b_mult(ones_mat * a_p, np.maximum(X, X * 0.0)) +
            b_mult(ones_mat * a_m, np.maximum(-X, X * 0.0))
        )

    def value(t, X):
        P_p = config.interp(P_plus, t)
        P_m = config.interp(P_minus, t)

        ones_mat = np.ones(shape=np.stack([np.shape(X)[0], 1, 1]), dtype=np.float64)
        return np.squeeze(
            b_quad(np.maximum(X, X * 0.0), ones_mat * P_p, np.maximum(X, X * 0.0)) +
            b_quad(np.maximum(-X, X * 0.0), ones_mat * P_m, np.maximum(-X, X * 0.0))
        )

    return phi, value
