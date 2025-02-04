import numpy as np
from scipy.integrate import odeint
import time
from matplotlib import pyplot as plt, rcParams
from matplotlib.ticker import MaxNLocator
from common import log, b_mult, saveplot, H2, b_quad, b_dot, sim_value
from config import LQConfig


rcParams['figure.dpi'] = 600
np.set_printoptions(precision=3)


def mod(X):
    # returns op-norm for matrices, euclidean norm for vectors
    if len(X.shape) == 1:
        return np.sqrt(np.sum(X ** 2))
    else:
        return np.real(np.sqrt(np.max(np.linalg.eigvals(X.T @ X))))


def mod0(alpha, beta=None):
    alpha0 = max(map(mod, alpha))
    if beta is not None:
        beta0 = max(map(mod, beta))
        return max(alpha0, beta0)
    else:
        return alpha0


def inv(A):
    if len(A.shape) <= 2:
        A_T = A.T
    else:
        A_T = np.transpose(A, axes=[0, 2, 1])
    return np.linalg.inv(A_T @ A + 1e-8 * np.eye(A.shape[-1])) @ A_T


class Algo(object):
    def __init__(self, config, loud=True, plot=True, rand=False, H2=True):
        self.config = config

        # output config
        self.loud = loud
        self.plot = plot
        self.H2 = H2

        self.print = log if self.loud else (lambda *args, **kwargs: None)
        self.warn(config.cone == False, "Problem must be unconstrained")
        self.soln = self.ode_value(config.a_hat[0], config.b_hat[0], config.xi_hat[0])

        x1 = np.concatenate([self.config.Q, self.config.S.T], axis=1)
        x2 = np.concatenate([self.config.S, self.config.R], axis=1)

        x = np.concatenate([x1, x2], axis=0)

        self.warn(np.all(np.linalg.eigvals(x) > 0),
                  'Running matrix positive definite')
        self.setup()

        self.value = lambda t, X: sim_value(self.config, self.phi, X)

    def ode_value(self, a, b, xi):

        # gets value, given intial matrix a
        x = np.ones(self.config.n)
        return 0.5 * (a @ x) @ x + b @ x + xi

    def value(self, t, X):

        a = self.config.interp(self.a, t)
        b = self.config.interp(self.b, t)
        xi = self.config.interp(self.config.xi_hat, t)

        ones_vec = np.ones(shape=np.stack([np.shape(X)[0], 1]), dtype=np.float64)
        ones_mat = np.ones(shape=np.stack([np.shape(X)[0], 1, 1]), dtype=np.float64)
        return 0.5 * b_quad(X, ones_mat * a, X) + b_dot(ones_vec * b, X) + xi

    def phi(self, t, X):
        alpha = self.config.interp(self.alpha, t)
        beta = self.config.interp(self.beta, t)

        ones_vec = np.ones(shape=np.stack([np.shape(X)[0], 1]), dtype=np.float64)
        ones_mat = np.ones(shape=np.stack([np.shape(X)[0], 1, 1]), dtype=np.float64)
        return b_mult(ones_mat * alpha, X) + ones_vec * beta

    def run(self):
        self.start_time = time.time()
        self.iterate()
        self.final_time = time.time() - self.start_time
        self.print(f'Run finished at time {self.final_time:.1f} seconds')
        self.run_plots()
        return self

    def warn(self, foo, err):
        if not foo:
            self.print('warning: assumption not true: ' + err)
            self.error = True

    def setup(self):
        self.print(f'Setting up problem with n = {self.config.n}, m = {self.config.m}',
                   f', r = {self.config.r}' if hasattr(self.config, 'r') else "")
        # initialise
        self.alpha = np.ones((self.config.sim_steps, self.config.m, self.config.n)) / \
            (max(self.config.n, 5) * max(self.config.m, 5))
        self.beta = np.ones((self.config.sim_steps, self.config.m)) / max(self.config.m, 5)

        self.phis = []

        # calculate constants
        mu = self.config.mu
        self.tau = self.config.tau

        self.error = False

        self.warn(mu > 0, 'mu > 0')

        temp = self.error
        self.error = False

        self.wont_converge = self.error

        self.error = (temp or self.error)

        def fun_a(t, a):
            a_temp = a.reshape(self.config.n, self.config.n)
            alpha_temp = self.config.interp(self.alpha, t)
            ret = (
                a_temp @ (self.config.A + self.config.B @ alpha_temp)
                + self.config.A.T @ a_temp
                + self.config.C.T @ (a_temp @ (self.config.C + self.config.D @ alpha_temp))
                + self.config.Q + self.config.S.T @ alpha_temp
            )

            return -1 * ret.reshape(self.config.n ** 2)

        self.a = odeint(
            fun_a,
            self.config.G.reshape(self.config.n ** 2),
            np.linspace(self.config.T, 0, self.config.sim_steps),
            tfirst=True
        ).reshape(self.config.sim_steps, self.config.n, self.config.n)[::-1]

        # update b

        def fun_b(t, b):
            a_temp = self.config.interp(self.a, t)
            b_temp = b
            beta_temp = self.config.interp(self.beta, t)
            ret = (
                self.config.A.T @ b_temp
                + (a_temp @ self.config.B + self.config.C.T @
                   (a_temp @ self.config.D) + self.config.S.T) @ beta_temp
                + a_temp @ self.config.gamma +
                self.config.C.T @ (a_temp @ self.config.sigma) + self.config.q
            )
            return -1 * ret

        self.b = odeint(
            fun_b,
            self.config.g_vec,
            np.linspace(self.config.T, 0, self.config.sim_steps),
            tfirst=True
        ).reshape(self.config.sim_steps, self.config.n)[::-1]

        return self

    def iterate(self):

        try:

            times = [0]
            values = [None]

            step = 1
            diff = 1
            error = 0
            alpha = self.alpha
            beta = self.beta
            a = self.a
            b = self.b
            self.error0 = mod0(self.alpha - self.config.alpha_hat, self.beta -
                               self.config.beta_hat) / (mod0(self.config.alpha_hat, self.config.beta_hat) + 1e-4)

            time_now = time.time() - self.start_time

            self.print(f'Solution: {self.soln:.2e}')
            self.print(f'Initial control error: {self.error0:.2e}')
            self.print(f'Learning rate: {self.tau:.2e}')

            self.print('step \t cont err \t val err \t time')
            self.print(
                f'{0:4d} \t {self.error0:.2e}  \t -------- \t {int(time_now):4d}')

            errors = {
                'control': [self.error0],
                'a': [None],
                'value': [None]
            }

            H2_errors = []
            H2_rel = []
            # iterate
            while step <= self.config.iteration_steps - 1 and (diff > 1e-6 if step > 10 else True) and error <= 1e10:

                def phi(t, X):
                    ones_vec = np.ones(shape=np.stack([np.shape(X)[0], 1]), dtype=np.float64)
                    ones_mat = np.ones(shape=np.stack([np.shape(X)[0], 1, 1]), dtype=np.float64)
                    return b_mult(ones_mat * np.array(alpha[0]), X) + ones_vec * np.array(beta[0])

                def value(t, X):
                    ones_vec = np.ones(shape=np.stack([np.shape(X)[0], 1]), dtype=np.float64)
                    ones_mat = np.ones(shape=np.stack([np.shape(X)[0], 1, 1]), dtype=np.float64)
                    return 0.5 * b_quad(X, ones_mat * a[0], X) + b_dot(ones_vec * b[0], X) + self.config.xi_hat[0]

                self.phis.append([alpha, beta])

                step += 1

                old_alpha = alpha
                old_beta = beta

                def old_phi(t, X):
                    a = self.config.interp(old_alpha, t)
                    b = self.config.interp(old_beta, t)
                    ones_vec = np.ones(shape=np.stack([np.shape(X)[0], 1]), dtype=np.float64)
                    ones_mat = np.ones(shape=np.stack([np.shape(X)[0], 1, 1]), dtype=np.float64)
                    return b_mult(ones_mat * a, X) + ones_vec * b
                # update a

                def fun_a(t, a):
                    a_temp = a.reshape(self.config.n, self.config.n)
                    alpha_temp = self.config.interp(alpha, t)
                    ret = (
                        a_temp @ (self.config.A + self.config.B @ alpha_temp)
                        + self.config.A.T @ a_temp
                        + self.config.C.T @ (a_temp @ (self.config.C + self.config.D @ alpha_temp))
                        + self.config.Q + self.config.S.T @ alpha_temp
                    )

                    return -1 * ret.reshape(self.config.n ** 2)

                a = odeint(
                    fun_a,
                    self.config.G.reshape(self.config.n ** 2),
                    np.linspace(self.config.T, 0, self.config.sim_steps),
                    tfirst=True
                ).reshape(self.config.sim_steps, self.config.n, self.config.n)[::-1]

                # update b

                def fun_b(t, b):
                    a_temp = self.config.interp(a, t)
                    b_temp = b
                    beta_temp = self.config.interp(beta, t)
                    ret = (
                        self.config.A.T @ b_temp
                        + (a_temp @ self.config.B + self.config.C.T @
                           (a_temp @ self.config.D) + self.config.S.T) @ beta_temp
                        + a_temp @ self.config.gamma +
                        self.config.C.T @ (a_temp @ self.config.sigma) + self.config.q
                    )
                    return -1 * ret

                b = odeint(
                    fun_b,
                    self.config.g_vec,
                    np.linspace(self.config.T, 0, self.config.sim_steps),
                    tfirst=True
                ).reshape(self.config.sim_steps, self.config.n)[::-1]

                # update alpha

                alpha = (
                    alpha
                    - self.tau * (
                        self.config.B.T @ a
                        + self.config.D.T @ (a @ (self.config.C + self.config.D @ alpha))
                        + self.config.R @ alpha
                        + self.config.S
                    )
                )

                # update beta

                beta = (
                    beta
                    - self.tau * np.squeeze(
                        self.config.B.T @ np.expand_dims(b, -1)
                        + (
                            self.config.D.T @ (a @ self.config.D)
                            + self.config.R
                        ) @ np.expand_dims(beta, -1)
                        + self.config.D.T @ np.expand_dims(a @ self.config.sigma, -1)
                        + np.expand_dims(self.config.p, -1),
                        -1)
                )

                # calculate error

                error = mod0(alpha - self.config.alpha_hat, beta -
                             self.config.beta_hat) / (mod0(self.config.alpha_hat, self.config.beta_hat) + 1e-4)

                diff = mod0(alpha - old_alpha, beta -
                            old_beta) / (mod0(old_alpha, old_beta) + 1e-4)

                error_a = mod0(a - self.config.a_hat) / \
                    (mod0(self.config.a_hat) + 1e-8)

                curr_value = self.ode_value(a[0], b[0], self.config.xi_hat[0])

                time_now = time.time() - self.start_time
                rel_err = np.abs(curr_value - self.soln) / np.abs(self.soln)

                times.append(time_now)
                errors['control'].append(error)
                errors['a'].append(error_a)
                errors['value'] .append(rel_err)
                values.append(curr_value)

                self.alpha = alpha
                self.beta = beta
                self.a = a
                self.b = b
                if self.H2:
                    H2_errors.append(H2(self.config, self.phi, self.config.phi) if self.config.phi is not None else 0)
                    H2_rel.append(H2(self.config, self.phi, old_phi) / H2(self.config, old_phi))

                self.print(
                    f'{step:4d} \t {error:.2e} \t {rel_err:.2e} \t {int(time_now):4d}')

        except KeyboardInterrupt:
            self.print(f'Manually terminated at step {step}')

        if diff < 1e-6:
            self.print(f'Suspected plateau, terminated at step {step}')
        elif step == self.config.iteration_steps:
            self.print(f'Reached max iteration, terminated at step {step}')

        # print final error

        errorf = mod0(alpha - self.config.alpha_hat, beta - self.config.beta_hat)

        self.print(f'Final (absolute) control error: {errorf:.5e}')

        approx = self.ode_value(a[0], b[0], self.config.xi_hat[0])
        err = np.abs(approx - self.soln) / self.soln

        self.print(
            f'Solution: {self.soln:.5e}, Approx: {approx:.5e}, rel err: {err:.2e} ({100 * err:.2f}%)'
        )

        self.data = {
            'errors': errors,
            'times': times,
            'values': values,
            'steps': np.arange(len(times)),
            'H2': H2_errors,
            'H2_rel': H2_rel
        }

    def run_plots(self):
        if not self.plot:
            return self

        times = self.data['times']
        errors = self.data['errors']

        steps = np.arange(len(times))

        # plot control error

        plt.figure()
        plt.grid(True, alpha=0.5)
        plt.plot(steps, errors['control'], label='error')
        plt.yscale('log')
        plt.legend()
        plt.xlabel('Step ($k$)')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.ylabel(r'Relative Error of $\alpha$')
        saveplot()

        # plot matrix error

        plt.figure()
        plt.grid(True, alpha=0.5)
        plt.plot(steps, errors['a'], label='error')
        plt.yscale('log')
        plt.legend()
        plt.xlabel('Step ($k$)')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.ylabel(r'Relative Error of $a$')
        saveplot()

        return self


if __name__ == '__main__':
    repeats = 1

    config = LQConfig(n=10, m=10)
    pgm = Algo(config, H2=False)
    pgm.run()

    run_times = []
    solve_times = []
    dim = 60
    dims = [max(1, x) for x in np.arange(0, dim + 1, 5)]

    for n in dims:
        rTime = 0
        config = LQConfig(n=n, m=n)
        for i in range(repeats):
            result = Algo(config, plot=False, H2=False).run()
            rTime += result.final_time / repeats
            del result
        run_times.append(rTime)
        solve_times.append(config.solve_time)

    plt.figure()
    plt.grid(0.5)
    plt.plot(dims, run_times, label='Run PGM')
    # plt.plot(dims, solve_times, label='Solve Ricatti')
    plt.xlabel('Dimension')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().set_xticks(np.arange(0, dim + 1, int(dim / 5)))
    plt.ylabel('Runtime')
    saveplot()

    plt.figure()
    plt.grid(0.5)
    plt.plot(dims, run_times, label='Run PGM')
    # plt.plot(dims, solve_times, label='Solve Ricatti')
    plt.xlabel('Dimension')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().set_xticks(np.arange(0, dim + 1, int(dim / 5)))
    plt.ylabel('Runtime')
    plt.yscale('log')
    saveplot()

    xaxis = []
    errors_c = []
    errors_a = []

    for rs in [np.linspace(0.05, 0.2, 20), np.linspace(0.2, 2.0, 20)]:

        for r in rs:
            xaxis.append(r)
            err_c = 0
            err_a = 0
            try:
                config = LQConfig(n=5, m=5, r=r)
                for i in range(repeats):
                    result = Algo(config, plot=False, H2=False).run()
                    err_c += min(result.data['errors']['control'][-1], 1e10) / repeats
                    err_a += min(result.data['errors']['a'][-1], 1e10) / repeats
                    del result
                errors_c.append(err_c)
                errors_a.append(err_a)
            except Exception as e:
                log(e)
                errors_c.append(np.NAN)
                errors_a.append(np.NAN)

    plt.figure()
    plt.grid(0.5)
    plt.plot(xaxis, errors_c)
    plt.xlabel('r')
    plt.ylabel('Control Relative Error')
    plt.yscale('log')
    saveplot()

    plt.figure()
    plt.grid(0.5)
    plt.plot(xaxis, errors_a)
    plt.xlabel('r')
    plt.ylabel('Value Relative Error')
    plt.yscale('log')
    saveplot()
