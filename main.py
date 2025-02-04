import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt, rcParams
from matplotlib.ticker import MaxNLocator
from common import log, saveplot
from config import LQConfig, FConfig
from LQ import Algo
from ppgm import PPGMSolver

rcParams['figure.dpi'] = 600
np.set_printoptions(precision=3)


def run(config, LQ=True):

    pgm = Algo(config, plot=False) if LQ else None
    if LQ:
        pgm.run()

    tf.keras.backend.clear_session()
    tf.keras.backend.set_floatx('float64')

    deep = PPGMSolver(config, model='deep')
    if deep is not None:
        deep.train()

    plot(config, deep, pgm)


def plot(config, deep, pgm=None):
    LQ = pgm is not None

    try:
        if deep is not None:
            fig = plt.figure()
            plt.grid(0.5)
            plt.plot(deep.data['loss_bsde'], label='BSDE loss', c='g')
            plt.plot(deep.data['loss_control'], label='Control loss', c='m', linestyle='dashed')
            plt.title(r'loss of PPGM PPGM method')
            plt.xlabel('Iteration Step (k)')
            plt.ylabel(r'loss')
            plt.yscale('log')
            plt.legend()
            fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            saveplot()

        fig = plt.figure()
        plt.grid(0.5)
        if deep is not None:
            plt.plot(deep.data['H2'], label='PPGM')
        if LQ and not config.cone:
            plt.plot(pgm.data['H2'], label='LQ-PGM')
        plt.title(r'$\mathcal{H}^2$ error')
        plt.xlabel('Iteration Step (k)')
        plt.ylabel(r'$||u_k - \hat{u}||^2$')
        plt.yscale('log')
        fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.legend()
        saveplot()

    except Exception as e:
        log('Plot termination due to ', e)

    comp = []
    label = []

    if deep is not None:
        comp.append(deep.phi)
        label.append('PPGM')

    if LQ:
        comp.append(pgm.phi)
        label.append('LQ-PGM')

    comp.append(config.phi)
    label.append('soln')

    config.compare(comp, label, r'$u$')

    comp = []
    label = []

    if deep is not None:
        comp.append(deep.value)
        label.append('PPGM')

    if LQ:
        comp.append(pgm.value)
        label.append('LQ-PGM')

    comp.append(config.value)
    label.append('soln')

    config.compare(comp, label, r'$v$')


if __name__ == '__main__':
    for kwargs in [{
            'optimisation_steps_bsde': 100,
            'optimisation_steps_control': 100,
            'iteration_steps': 20,
            'learning_rate_control': 0.01,
            'learning_rate_bsde': 0.01,
            'batchsize': 50,
    }]:
        run(LQConfig(n=2, m=3, **kwargs))  # ex1
        run(LQConfig(n=1, m=5, **kwargs))  # ex3
        run(LQConfig(n=1, m=5, cone=True, **kwargs))  # ex4

    for kwargs in [{
            'optimisation_steps_bsde': 10,
            'optimisation_steps_control': 10,
            'iteration_steps': 500,
            'learning_rate_control': 0.002,
            'learning_rate_bsde': 0.005,
            'batchsize': 100,
    }]:
        run(LQConfig(n=2, m=2, zero=True, **kwargs))  # ex2
        run(FConfig(n=3, m=3, **kwargs), LQ=False)  # ex5

    log('done')
