import tensorflow as tf
import numpy as np
from common import log, H2, sim_value
import time
from deep import DeepModel
from linear import LinearModel


rng = np.random.default_rng()
tf .get_logger().setLevel('ERROR')  # remove warning messages for taking second derivatives
np.set_printoptions(precision=3)


# solver

class PPGMSolver(object):
    def __init__(self, config, model='deep'):

        self.start_time = time.time()
        self.config = config
        self.modelType = {'deep': DeepModel, 'linear': LinearModel}[model]
        self.model = self.modelType(config)
        self.value = lambda t, X: sim_value(self.config, self.phi, X)

    def train(self):
        log('Running PPGM Algorithm')
        # begin sgd iteration

        data = {
            'loss_bsde': [],
            'loss_control': [],
            'times': [],
            'H2': [],
            'H2_rel': []
        }

        step = 0

        log(
            f"Step: {0:6d} \t Time: {time.time() - self.start_time:5.2f} ")
        # initialise
        loss_bsde, loss_control = self.loss()
        try:
            try:
                old_model = self.modelType(self.config)
                old_phi = old_model.control

                old_model.gen()
                diff = 1
                tolerance = 1e-10 if self.config.optimisation_steps_bsde <= 10 else 1e-5

                while step <= self.config.iteration_steps - 1 and (diff > tolerance if step > 10 else True):
                    log(
                        f"Step: {step + 1:6d} \t Time: {time.time() - self.start_time:5.2f} \t  error: {data['H2'][-1] if len(data['H2'])>0 else 0:.3e}")
                    old_model.set_weights(self.model.get_weights())

                    # train BSDE

                    log('BSDE step')
                    for substep in range(self.config.optimisation_steps_bsde):
                        display = (substep + 1) % min(self.config.display_step,
                                                      self.config.optimisation_steps_bsde) == 0 or substep == 0

                        loss_bsde, loss_control = self.bsde_step()

                        if display:
                            log(f"Substep: {substep + 1:6d} \t Time: {time.time() - self.start_time:5.2f} \t  bsde: {loss_bsde: .3e} \t control: {loss_control: .3e}")

                    data['loss_bsde'].append(loss_bsde.numpy())

                    log('Control step')
                    if self.model.optimiser_control is None:
                        substeps = 1
                    else:
                        substeps = self.config.optimisation_steps_control

                    for substep in range(substeps):
                        display = (substep + 1) % min(self.config.display_step,
                                                      self.config.optimisation_steps_control) == 0 or substep == 0

                        loss_bsde, loss_control = self.control_step(old_model.phi)

                        if display:
                            log(f"Substep: {substep + 1:6d} \t Time: {time.time() - self.start_time:5.2f} \t  bsde: {loss_bsde: .3e} \t control: {loss_control: .3e}")

                    diff = H2(self.config, self.phi, old_phi) / (H2(self.config, old_phi) + 1e-4)
                    data['H2'].append(H2(self.config, self.phi, self.config.phi))
                    data['H2_rel'].append(diff)
                    data['loss_control'].append(loss_control.numpy())
                    data['times'].append(time.time() - self.start_time)

                    step += 1

            except Exception as e:
                log('Termination due to ', e)
                pass
        except BaseException:
            log('Terminated Manually')
            pass

        if diff < tolerance:
            log(f'Suspected plateau, terminated at step {step + 1}')
        elif step == self.config.iteration_steps:
            log(f'Reached max iteration, terminated at step {step + 1}')

        log(
            f"Step: {step + 1:6d} \t Time: {time.time() - self.start_time:5.2f} \t  bsde: {loss_bsde: .3e} \t control: {loss_control: .3e} \t  error: {data['H2'][-1] if len(data['H2'])>0 else 0:.3e}")
        self.data = data

    @tf.function
    def bsde_step(self):
        return self.model.bsde_step()

    @tf.function
    def control_step(self, old_phi):
        return self.model.control_step(old_phi)

    def loss(self):
        return self.model.loss()

    def phi(self, t, X):
        return self.model.control(t, X)


if __name__ == '__main__':
    pass
