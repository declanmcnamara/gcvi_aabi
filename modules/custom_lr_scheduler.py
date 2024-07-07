# From:
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/132907dd272e2cc92e3c10e6c4e783a87ff8893d/transformer/Optim.py#L4

'''A wrapper class for scheduled optimizer '''
import numpy as np

class CustomOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, start_lr=1e-3):
        self._optimizer = optimizer
        self.start_lr = start_lr
        self.dummy = 1/self.start_lr
        self.n_steps = 0


    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = 1/(self.dummy + self.n_steps/10)

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

class CustomOptim2():
    '''A simple wrapper class for learning rate scheduling
    Bigger jumps in decrease in learning rate here, still
    square summable
    '''

    def __init__(self, optimizer, start_lr=1e-3, modulus=1000, dummy_increment=2):
        self._optimizer = optimizer
        self.start_lr = start_lr
        self.dummy = 2.
        self.n_steps = 0
        self.modulus = modulus
        self.dummy_increment = dummy_increment


    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1

        if self.n_steps % 500 == 0:
            lr = self.start_lr/self.dummy
            self.dummy += 2
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = lr