"""
Custom class to track a metric and call a specific function when the criterion is fullfilled.
"""
from pytorch_lightning.callbacks import Callback


class ValMetricCallback(Callback):
    def __init__(self, mode, metric_name, patience, callback_fn):
        super().__init__()
        assert mode in ['min', 'max']
        self._fn = callback_fn
        self._patience = patience
        self._metric_val = None
        self._metric_name = metric_name
        self._mode = mode
        self._epochs_to_best = None

        msg = f'[{self.__class__.__name__}] {self._metric_name} {self._mode} {self._patience} -> {self._fn.__name__}()'
        print(msg)

    def is_better(self, value):
        if self._metric_val is None:
            return True
        if self._mode == 'min':
            return value < self._metric_val
        else:
            return value > self._metric_val

    def on_validation_end(self, trainer, pl_module):
        value = trainer.callback_metrics[self._metric_name]
        if self.is_better(value):
            self._metric_val = value
            self._epochs_to_best = 0
        else:
            self._epochs_to_best += 1

        if self._epochs_to_best >= self._patience:
            change_made = self._fn()
            if change_made:
                msg = f'[{self.__class__.__name__}] {self._fn.__name__}() was called '
                msg += f'due to plateauing of {self._metric_name}'
                print(msg)
            self._epochs_to_best = 0
