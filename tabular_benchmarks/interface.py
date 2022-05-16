import ConfigSpace
import numpy as np

class TabularNasBenchmark(object):
    def __init__(self, seed=42):
        self.seed = seed

        self.reset_tracker()

    @staticmethod
    def get_configuration_space(seed=None):
        raise NotImplementedError()

    def get_best_configuration(self):
        raise NotImplementedError()

    def get_benchmark_budget(self):
        raise NotImplementedError()

    def objective_function(self, config):
        raise NotImplementedError()

    def objective_function_learning_curve(self, config):
        raise NotImplementedError()

    def objective_function_deterministic(self, config, idx=0):
        raise NotImplementedError()

    def objective_function_test(self, config):
        raise NotImplementedError()

    def get_results(self):
        raise NotImplementedError()

    def reset_tracker(self):
        self.rng = np.random.RandomState(self.seed)

    def __enter__(self):
        self.reset_tracker()
        return self

    def __exit__(self):
        self.reset_tracker()
