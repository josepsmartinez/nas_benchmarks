import typing

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np

from .interface import TabularNasBenchmark


class DummyBenchmark(TabularNasBenchmark):
    """Implements Lévi function N. 13 on TabularNasBenchmark interface"""
    @staticmethod
    def get_configuration_space(
        seed=None
    ) -> CS.ConfigurationSpace:
        CS.ConfigurationSpace
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(
            CSH.UniformFloatHyperparameter(name='x', lower=-10, upper=10)
        )
        cs.add_hyperparameter(
            CSH.UniformFloatHyperparameter(name='y', lower=-10, upper=10)
        )
        return cs

    @classmethod
    def get_configuration_space_relaxed(
        cls, seed=None
    ) -> CS.ConfigurationSpace:
        return cls.get_configuration_space(seed=seed)

    def relax_configuration(self, config):
        return config

    def get_best_configuration(
        self
    ) -> typing.Dict:
        # TODO: cant we get it once at __init__?
        return CS.Configuration(
            self.get_configuration_space(),
            values = {'x': 0, 'y': 0}
        )

    def objective_function(self, config, **objfn_kw) -> typing.Dict:
        x, y = config.get('x'), config.get('y')
        return {
            'config': config,
            'loss': (
                np.square(np.sin(3*np.pi*x)) +
                np.square(x - 1) * (1 + np.square(np.sin(3*np.pi*y))) +
                np.square(y - 1) * (1 + np.square(np.sin(2*np.pi*y)))
            ),
            'cost': .0
        }

    def objective_function_learning_curve(
        self,
        config: CS.Configuration, **objfn_kw
    ) -> typing.Dict:
        return self.objective_function(config)

    def objective_function_deterministic(
        self,
        config: CS.Configuration,
        idx: int = 0, **objfn_kw
    ) -> typing.Dict:
        return self.objective_function(config)

    def objective_function_test(
        self,
        config: CS.Configuration, **objfn_kw
    ) -> typing.Dict:
        # TODO: do we need it?
        return self.objective_function(config)

    def get_num_valid_configs(self) -> int:
        raise np.inf
