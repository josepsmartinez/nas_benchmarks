import typing

import ConfigSpace
import numpy as np

ObjectiveFunction = typing.NewType(
    'TabularNasBenchmarkObjectiveSignature',
    typing.Tuple[float, float]
)

BenchmarkResult = typing.NewType(
    'TabularNasBenchmarkResult',
    typing.Dict
)


class TabularNasBenchmark(object):
    def __init__(self, seed=42):
        self.seed = seed

        self.reset_tracker()

    @staticmethod
    def get_configuration_space(
        seed=None
    ) -> ConfigSpace.ConfigurationSpace:
        raise NotImplementedError()

    @classmethod
    def get_configuration_space_relaxed(
        cls, seed=None
    ) -> ConfigSpace.ConfigurationSpace:
        return cls.get_configuration_space(seed=seed)

    def relax_configuration(self, config):
        return config

    def get_best_configuration(
        self
    ) -> typing.Tuple[ConfigSpace.Configuration, float, float]:
        raise NotImplementedError()

    def get_benchmark_min_budget(self) -> int:
        raise NotImplementedError()

    def get_benchmark_max_budget(self) -> int:
        raise NotImplementedError()

    def objective_function(self, config) -> ObjectiveFunction:
        raise NotImplementedError()

    def objective_function_learning_curve(
        self,
        config: ConfigSpace.Configuration
    ) -> ObjectiveFunction:
        raise NotImplementedError()

    def objective_function_deterministic(
        self,
        config: ConfigSpace.Configuration,
        idx: int = 0
    ) -> ObjectiveFunction:
        raise NotImplementedError()

    def objective_function_test(
        self,
        config: ConfigSpace.Configuration
    ) -> ObjectiveFunction:
        raise NotImplementedError()

    def get_results(self) -> BenchmarkResult:
        raise NotImplementedError()

    def get_num_valid_configs(self) -> int:
        raise NotImplementedError()

    def reset_tracker(self):
        self.rng = np.random.RandomState(self.seed)

    """Context-manager interface"""
    def __enter__(self):
        self.reset_tracker()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            raise exc_value

        self.reset_tracker()
