import typing

import ConfigSpace


class TabularNasBenchmark(object):
    def __init__(self, seed=0):
        self.seed = seed

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
    ) -> typing.Dict:
        raise NotImplementedError()

    def get_benchmark_min_budget(self) -> int:
        raise NotImplementedError()

    def get_benchmark_max_budget(self) -> int:
        raise NotImplementedError()

    def objective_function(self, config) -> typing.Dict:
        raise NotImplementedError()

    def objective_function_learning_curve(
        self,
        config: ConfigSpace.Configuration
    ) -> typing.Dict:
        raise NotImplementedError()

    def objective_function_deterministic(
        self,
        config: ConfigSpace.Configuration,
        idx: int = 0
    ) -> typing.Dict:
        raise NotImplementedError()

    def objective_function_test(
        self,
        config: ConfigSpace.Configuration
    ) -> typing.Dict:
        raise NotImplementedError()

    def get_num_valid_configs(self) -> int:
        raise NotImplementedError()
