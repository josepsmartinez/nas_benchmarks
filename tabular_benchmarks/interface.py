import typing

import ConfigSpace as CS


class TabularNasBenchmark(object):
    def __init__(self, seed=0):
        self.seed = seed

    @staticmethod
    def get_configuration_space(
        seed=None
    ) -> CS.ConfigurationSpace:
        raise NotImplementedError()

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
        raise NotImplementedError()

    def objective_function(self, config, **objfn_kw) -> typing.Dict:
        raise NotImplementedError()

    def objective_function_learning_curve(
        self,
        config: CS.Configuration, **objfn_kw
    ) -> typing.Dict:
        raise NotImplementedError()

    def objective_function_deterministic(
        self,
        config: CS.Configuration,
        idx: int = 0, **objfn_kw
    ) -> typing.Dict:
        raise NotImplementedError()

    def objective_function_test(
        self,
        config: CS.Configuration, **objfn_kw
    ) -> typing.Dict:
        raise NotImplementedError()

    def get_num_valid_configs(self) -> int:
        raise NotImplementedError()
