import json
import os

import ConfigSpace
import h5py
import numpy as np

from .interface import TabularNasBenchmark

class FCNetBenchmark(TabularNasBenchmark):
    MAX_EPOCHS = 100

    def __init__(
        self,
        path,
        dataset="fcnet_protein_structure_data.hdf5",
        seed=None
    ):
        self.cs = self.get_configuration_space(seed=seed)
        self.names = [h.name for h in self.cs.get_hyperparameters()]

        self.data = h5py.File(os.path.join(path, dataset), "r")

        super().__init__(seed=seed)

    def get_benchmark_min_budget(self):
        return 3

    def get_benchmark_max_budget(self):
        return 100

    def reset_tracker(self):
        self.X = []
        self.y = []
        self.c = []
        self.rng = np.random.RandomState(self.seed)

    def get_best_configuration(self):
        """Returns the best configuration in the dataset that
        achieves the lowest test performance.

        :return: Returns tuple with the best configuration,
        its final validation performance and its test performance
        """

        configs, te, ve = [], [], []
        for k in self.data.keys():
            configs.append(json.loads(k))
            te.append(np.mean(self.data[k]["final_test_error"]))
            ve.append(np.mean(self.data[k]["valid_mse"][:, -1]))

        b = np.argmin(te)

        return configs[b], ve[b], te[b]

    def objective_function(self, config, budget=None, **kwargs):
        if budget is None:
            budget = self.get_benchmark_max_budget()
        assert budget <= self.get_benchmark_max_budget()

        i = self.rng.randint(4)

        if type(config) == ConfigSpace.Configuration:
            k = json.dumps(config.get_dictionary(), sort_keys=True)
        else:
            k = json.dumps(config, sort_keys=True)

        valid = self.data[k]["valid_mse"][i]
        runtime = self.data[k]["runtime"][i]

        time_per_epoch = runtime / self.MAX_EPOCHS

        rt = time_per_epoch * budget

        self.X.append(config)
        self.y.append(valid[budget - 1])
        self.c.append(rt)

        return valid[budget - 1], rt

    def objective_function_learning_curve(self, config, budget=None):
        if budget is None:
            budget = self.get_benchmark_max_budget()
        assert budget <= self.get_benchmark_max_budget()

        index = self.rng.randint(4)

        if type(config) == ConfigSpace.Configuration:
            k = json.dumps(config.get_dictionary(), sort_keys=True)
        else:
            k = json.dumps(config, sort_keys=True)

        lc = [self.data[k]["valid_mse"][index][i] for i in range(budget)]
        runtime = self.data[k]["runtime"][index]

        time_per_epoch = runtime / self.MAX_EPOCHS

        rt = [time_per_epoch * (i + 1) for i in range(budget)]

        self.X.append(config)
        self.y.append(lc[-1])
        self.c.append(rt[-1])

        return lc, rt

    def objective_function_deterministic(
        self,
        config,
        budget=None,
        index=0,
        **kwargs
    ):
        if budget is None:
            budget = self.get_benchmark_max_budget()
        assert budget <= self.get_benchmark_max_budget()

        if type(config) == ConfigSpace.Configuration:
            k = json.dumps(config.get_dictionary(), sort_keys=True)
        else:
            k = json.dumps(config, sort_keys=True)

        valid = self.data[k]["valid_mse"][index]
        runtime = self.data[k]["runtime"][index]

        time_per_epoch = runtime / self.MAX_EPOCHS

        rt = time_per_epoch * budget

        self.X.append(config)
        self.y.append(valid[budget - 1])
        self.c.append(rt)

        return valid[budget - 1], rt

    def objective_function_test(self, config, **kwargs):
        if type(config) == ConfigSpace.Configuration:
            k = json.dumps(config.get_dictionary(), sort_keys=True)
        else:
            k = json.dumps(config, sort_keys=True)

        test = np.mean(self.data[k]["final_test_error"])
        runtime = np.mean(self.data[k]["runtime"])

        return test, runtime

    def get_results(self):
        inc, y_star_valid, y_star_test = self.get_best_configuration()

        regret_validation = []
        regret_test = []
        runtime = []
        rt = 0

        inc_valid = np.inf
        inc_test = np.inf

        for i in range(len(self.X)):
            if inc_valid > self.y[i]:
                inc_valid = self.y[i]
                inc_test, _ = self.objective_function_test(self.X[i])

            regret_validation.append(float(inc_valid - y_star_valid))
            regret_test.append(float(inc_test - y_star_test))
            rt += self.c[i]
            runtime.append(float(rt))

        res = dict()
        res['regret_validation'] = regret_validation
        res['regret_test'] = regret_test
        res['runtime'] = runtime

        return res

    def get_num_valid_configs(self):
        return len(self.X)

    @staticmethod
    def get_configuration_space(seed=None):
        cs = ConfigSpace.ConfigurationSpace(seed=seed)
        cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter("n_units_1", [16, 32, 64, 128, 256, 512]))
        cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter("n_units_2", [16, 32, 64, 128, 256, 512]))
        cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter("dropout_1", [0.0, 0.3, 0.6]))
        cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter("dropout_2", [0.0, 0.3, 0.6]))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("activation_fn_1", ["tanh", "relu"]))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("activation_fn_2", ["tanh", "relu"]))
        cs.add_hyperparameter(
            ConfigSpace.OrdinalHyperparameter("init_lr", [5 * 1e-4, 1e-3, 5 * 1e-3, 1e-2, 5 * 1e-2, 1e-1]))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("lr_schedule", ["cosine", "const"]))
        cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter("batch_size", [8, 16, 32, 64]))
        return cs

    @staticmethod
    def get_configuration_space_relaxed(seed=None):
        cs = ConfigSpace.ConfigurationSpace(seed=seed)
        cs.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter("n_units_1", lower=0, upper=5))
        cs.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter("n_units_2", lower=0, upper=5))
        cs.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter("dropout_1", lower=0, upper=2))
        cs.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter("dropout_2", lower=0, upper=2))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("activation_fn_1", ["tanh", "relu"]))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("activation_fn_2", ["tanh", "relu"]))
        cs.add_hyperparameter(
            ConfigSpace.UniformIntegerHyperparameter("init_lr", lower=0, upper=5))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("lr_schedule", ["cosine", "const"]))
        cs.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter("batch_size", lower=0, upper=3))
        return cs

    def relax_configuration(self, config):
        c = self.cs.sample_configuration()
        c["n_units_1"] = self.cs.get_hyperparameter("n_units_1").sequence[
            config["n_units_1"]]
        c["n_units_2"] = self.cs.get_hyperparameter("n_units_2").sequence[
            config["n_units_2"]]
        c["dropout_1"] = self.cs.get_hyperparameter("dropout_1").sequence[
            config["dropout_1"]]
        c["dropout_2"] = self.cs.get_hyperparameter("dropout_2").sequence[
            config["dropout_2"]]
        c["init_lr"] = self.cs.get_hyperparameter("init_lr").sequence[
            config["init_lr"]]
        c["batch_size"] = self.cs.get_hyperparameter("batch_size").sequence[
            config["batch_size"]]
        c["activation_fn_1"] = config["activation_fn_1"]
        c["activation_fn_2"] = config["activation_fn_2"]
        c["lr_schedule"] = config["lr_schedule"]
        return c


class FCNetSliceLocalizationBenchmark(FCNetBenchmark):
    def __init__(self, data_dir="./", **kw):
        super(FCNetSliceLocalizationBenchmark, self).__init__(
            path=data_dir,
            dataset="fcnet_slice_localization_data.hdf5",
            **kw)


class FCNetProteinStructureBenchmark(FCNetBenchmark):
    def __init__(self, data_dir="./", **kw):
        super(FCNetProteinStructureBenchmark, self).__init__(
            path=data_dir,
            dataset="fcnet_protein_structure_data.hdf5",
            **kw)


class FCNetNavalPropulsionBenchmark(FCNetBenchmark):
    def __init__(self, data_dir="./", **kw):
        super(FCNetNavalPropulsionBenchmark, self).__init__(
            path=data_dir,
            dataset="fcnet_naval_propulsion_data.hdf5",
            **kw)


class FCNetParkinsonsTelemonitoringBenchmark(FCNetBenchmark):
    def __init__(self, data_dir="./", **kw):
        super(FCNetParkinsonsTelemonitoringBenchmark, self).__init__(
            path=data_dir,
            dataset="fcnet_parkinsons_telemonitoring_data.hdf5",
            **kw)
