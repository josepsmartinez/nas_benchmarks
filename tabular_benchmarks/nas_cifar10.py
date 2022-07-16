import typing
import os

import ConfigSpace
import numpy as np
from nasbench import api
from nasbench.lib import graph_util

from .interface import TabularNasBenchmark

MAX_EDGES = 9
VERTICES = 7


class NASCifar10(TabularNasBenchmark):
    null_res = {
        'loss_valid': np.inf,
        'cost': np.inf
    }

    def __init__(self, data_dir, multi_fidelity=True):
        self.multi_fidelity = multi_fidelity
        if self.multi_fidelity:
            self.dataset = api.NASBench(os.path.join(data_dir, 'nasbench_full.tfrecord'))
        else:
            self.dataset = api.NASBench(os.path.join(data_dir, 'nasbench_only108.tfrecord'))

        self.y_star_valid = 0.04944576819737756  # lowest mean validation error
        self.y_star_test = 0.056824247042338016  # lowest mean test error

    def get_benchmark_min_budget(self):
        return 4

    def get_benchmark_max_budget(self):
        return 108

    @staticmethod
    def objective_function(self, config) -> typing.Union[typing.Dict, typing.NoReturn]:
        raise NotImplementedError()

    @staticmethod
    def get_configuration_space():
        raise NotImplementedError()

    def get_num_valid_configs(self):
        X, costs, y_valid = map(np.array, (self.X, self.costs, self.y_valid))
        return np.count_nonzero(X[(costs != 0) & np.isfinite(y_valid)])


class NASCifar10A(NASCifar10):
    def objective_function(self, config, budget=108) -> typing.Union[typing.Dict, typing.NoReturn]:
        if self.multi_fidelity is False:
            assert budget == 108

        matrix = np.zeros([VERTICES, VERTICES], dtype=np.int8)
        idx = np.triu_indices(matrix.shape[0], k=1)
        for i in range(VERTICES * (VERTICES - 1) // 2):
            row = idx[0][i]
            col = idx[1][i]
            matrix[row, col] = config["edge_%d" % i]

        # if not graph_util.is_full_dag(matrix) or graph_util.num_edges(matrix) > MAX_EDGES:
        if graph_util.num_edges(matrix) > MAX_EDGES:
            return {'config': config} | self.null_res

        labeling = [config["op_node_%d" % i] for i in range(5)]
        labeling = ['input'] + list(labeling) + ['output']
        model_spec = api.ModelSpec(matrix, labeling)
        try:
            data = self.dataset.query(model_spec, epochs=budget)
        except api.OutOfDomainError:
            return {'config': config} | self.null_res

        return {
            'config': config,
            'loss_valid': 1 - data["validation_accuracy"],
            'cost': data["training_time"]
        }

    @staticmethod
    def get_configuration_space(seed=None):
        cs = ConfigSpace.ConfigurationSpace(seed=seed)

        ops_choices = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_0", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_1", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_2", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_3", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_4", ops_choices))
        for i in range(VERTICES * (VERTICES - 1) // 2):
            cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("edge_%d" % i, [0, 1]))
        return cs


class NASCifar10B(NASCifar10):
    def objective_function(self, config, budget=108):
        if self.multi_fidelity is False:
            assert budget == 108

        bitlist = [0] * (VERTICES * (VERTICES - 1) // 2)
        for i in range(MAX_EDGES):
            bitlist[config["edge_%d" % i]] = 1
        out = 0
        for bit in bitlist:
            out = (out << 1) | bit

        matrix = np.fromfunction(graph_util.gen_is_edge_fn(out),
                                 (VERTICES, VERTICES),
                                 dtype=np.int8)
        # if not graph_util.is_full_dag(matrix) or graph_util.num_edges(matrix) > MAX_EDGES:
        if graph_util.num_edges(matrix) > MAX_EDGES:
            return {'config': config} | self.null_res

        labeling = [config["op_node_%d" % i] for i in range(5)]
        labeling = ['input'] + list(labeling) + ['output']
        model_spec = api.ModelSpec(matrix, labeling)
        try:
            data = self.dataset.query(model_spec, epochs=budget)
        except api.OutOfDomainError:
            return {'config': config} | self.null_res

        return {
            'config': config,
            'loss_valid': 1 - data["validation_accuracy"],
            'cost': data["training_time"]
        }

    @staticmethod
    def get_configuration_space(seed=None):
        cs = ConfigSpace.ConfigurationSpace(seed=seed)

        ops_choices = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_0", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_1", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_2", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_3", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_4", ops_choices))
        cat = [i for i in range((VERTICES * (VERTICES - 1)) // 2)]
        for i in range(MAX_EDGES):
            cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("edge_%d" % i, cat))
        return cs


class NASCifar10C(NASCifar10):
    def objective_function(self, config, budget=108):
        if self.multi_fidelity is False:
            assert budget == 108

        edge_prob = []
        for i in range(VERTICES * (VERTICES - 1) // 2):
            edge_prob.append(config["edge_%d" % i])

        idx = np.argsort(edge_prob)[::-1][:config["num_edges"]]
        binay_encoding = np.zeros(len(edge_prob))
        binay_encoding[idx] = 1
        matrix = np.zeros([VERTICES, VERTICES], dtype=np.int8)
        idx = np.triu_indices(matrix.shape[0], k=1)
        for i in range(VERTICES * (VERTICES - 1) // 2):
            row = idx[0][i]
            col = idx[1][i]
            matrix[row, col] = binay_encoding[i]

        if graph_util.num_edges(matrix) > MAX_EDGES:
            return {'config': config} | self.null_res

        labeling = [config["op_node_%d" % i] for i in range(5)]
        labeling = ['input'] + list(labeling) + ['output']
        model_spec = api.ModelSpec(matrix, labeling)
        try:
            data = self.dataset.query(model_spec, epochs=budget)
        except api.OutOfDomainError:
            return {'config': config} | self.null_res

        return {
            'config': config,
            'loss_valid': 1 - data["validation_accuracy"],
            'cost': data["training_time"]
        }

    @staticmethod
    def get_configuration_space(seed=None):
        cs = ConfigSpace.ConfigurationSpace(seed=seed)

        ops_choices = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_0", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_1", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_2", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_3", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_4", ops_choices))

        cs.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter("num_edges", 0, MAX_EDGES))

        for i in range(VERTICES * (VERTICES - 1) // 2):
            cs.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter("edge_%d" % i, 0, 1))
        return cs
