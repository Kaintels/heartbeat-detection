try:
    import torch
    import torch.utils.data
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Variable
    import numpy as np
    from config.parameter import *
    from models.model import Net
    from torch.optim import Adam, SGD, RMSprop
    from models.pool import Pool
    import os
    import random
    from experiments.base_train import Train
    import experiments.base_test as testing
    import pandas as pd
    from sklearn.metrics import precision_score, recall_score, accuracy_score
    from sklearn.model_selection import train_test_split

except:
    raise ImportError("For this example you need to install pytorch.")

try:
    import torchvision
    import torchvision.transforms as transforms
except:
    raise ImportError("For this example you need to install pytorch-vision.")

# data = '/'

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker

import logging

logging.basicConfig(level=logging.DEBUG)

testing_num = 0


class PyTorchWorker(Worker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        SEED = 111222
        np.random.seed(SEED)
        random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)

        self.data = data
        # data_x = self.data[:, :-1].astype('float32')
        # data_y = self.data[:, -1].astype('int32')
        data, data_val = train_test_split(data, test_size=0.3)

        self.data = data
        self.data_val = data_val

    #######################################################

    def compute(self, config, budget, working_directory, *args, **kwargs):
        self.layers = config['layers']
        self.pool = Pool(POOL_SIZE)
        self.train = Train(self.data, self.data_val, self.layers)
        flow = 0
        self.flow = flow

        model = Net(layers=self.layers)
        model_ = Net(layers=self.layers)

        if config['optimizer'] == 'Adam':
            self.train.brain.model.opt = Adam(model.parameters(), lr=config['lr'])
            self.train.brain.model_.opt = Adam(model.parameters(), lr=config['lr'])
        if config['optimizer'] == 'RMSprop':
            self.train.brain.model.opt = RMSprop(model.parameters(), lr=config['lr'], alpha=OPT_ALPHA)
            self.train.brain.model_.opt = RMSprop(model.parameters(), lr=config['lr'], alpha=OPT_ALPHA)
        if config['optimizer'] == 'SGD':
            self.train.brain.model.opt = SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
            self.train.brain.model_.opt = SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
        if config['activation'] == 'relu':
            self.train.brain.model.activation = 'relu'
        if config['activation'] == 'sigmoid':
            self.train.brain.model.activation = 'sigmoid'

        print("Network architecture:\n" + str(model))
        self.train.run()
        del self.train
        global testing_num

        data_x = self.data_val[:, :-1].astype('float32')
        data_y = self.data_val[:, -1].astype('int32')

        data_batch_x = data_x
        data_batch_y = data_y
        brain = testing.Brain(self.pool, self.layers)
        brain._load()
        costs = pd.Series(np.ones(FEATURE_DIM))
        env = testing.PerfEnv(data_batch_x, data_batch_y, costs, FEATURE_FACTOR)  # testset
        agent = testing.PerfAgent(env, brain)
        _r, _len, _corr, _lens, selct, corr, prediction = agent.run()

        prediction = np.array(prediction)
        acc = accuracy_score(data_y, prediction)
        precison = precision_score(data_y, prediction)  # average='binary'
        recall = recall_score(data_y, prediction)
        testing_num += 1

        print("Acc:", acc)
        print("Recall:", precison)
        print("Precison:", recall)
        return ({
            'loss': 1 - acc,  # remember: HpBandSter always minimizes!
            'info': {
                'valAccuracy': acc,
                'valPrecision': precison,
                'valRecall': recall,
            }

        })

    @staticmethod
    def get_configspace():
        """
        It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
        :return: ConfigurationsSpace-Object
        """
        cs = CS.ConfigurationSpace()

        lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, default_value='1e-2', log=True)

        # For demonstration purposes, we add different optimizers as categorical hyperparameters.
        # To show how to use conditional hyperparameters with ConfigSpace, we'll add the optimizers 'Adam' and 'SGD'.
        # SGD has a different parameter 'momentum'.
        optimizer = CSH.CategoricalHyperparameter('optimizer', ['Adam', 'SGD', 'RMSprop'])
        momentum = CSH.UniformFloatHyperparameter('momentum', lower=0.0, upper=0.99, default_value=0.9, log=False)
        activation = CSH.CategoricalHyperparameter('activation', ['relu', 'sigmoid'])
        layers = CSH.UniformIntegerHyperparameter('layers', lower=3, upper=8, default_value=3.0, log=False)
        cs.add_hyperparameters([lr, optimizer, momentum, activation, layers])

        # The hyperparameter sgd_momentum will be used,if the configuration
        # contains 'SGD' as optimizer.
        cond = CS.EqualsCondition(momentum, optimizer, 'SGD')
        cs.add_condition(cond)

        # num_conv_layers =  CSH.UniformIntegerHyperparameter('num_conv_layers', lower=1, upper=4, default_value=2)

        # num_filters_1 = CSH.UniformIntegerHyperparameter('num_filters_1', lower=16, upper=256, default_value=64, log=True)
        # num_filters_2 = CSH.UniformIntegerHyperparameter('num_filters_2', lower=16, upper=256, default_value=64, log=True)
        # num_filters_3 = CSH.UniformIntegerHyperparameter('num_filters_3', lower=16, upper=256, default_value=64, log=True)
        # num_filters_4 = CSH.UniformIntegerHyperparameter('num_filters_4', lower=16, upper=256, default_value=64, log=True)

        return cs


if __name__ == "__main__":
    worker = PyTorchWorker(run_id='0')
    cs = worker.get_configspace()
    config = cs.sample_configuration().get_dictionary()
    print(config)
    res = worker.compute(config=config, budget=4, working_directory='/results')
    print(res)
