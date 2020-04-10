import numpy as np
import pandas as pd
import torch
import random
import json
import utils.util as utils

from models.agent import Agent, Brain
from config.parameter import *
from models.env import Environment
from config.log import Log
from models.pool import Pool


class Train():
    def __init__(self, data, data_val):
        self.data = data

    def is_time(self, epoch, trigger):
        return (trigger > 0) and (epoch % trigger == 0)

    def run(self):
        epoch_start = 0
        self.brain.update_lr(epoch_start)
        self.agent.update_epsilon(epoch_start)

        print("Initializing pool..")
        for i in range(POOL_SIZE // AGENTS):
            utils.print_progress(i, POOL_SIZE // AGENTS)
            self.agent.step()
            self.pool.cuda()

            # SET VALUES
            if self.is_time(epoch, EPSILON_UPDATE_EPOCHS):
                self.agent.update_epsilon(epoch)

            if self.is_time(epoch, LR_SC_EPOCHS):
                self.brain.update_lr(epoch)

            # LOG
            if self.is_time(epoch, LOG_EPOCHS):
                print("Epoch: {}/{}".format(epoch, TRAINING_EPOCHS))
                self.log.log()
                self.log.print_speed()

            if self.is_time(epoch, LOG_PERF_EPOCHS):
                self.log.log_perf()

            # TRAIN
            self.brain.train()

            for i in range(EPOCH_STEPS):
                self.agent.step()


SEED = 111222
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

if __name__ == "__main__":
    train = Train(np.array(tr_data_shuffle), np.array(ts_data_shuffle))
    train.run()
    del train
