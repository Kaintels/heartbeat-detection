import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics  import precision_score,recall_score,accuracy_score
from config.parameter import *

#==============================

#DATA_FILE = "./data/BSW_test.csv"
#META_FILE = "../data/forest-meta"
MODEL_FILE = "../saved/model/model"  # 이상하게 직접경로로 해야함

#META_COSTS = 'cost'
#META_AVG   = 'avg'
#META_STD   = 'std'

MAX_MASK_CONST = 1.e6


#==============================
class PerfAgent():
	def __init__(self, env, brain):
		self.env  = env
		self.brain = brain

		self.agents = self.env.agents

		self.done = np.zeros(self.agents, dtype=np.bool)

		self.total_r = np.zeros(self.agents)
		self.total_corr  = np.zeros(self.agents, dtype=np.int32)
		self.total_len  = np.zeros(self.agents, dtype=np.int32)

		self.s = self.env.reset()

		self.selected = []

	def act(self, s):
		m = np.zeros((self.agents, ACTION_DIM))	# create max_mask
		m[:, CLASSES:] = s[:, FEATURE_DIM:]

		p = self.brain.predict(s) - MAX_MASK_CONST * m 	# select an action not considering those already performed
		a = np.argmax(p, axis=1)

		return a

	def step(self):
		a = self.act(self.s)
		self.selected.append(a)
		s_, r, done = self.env.step(a)
		self.s = s_

		newly_finished = ~self.done & done     # 방금 전 수행에서 분류를 수행해서 에피소드가 끝난 agent 체크
		self.done = self.done | done

		self.total_len = self.total_len + ~done	    # classification action not counted
		self.total_r   = self.total_r   + r * (newly_finished | ~done)
		self.total_corr = self.total_corr + (r == REWARD_CORRECT) * newly_finished

	def run(self):
		lens = []

		while not np.all(self.done):    # self.done이 모두 True가 될 때까지
			self.step()

		get_s = pd.DataFrame(self.selected[-1])

		avg_r    = np.sum(self.total_r)
		avg_len  = np.sum(self.total_len)
		avg_corr = np.sum(self.total_corr)

		lens.append(self.total_len)

		return avg_r, avg_len, avg_corr, lens, self.selected, self.total_corr, self.selected[-1]


class PerfEnv:
	def __init__(self, data_x, data_y, costs, ff):
		self.x = data_x
		self.y = data_y
		self.costs = costs.to_numpy()
		self.agents = len(data_x)    # 1000
		self.lin_array = np.arange(self.agents)    # 0~999 np.array

		self.ff = ff    # feature factor

	def reset(self):
		self.mask = np.zeros( (self.agents, FEATURE_DIM) )    # 1000, 11
		self.done = np.zeros( self.agents, dtype=np.bool )

		return self._get_state()

	def step(self, action):
		# print(np.shape(action))
		r = np.zeros((len(action),))
#		print(self.mask)
#		print(action)
        
		for i in np.where(action >= CLASSES)[0]:
			# print(i)
			self.mask[i, action[i] - CLASSES] = 1
        
		# self.mask[self.lin_array, action - CLASSES] = 1    # 0~999,
		# print(self.mask)
#		print(self.costs)
		for i in range(len(action)):
			if action[i] >= CLASSES:
				r[i] = -self.costs[action[i] - CLASSES] * self.ff    # 각 agent의 행동에 대한 보상
			else:
				pass

		for i in np.where(action < CLASSES)[0]:
			# print(i)
			r[i] = REWARD_CORRECT if (action[i] >= self.y[i]-3) and (action[i] <= self.y[i]+3) else REWARD_INCORRECT
#			r[i] = REWARD_CORRECT if action[i] == self.y[i] else REWARD_INCORRECT
			self.done[i] = 1

		s_ = self._get_state()

		return (s_, r, self.done)

	def _get_state(self):
		x_ = self.x * self.mask
		x_ = np.concatenate( (x_, self.mask), axis=1 ).astype(np.float32)
		return x_

class Brain:
	def __init__(self):
		self.model  = Net()
		print("Network architecture:\n"+str(self.model))

	def _load(self):
		self.model.load_state_dict( torch.load(MODEL_FILE, map_location={'cuda:0': 'cpu'}) )

	def predict(self, s):
		# s = Variable(torch.from_numpy(s).cuda())
		s = Variable(torch.from_numpy(s))
		res = self.model(s)
		# return res.data.cpu().numpy()
		return res.data.numpy()

print("R:", avg_r)
print("Correct:", avg_corr)
print("Length:", avg_len)

print("Acc:", acc )
print("Recall:", precison )
print("Precison:", recall )
# selected = np.array([a-1 for a in agent.selected[:-1]])
# print(selected)

lens = np.concatenate(lens, axis=1).flatten()

np.set_printoptions(suppress=True, threshold=1e8, precision=4)
