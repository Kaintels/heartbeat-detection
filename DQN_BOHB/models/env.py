import numpy as np

from config.parameter import *

class Environment:
	def __init__(self, data, costs, ff):
		self.data_x = data[:, :-1]
		#self.data_x = np.delete(self.data_x,-2,1)
		self.data_y = data[:,   -1].astype('int32')
		#self.data_y = self.data_y.astype('int32')
		self.data_len = len(data)
		self.costs = costs.values

		self.mask = np.zeros( (AGENTS, FEATURE_DIM) )    # 1000, 11
		self.x    = np.zeros( (AGENTS, FEATURE_DIM) )
		self.y    = np.zeros( AGENTS )

		self.ff = ff

	def reset(self):
		for i in range(AGENTS):
			self._reset(i)

		return self._get_state()

	def _reset(self, i):
		self.mask[i] = 0
		self.x[i], self.y[i] = self._generate_sample()

	def step(self, action):

		return (s_, r)

	def _generate_sample(self):

		return (x, y)

	def _get_state(self):
		x_ = self.x * self.mask
		x_ = np.concatenate( (x_, self.mask), axis=1 ).astype(np.float32)
		return x_
		