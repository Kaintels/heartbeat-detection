import numpy as np
import torch
from torch.autograd import Variable

from config.parameter import *
from models.model import Net

class Agent():
	def __init__(self, env, pool, brain):
		self.env = env
		self.pool = pool
		self.brain = brain
		self.epsilon = EPSILON_START
		self.s = self.env.reset()

	def store(self, x):
		self.pool.put(x)

	def act(self, s):
		m = np.zeros((AGENTS, ACTION_DIM))	      
		m[:, CLASSES:] = s[:, FEATURE_DIM:]  

		return a  

	def step(self):
		a = self.act(self.s)
		s_, r = self.env.step(a)
		self.store( (self.s, a, r, s_) )
		self.s = s_

	def update_epsilon(self, epoch):
		if epoch >= EPSILON_EPOCHS:
			self.epsilon = EPSILON_END
		else:
			self.epsilon = EPSILON_START + epoch * (EPSILON_END - EPSILON_START) / EPSILON_EPOCHS


class Brain:
	def __init__(self, pool,layers):
		self.pool = pool   
		self.layers = layers
		self.model = Net(layers=layers)
		self.model_ = Net(layers=layers)

		# print("Network architecture:\n"+str(self.model))

	def _save(self):

	def predict_pt(self, s, target):
		s = Variable(s)   #

		if target:    # false
			return self.model_(s).data
		else:
			return self.model(s).data

	def predict_np(self, s, target=False):
		s = torch.from_numpy(s).cuda()
		res = self.predict_pt(s, target)
		return res.cpu().numpy()

	def train(self):
		s, a, r, s_ = self.pool.sample(BATCH_SIZE)

		# extract the mask
		m_ = torch.FloatTensor(BATCH_SIZE, ACTION_DIM).zero_().cuda()
		m_[:, CLASSES:] = s_[:, FEATURE_DIM:]

		# compute
		q_current = self.predict_pt(s_, target=False) - (MAX_MASK_CONST * m_)    # masked actions do not influence the max
		q_target = self.predict_pt(s_, target=True)

		_, amax = q_current.max(1, keepdim=True) # max값이 들어있는 index 반환

		self.model.train_network(s, a, q_)
		self.model_.copy_weights(self.model)

	def update_lr(self, epoch):
		lr = OPT_LR * (LR_SC_FACTOR ** (epoch // LR_SC_EPOCHS))
		lr = max(lr, LR_SC_MIN)

		self.model.set_lr(lr)
		print("Setting LR:", lr)