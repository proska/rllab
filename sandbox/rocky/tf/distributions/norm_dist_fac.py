import tensorflow as tf
import numpy as np
from tensorflow.contrib.distributions import Normal

class NormDistFactory():
	def __init__(self, dim):
		self._dist_info_keys = ["mean", "log_std"]
		self._dim = dim

	def create_dist_by_distinfo(self, dist_info_vars):
		std = tf.exp(dist_info_vars["log_std"])
		new_dist = Normal(loc=dist_info_vars["mean"], scale=std)
		return new_dist

	def create_empty_dist(self, recurrent=False):
		is_recurrent = int(recurrent)
		mean = tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + [self._dim], name='mean')
		log_std = tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + [self._dim], name='log_std')
		std = tf.exp(log_std)
		new_dist = Normal(loc=mean, scale=std)
		return [mean, log_std], new_dist

	@property
	def dist_info_keys(self):
		return self._dist_info_keys

