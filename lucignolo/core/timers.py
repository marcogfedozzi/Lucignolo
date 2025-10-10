
import numpy as np
from collections import deque
from mujoco import MjData

class cTimer:
		def __init__(self, sim_data: MjData, period: float, trigger_on_start=False):
			self.sim_data = sim_data
			self.period = period
			self.T = period
			self.time = -period if trigger_on_start else 0

		# Init the time the first time it is called
		def reset(self):
			self.time = self.sim_data.time

		def time_passed(self):
			return self.sim_data.time - self.time
		
		def time_passed_lim(self):
			return min(self.time_passed(), self.T)
		
		def time_passed_frac(self):
			return self.time_passed() / self.T
		
		def time_passed_frac_lim(self):
			return self.time_passed_lim() / self.T
		
			
		def time_left(self):
			return self.T - (self.sim_data.time - self.time)
		
		def time_left_frac(self):
			return self.time_left() / self.T
			
		def __call__(self) -> bool:
			return self.time_left() <= 0
		
class cRandomTimer(cTimer):
	def __init__(self, sim_data: MjData, period, pdf, trigger_on_start=False):
		self.sim_data = sim_data
		self.pdf = pdf

		super().__init__(sim_data, period, trigger_on_start=trigger_on_start)

	def reset(self):
		self.time = self.sim_data.time
		self.T = self.pdf()

class flagTimer(cTimer):
	def __init__(self, sim_data: MjData, period):
		super().__init__(sim_data, period)
		self.flag = False

	def __call__(self) -> bool:
		return self.flag and self.time_left() <= 0
	
	def on(self):
		self.flag = True
		self.reset()

	def off(self):
		self.flag = False
	

class GeneratorTimer:
	def __call__(self, sim_data: MjData, period=None, dist_args=None, trigger_on_start=False):

		assert period is not None or dist_args is not None, "At least one between period and prob must be set."

		if dist_args is None or dist_args == {}:
			return cTimer(sim_data, period, trigger_on_start=trigger_on_start)
		
		rng = np.random.default_rng()
	
		if 'lambda' in dist_args:
			pdf = lambda : rng.exponential(scale=dist_args['lambda'])
		
		else:
			assert period is not None, "a period is required"
		
			if 'a' in dist_args and 'b' in dist_args:
				pdf = lambda: period*rng.beta(a=dist_args['a'], b=dist_args['b'])

			elif 'a' in dist_args:
				pdf = lambda: period*rng.power(a=dist_args['a'])
			
			else:
				raise AttributeError(f"unknown distribution with parameters {dist_args}")
		
		return cRandomTimer(sim_data=sim_data, period=period, pdf=pdf, trigger_on_start=trigger_on_start)
	
class DeltaTimer:
	def __init__(self, sim_data: MjData, maxlen=100):
		self.sim_data = sim_data
		self.dt_win = deque(maxlen=maxlen)
		self.trigger_time = 0
	
	def reset(self):
		self.trigger_time = self.sim_data.time
	
	def step(self):
		self.dt_win.append(self.sim_data.time - self.trigger_time)
		self.trigger_time = self.sim_data.time
	
	@property
	def dt(self):
		return np.mean(self.dt_win)