from __future__ import division
import numpy as np
import os
from multiprocessing import Process,Array
import pandas as pd
import time
#import matplotlib.pyplot as plt
NumberofDimensions=0
class AdaBoost:

	def __init__(self, training_set):
		self.training_set = training_set
		self.N = len(self.training_set)
		self.weights = np.ones(self.N)/self.N
		self.NumberofDimensions = len(self.training_set[0][0])
		self.min_errors = Array('d',[1]*self.NumberofDimensions,lock=False)
		self.polarities = Array('d',[-1]*self.NumberofDimensions,lock=False)
		
		self.lines = Array('d',[0]*self.NumberofDimensions,lock=False)
		self.RULES = []
		self.ALPHA = []
		self.processes = []
	def linearSweep(self,dimension):
		l = [x[0][dimension] for x in (self.training_set)]
                l.sort()
                for k in xrange(len(l)-1):
                	bound=l[k]+(l[k+1]-l[k])/2
                        #print k,bound
                        errors_left=((np.array([t[1]!=(2*(t[0][dimension]>=bound)-1) for t in self.training_set]))*self.weights).sum();
                        if(self.min_errors[dimension] > errors_left):
                                self.polarities[dimension] = 0
                                self.lines[dimension] = bound
                                self.min_errors[dimension] = errors_left
                        errors_right=((np.array([t[1]!=(2*(t[0][dimension]<bound)-1) for t in self.training_set]))*self.weights).sum();
                        if(self.min_errors[dimension] > errors_right):
                                self.polarities[dimension] = 1
                                self.lines[dimension] = bound
                                self.min_errors[dimension] = errors_right
		return	
	def fit(self,rounds=50):
		for i in xrange(rounds):
			#print i
			min_error=1;
			min_index=-1;
			polarity=-1;
			line=0;
			dimensionProcesses=[0]*self.NumberofDimensions
			for j in xrange(self.NumberofDimensions):#parallelize starting from here
				dimensionProcesses[j] = Process(target=self.linearSweep,args=(j,))
                        	self.processes.append(dimensionProcesses[j])
                        	dimensionProcesses[j].start()
			for j in xrange(self.NumberofDimensions):
				dimensionProcesses[j].join()
			min_index = list(self.min_errors).index(min(self.min_errors))
			polarity = self.polarities[min_index]
			line = self.lines[min_index]
			#print min_index,polarity,line
			print self.polarities
			self.set_rule(min_index,polarity,line)	
					
	def func(self,x,min_index,polarity,bound):
		if(polarity):
			return 2*(x[min_index]<bound)-1
		else:
			return 2*(x[min_index]>=bound)-1

	def set_rule(self, min_index,polarity,bound,test=False):
		errors = np.array([t[1]!=self.func(t[0],min_index,polarity,bound) for t in self.training_set])
		e = (errors*self.weights).sum()
		#print errors
		if test: return e
		alpha = 0.5*np.log((1-e)/e)
		#print 'e=%.2f a=%.2f'%(e, alpha)
		w = np.zeros(self.N)
		for i in range(self.N):
			if errors[i] == 1: w[i] = self.weights[i] * np.exp(alpha)
			else: w[i] = self.weights[i] * np.exp(-alpha)
		self.weights = w / w.sum()
		self.RULES.append((min_index,polarity,bound))
		self.ALPHA.append(alpha)
	#def fit(self,rounds=10):
		

	def evaluate(self):
		NR = len(self.RULES)
		count=0.0
		for (x,l) in self.training_set:
			hx = [self.ALPHA[i]*self.func(x,self.RULES[i][0],self.RULES[i][1],self.RULES[i][2]) for i in range(NR)]
			if np.sign(l)==np.sign(sum(hx)):
				count=count+1
			#print np.sign(l) == np.sign(sum(hx))
		print "Accuracy:",count/len(self.training_set)

if __name__ == '__main__':
	examples = []
	df=pd.read_csv("data.csv")
	na=df.values
	trainattr=na[0:100,0:100].tolist()
	trainlabel=na[0:100,1000:1001].tolist()
	trainattr=trainattr+na[500:600,0:100].tolist()
	trainlabel=trainlabel+na[500:600,1000:1001].tolist()
	examples=[[trainattr[i],trainlabel[i]] for i in range(200)]
	m = AdaBoost(examples)
	
	m = AdaBoost(examples)
	time_start=time.time()
	m.fit(rounds=3)
	print 'Time:',time.time()-time_start
	'''
	m.set_rule(lambda x: 2*(x[0] < 1.5)-1)
	m.set_rule(lambda x: 2*(x[0] < 4.5)-1)
	m.set_rule(lambda x: 2*(x[1] > 5)-1)
	'''
	#m.fit(rounds=5)
	m.evaluate()
	'''
	for i in range(len(examples)):
		if(examples[i][1]==1):
			plt.plot(examples[i][0][0],examples[i][0][1],'ro')
		elif(examples[i][1]==-1):
			plt.plot(examples[i][0][0],examples[i][0][1],'go')
	plt.show()
	'''
	
