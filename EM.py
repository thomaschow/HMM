import math, sys

"""
X is observed data
Q is unobserved (latents) states
theta is parameters
"""

L = 100000
hidden_states = {1,2,3,4}
length = len(hidden_states)

"""
This method reads in the initial parameters of ps5 and returns 4 arguments: parameters, observed_space, hidden_space, emissions.
parameters[0] = marginal probabilities, parameters[1] = transition probabilities, parameters[2] = emission probabilities
"""

def readParams():  
	parameterFile = open(sys.argv[1], 'r')
	sequenceFile = open(sys.argv[2], 'r')
	temp = sequenceFile.read().split(">Sequence 2")
	seq1 = "".join(temp[0].replace(">Sequence 1\n", "").split("\n"))
	seq2 = "".join(temp[1].replace(">Sequence 2\n", "").split("\n"))
	emissions = []
	for i in range(len(seq1)):
		if seq1[i] == seq2[i]:
			emissions.append('I')
		else:
			emissions.append('D')	
	margProb = {}
	transProb = {}
	emitProb = {}
	temp = parameterFile.read().split("# Transition Probabilities")
	marg = filter(None,temp[0].split('\n'))
	temp = temp[1].split("# Emission Probabilities")
	trans = filter(None,temp[0].split('\n'))
	emit = filter(None,temp[1].split('\n'))
	for line in marg:
		if line[0] != "#":
			temp = line.split()
			margProb[int(temp[0])] = float(temp[1])
	k = 1
	for line in trans:
		if line[0] != "#":
			temp = line.split()
			transProb[k] = {}
			for j in range(len(temp)):	
				transProb[k][j+1] = float(temp[j])
			k+=1
	for line in emit:
		if line[2].isdigit():
			temp = line.split()
			emitProb[int(temp[0])] = [float(temp[1]), float(temp[2])]	

	parameters = [margProb, transProb, emitProb]
	observation_space = ['I','D']
	hidden_space = [1, 2, 3, 4]

	return parameters, observation_space , hidden_space, emissions 


#forward decoding
def forward(parameters, observation_space, hidden_space, emissions):
	f = []
	logf = []	
	marg = parameters[0]
	trans = parameters[1]
	emit = parameters[2]
	f.append({})
	logf.append({})
	for state in hidden_space:
		f[0][state] = (emit[state][observation_space.index(emissions[0])] * marg[state])
		logf[0][state] =  math.log(f[0][state]) 
	
	summation = 0
	logsummation = []
	
	for t in xrange(1, len(emissions)):
		f.append({})
		logf.append({})
		for j in hidden_space:
			logsummation = []
			for i in hidden_space:
				logsummation.append(logf[t-1][i] + math.log(trans[i][j]))
			m = max(logsummation)
			for a in xrange(len(logsummation)):
				logsummation[a] = math.exp(logsummation[a] - m)
			logf[t][j] = math.log(emit[j][observation_space.index(emissions[t])])  + m + math.log(sum(logsummation)) 
	fList = []
	llf = 0
	for k in hidden_space:
		fList.append(logf[len(emissions)-1][k])
	maxF = max(fList)
	for item in fList:
		llf = llf + math.exp(item - maxF)
	llf = maxF + math.log(llf)
	#print len(f)
	return logf, llf

def backward(parameters, observation_space, hidden_space, emissions):
	
	b = [0 for x in xrange(len(emissions))] 	
	logb = [0 for x in xrange(len(emissions))] 	
	marg = parameters[0]
	trans = parameters[1]
	emit = parameters[2]
	
	b[len(emissions) - 1] = {} 
	logb[len(emissions) - 1] = {} 

	for state in hidden_space:
		b[len(emissions) - 1][state] = 1
		logb[len(emissions) - 1][state] = 0
	logsummation = []
	for t in xrange(len(emissions) - 2, -1, -1):
		b[t] = {}
		logb[t] = {}
		for i in hidden_space:
			logsummation = []
			for j in hidden_space:
				logsummation.append(logb[t+1][j] + math.log(trans[i][j]) + math.log(emit[j][observation_space.index(emissions[t+1])])) 	 
			m = max(logsummation)
			for a in xrange(len(logsummation)):
				logsummation[a] = math.exp(logsummation[a] - m)			
			logb[t][i] = m + math.log(sum(logsummation))
	"""print len(b)
	for i in range( 10):
		print b[i] """
	bList = []
	llb = 0	
	for k in hidden_space:
		bList.append(math.log(emit[k][observation_space.index(emissions[0])]) + math.log(marg[k]) + logb[0][k])
	maxB = max(bList)
	for a in xrange(len(bList)):
		llb = llb + math.exp(bList[a] - maxB)
	llb = maxB + math.log(llb)
	return logb, llb

def EM(X, x, Q, old_params, old_log_lk):

	post_params = e_step(old_params, X, x, Q)
	new_params = m_step(post_params, X, x, Q)
	new_log_lk = compute_likelihood(new_params, post_params, old_log_lk)

	return new_params, post_params, new_log_lk

def e_step(params, X, x, Q):

	marg, trans, emit = params
	
	for k in Q:
		marg[k] = math.exp(marg[k])
	for i in Q:
		for j in Q:
			trans[i][j] = math.exp(trans[i][j])
	for k in Q:
		for i in xrange(2):
			emit[k][i] = math.exp(emit[k][i])
	params = [marg, trans, emit]
	f_log, f_log_lk = forward(params, X, Q, x)
	b_log, b_log_lk = backward(params, X, Q, x)
	a = params[1]
	e = params[2]

	"""stationary"""
	Pi_k = {}
	for k in range(1, 1 + length):
		# Pi_k[k] = f[0][k] * b[0][k] / likelihood
		Pi_k[k] = f_log[0][k] + b_log[0][k] - f_log_lk

	"""transition"""
	A_ij = {}
	for j in range(1, 1 + length):
		A_ij[j] = {}
	for t in range(L-1):
		for i in range(1, 1 + length):
			for j in range(1, 1 + length):
				# A_ij[i][j] = f[t][i] * b[t+1][j] * a[i][j] * e[j][t+1] / likelihood
				A_ij[i][j] = f_log[t][i] + b_log[t+1][j] + a[i][j] + e[j][X.index(x[t+1])] - f_log_lk
	"""emission"""
	E_k = {}
	for j in range(1, 1 + length):
		E_k[j] = [0.0, 0.0]
	for k in range(1, 1 + length):
		for t in range(L-1):
			if x[t] == 'I':
				# E_k[k][0] = f[t][k] * b[t][k] / f_log_lk
				E_k[k][0] = f_log[t][k] + b_log[t][k] - f_log_lk

			elif x[t] == 'D':
				# E_k[k][1] = f[t][k] * b[t][k] / f_log_lk
				E_k[k][1] = f_log[t][k] + b_log[t][k] - f_log_lk

	posterior_params = (Pi_k, A_ij, E_k)
	return posterior_params

def m_step(params, X, x, Q):
	pi_k_ml = {}

	a_ij_ml = {}
	for j in range(1, 1 + length):
		a_ij_ml[j] = {}

	e_k_ml = {}
	for j in range(1, 1 + length):
		e_k_ml[j] = [0.0, 0.0]

	Pi_k, A_ij, E_k = params[0], params[1], params[2]

	"""
	HERE
	"""
	p_sum = []
	e_sum = []
	a_sum = [] 
	p_max = 0
	e_max = 0
	a_max = 0
	for k in range(1, 1 + length):
		p_sum = []
		for j in xrange(1,1+length):
			p_sum.append(Pi_k[j])
		p_max = max(p_sum)
		for j in xrange(1, 1+length):
			p_sum[j-1] = math.exp(p_sum[j-1] - p_max)
		print p_sum
		pi_k_ml[k] = Pi_k[k] - (p_max + math.log(sum(p_sum)))
		
		
	for x in xrange(1,1+length):
		a_sum = []
		for y in xrange(1,1+length):
			a_sum.append(A_ij[x][y])
		a_max = max(a_sum)
		for y in xrange(1, 1+length):
			a_sum[y-1] = math.exp(a_sum[y-1] - a_max)
		for y in xrange(1, 1+length):
			a_ij_ml[x][y] = A_ij[x][y] - (a_max + sum(a_sum))	
			
	for x in xrange(1, 1 + length):
		e_sum = []
		for y in xrange(2):
			e_sum.append(E_k[x][y])
		e_max = max(e_sum)
		for y in xrange(2):
			e_sum[y-1] = math.exp(e_sum[y-1] - e_max)
		for y in xrange(2):
			e_k_ml[x][y] = E_k[x][y] - (e_max + sum(e_sum))	
		
		"""e_k_ml[k][0] = E_k[k][0] - math.log (sum([E_k[k][sig] for sig in range(2)]))
		e_k_ml[k][1] = E_k[k][1] - math.log(sum([E_k[k][sig] for sig in range(2)]))
		a_ij_ml[i][j] = A_ij[i][j] - math.log(sum([A_ij[i][r] for r in range(1, 1 + length)]))"""

		
	max_likelihood_params = (pi_k_ml, a_ij_ml, e_k_ml)
	return max_likelihood_params

def compute_likelihood(ml_params, post_params, old_log_lk):
	
	Pi_k, A_ij, E_k = post_params[0], post_params[1], post_params[2]
	pi_k_ml, a_ij_ml, e_k_ml = ml_params[0], ml_params[1], ml_params[2]
	new_log_lk = 0.0
	for i in xrange(1, 1 + length):
		for j in xrange(1, 1 + length):
			new_log_lk += math.exp(A_ij[i][j]) * a_ij_ml[i][j] #Transition probabilities
		new_log_lk += math.exp(E_k[i][0]) * e_k_ml[i][0] #Emission probabilities
		new_log_lk += math.exp(E_k[i][1]) * e_k_ml[i][1] #Emission probabilities
		new_log_lk += math.exp(Pi_k[i]) * pi_k_ml[i] #Stationary probabilities
	old_log_lk = new_log_lk
	return new_log_lk


def main():
	parameters, observation_space, hidden_space, emissions = readParams()
	logf, llf = forward(parameters, observation_space, hidden_space, emissions)
	logb, llb = backward(parameters, observation_space, hidden_space, emissions)

	X = observation_space
	Q = hidden_space
	x = emissions
	old_params = parameters
	old_log_lk = llf

	for i in range(15):
		new_params, post_params, new_log_lk = EM(X, x, Q, old_params, old_log_lk)
		old_params = new_params
		old_log_lk = new_log_lk

		print ("ITERATION NUMBER: " + str(i))
		print "============================"
		print ""
		print "Parameters"
		print "----------"
		print "pi_k_ml..........a_ij_ml..........e_k_ml"
		print (str(new_params[0]) + ".........." + str(new_params[1]) + ".........." + str(new_params[2]))
		print "============================"
		print ""
		print "Log-Likelihood with Estimated Parameters"
		print "---------------------------------------"
		print str(new_log_lk)

if __name__ == '__main__':
	main()
