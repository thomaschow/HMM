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

	return parameters, observation_space, hidden_space, emissions 


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
	# print post_params[0]
	# print post_params[1]
	# print post_params[2]	
	new_params = m_step(post_params, X, x, Q)

	marg = new_params[0]
	trans = new_params[1]
	emit = new_params[2]


	# for k in Q:
	# 	marg[k] = math.exp(marg[k])
	# for i in Q:
	# 	for j in Q:
	# 		trans[i][j] = math.exp(trans[i][j])
	# for k in Q:
	# 	for i in xrange(2):
	# 		emit[k][i] = math.exp(emit[k][i])
	params = [marg, trans, emit]

	# print marg
	# print trans
	# print emit

	new_f_log, new_log_lk = forward(params, X, Q, x)
	return new_params, post_params, new_log_lk

def e_step(params, X, x, Q):

	marg, trans, emit = params
	# for k in Q:
	# 	marg[k] = math.exp(marg[k])
	# for i in Q:
	# 	for j in Q:
	# 		trans[i][j] = math.exp(trans[i][j])
	# for k in Q:
	# 	for i in xrange(2):
	# 		emit[k][i] = math.exp(emit[k][i])
	params = [marg, trans, emit]
	f_log, f_log_lk = forward(params, X, Q, x)
	b_log, b_log_lk = backward(params, X, Q, x)
	# print f_log_lk
	# print b_log_lk
	# print "HERE"
	a = params[1]
	e = params[2]

	"""stationary"""
	Pi_k = {}
	exp_Pi_k = {}
	for k in range(1, 1 + length):
		# Pi_k[k] = f[0][k] * b[0][k] / likelihood
		Pi_k[k] = f_log[0][k] + b_log[0][k] - f_log_lk
		Pi_k[k] = math.exp(Pi_k[k])
		# print f_log_lk
		# print Pi_k[k]

	"""transition"""
	A_ij = {}
	for j in range(1, 1 + length):
		A_ij[j] = {}
	for i in range(1, 1 + length):
		for j in range(1, 1 + length):
			A_ij_sum = []
			for t in range(L-1):
				# A_ij[i][j] = f[t][i] * b[t+1][j] * a[i][j] * e[j][t+1] / likelihood
				A_ij[i][j] = f_log[t][i] + b_log[t+1][j] + math.log(a[i][j]) + math.log(e[j][X.index(x[t+1])])
				A_ij_sum.append(A_ij[i][j])
			max_transition = max(A_ij_sum)
			trans_sum = 0.0
			for r in range(len(A_ij_sum)):
				trans_sum += math.exp(A_ij_sum[r] - max_transition)
			log_trans_sum = math.log(trans_sum) + max_transition
			log_diff_transition = log_trans_sum - f_log_lk
			A_ij[i][j] = math.exp(log_diff_transition)

	"""emission"""
	E_k = {}
	observed = ['I', 'D']
	for j in range(1, 1 + length):
		E_k[j] = [0.0, 0.0]
	for k in range(1, 1 + length):
		for obs in observed:
			E_k_sum = []
			for t in range(L):
				if obs == 'I' and x[t] == 'I':
				# E_k[k][0] = f[t][k] * b[t][k] / f_log_lk
					E_k[k][0] = f_log[t][k] + b_log[t][k]
					E_k_sum.append(E_k[k][0])
				# E_k[k][0] += f_log[t][k] + b_log[t][k] - f_log_lk
				elif obs == 'D' and x[t] == 'D':
					# E_k[k][1] = f[t][k] * b[t][k] / f_log_lk
					E_k[k][1] = f_log[t][k] + b_log[t][k]
					E_k_sum.append(E_k[k][1])
					# E_k[k][1] += f_log[t][k] + b_log[t][k] - f_log_lk
			max_emission = max(E_k_sum)
			emis_sum = 0.0
			for r in range(len(E_k_sum)):
				emis_sum  += math.exp(E_k_sum[r] - max_emission)
			log_emis_sum  = math.log(emis_sum) + max_emission
			log_diff_emission = log_emis_sum - f_log_lk
			if obs == 'I':
				E_k[k][0] = math.exp(log_diff_emission)
			elif obs == 'D':
				E_k[k][1] = math.exp(log_diff_emission)
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

	pi_k_ml = Pi_k
	for i in range(1, 1+length):
		for j in range(1, 1+length):
			a_ij_ml[i][j] = A_ij[i][j] / sum([A_ij[i][r] for r in range(1, 1 + length)])
	for k in range(1, 1+length):
		e_k_ml[k][0] = E_k[k][0] / sum([E_k[k][sig] for sig in range(2)])
		e_k_ml[k][1] = E_k[k][1] / sum([E_k[k][sig] for sig in range(2)])

	# p_sum = []
	# e_sum = []
	# a_sum = [] 
	# p_max = 0
	# e_max = 0
	# a_max = 0
	# for k in range(1, 1 + length):
	# 	p_sum = []
	# 	for j in xrange(1,1+length):
	# 		p_sum.append(Pi_k[j])
	# 	p_max = max(p_sum)
	# 	for j in xrange(1, 1+length):
	# 		p_sum[j-1] = math.exp(p_sum[j-1] - p_max)
	# 	# print "p_sum: " + str(p_sum)
	# 	pi_k_ml[k] = Pi_k[k] - (p_max + math.log(sum(p_sum)))
	# print ""
		
		
	# for x in xrange(1,1+length):
	# 	a_sum = []
	# 	for y in xrange(1,1+length):
	# 		a_sum.append(A_ij[x][y])
	# 	a_max = max(a_sum)
	# 	for y in xrange(1, 1+length):
	# 		a_sum[y-1] = math.exp(a_sum[y-1] - a_max)
	# 	for y in xrange(1, 1+length):
	# 		a_ij_ml[x][y] = A_ij[x][y] - (a_max + sum(a_sum))	
			
	# for x in xrange(1, 1 + length):
	# 	e_sum = []
	# 	for y in xrange(2):
	# 		e_sum.append(E_k[x][y])
	# 	e_max = max(e_sum)
	# 	for y in xrange(2):
	# 		e_sum[y-1] = math.exp(e_sum[y-1] - e_max)
	# 	for y in xrange(2):
	# 		e_k_ml[x][y] = E_k[x][y] - (e_max + sum(e_sum))
		
	max_likelihood_params = (pi_k_ml, a_ij_ml, e_k_ml)
	return max_likelihood_params
def posterior_decoding (params, X, Q, x):
	logf, llf = forward(params, X, Q, x)
	logb, llb = backward(params, X, Q, x)
	S = [.32, 1.75, 4.54, 9.40]
	post_candidates = []
	post_decode = []
	post_all = []
	for t in xrange(len(x)):
		post_candidates = []
		for k in Q:
			post_candidates.append(math.exp(logf[t][k] + logb[t][k] - llf))
		post_decode.append(S[post_candidates.index(max(post_candidates))])
		post_all.append(post_candidates)
	return post_decode, post_all

def viterbi( params, X, Q, x):
	marg = params[0]
	trans = params[1]
	emit = params[2]
	v = []
	S = {1:.32, 2:1.75, 3:4.54, 4:9.40}
	v.append ({})
	viterbi = []
	for k in Q:
		v[0][k] = math.log(emit[k][X.index(x[0])]) + math.log( marg[k])
	for i in xrange(len(x)):
		viterbi.append(0)

	v_max = []
	ptr = [{}]
	for t in xrange(1, len(x)):
		v.append({})
		ptr.append({})
		for j in Q:
			v_max = [v[t-1][i] + math.log(trans[i][j]) for i in Q] 
			v[t][j] = math.log(emit[j][X.index(x[t])])  + max(v_max)
			ptr[t][j] = v_max.index(max(v_max)) + 1
	m = max(v[len(x)-1], key = v[len(x)-1].get)
	viterbi[len(x) -1] = m
	for t in xrange(len(x) - 2, -1, -1):
		viterbi[t] = ptr[t + 1][viterbi[t+1]]
	for t in xrange(len(x)-1):
		viterbi[t] = S[viterbi[t]]
	viterbi[len(x)-1] = S[m]
	return viterbi

def posterior_mean (params, X, Q, x):
	post_decode, post_all = posterior_decoding(params, X, Q, x)
	post_mean = []
	S = [.32, 1.75, 4.54, 9.40]
	for t in xrange(len(x)):
		post_mean.append( sum(S[i] * post_all[t][i] for i in xrange(len(S))))	
	return post_mean, post_decode
# def compute_likelihood(ml_params, post_params, old_log_lk):
	
# 	Pi_k, A_ij, E_k = post_params[0], post_params[1], post_params[2]
# 	pi_k_ml, a_ij_ml, e_k_ml = ml_params[0], ml_params[1], ml_params[2]
# 	new_log_lk = 0.0
# 	# for i in xrange(1, 1 + length):
# 	# 	for j in xrange(1, 1 + length):
# 	# 	# 	new_log_lk += math.exp(A_ij[i][j]) * a_ij_ml[i][j] #Transition probabilities
# 	# 	# new_log_lk += math.exp(E_k[i][0]) * e_k_ml[i][0] #Emission probabilities
# 	# 	# new_log_lk += math.exp(E_k[i][1]) * e_k_ml[i][1] #Emission probabilities
# 	# 	# new_log_lk += math.exp(Pi_k[i]) * pi_k_ml[i] #Stationary probabilities

# 	# 		new_log_lk += A_ij[i][j] * math.log(a_ij_ml[i][j]) #Transition probabilities
# 	# 	new_log_lk += E_k[i][0] * math.log(e_k_ml[i][0]) #Emission probabilities
# 	# 	new_log_lk += E_k[i][1] * math.log(e_k_ml[i][1]) #Emission probabilities
# 	# 	new_log_lk += Pi_k[i] * math.log(pi_k_ml[i]) #Stationary probabilities

# 	f_log, new_log_llk = forward(params, X, Q, x

# 	old_log_lk = new_log_lk
# 	return new_log_lk


def main():
	parameters, observation_space, hidden_space, emissions = readParams()
	logf, llf = forward(parameters, observation_space, hidden_space, emissions)
	logb, llb = backward(parameters, observation_space, hidden_space, emissions)

	X = observation_space
	Q = hidden_space
	x = emissions
	old_params = parameters
	old_log_lk = llf

	post_mean, post_decode = posterior_mean(old_params, X, Q, x)
	v = viterbi(old_params, X, Q, x)
	for i in xrange(len(post_decode)):
		#decodings_initial_output_file.write( str(v[i]) + "\t" + str(post_decode[i]) + "\t" + str(post_mean[i]) + "\n")
		print(str(v[i]) + "\t" + str(post_decode[i]) + "\t" + str(post_mean[i])) 
	for i in range(15):
		print ("ITERATION NUMBER: " + str(i+1))
		print "============================"
		print ""
		new_params, post_params, new_log_lk = EM(X, x, Q, old_params, old_log_lk)
		old_params = new_params
		if (abs(old_log_lk - new_log_lk) < 1):
			print "CONVERGED!"
			print ""
			count = 1
		old_log_lk = new_log_lk


		print "Parameters"
		print "----------"
		print "pi_k_ml: " + str(new_params[0])
		print ""
		print "a_ij_ml: " + str(new_params[1])
		print ""
		print "e_k_ml: "  + str(new_params[2])
		print ""
		print ""
		print "Log-Likelihood with Estimated Parameters"
		print "---------------------------------------"
		print str(new_log_lk)
		print ""
		print ""
	
	print "Log-Likelihood with Initial Parameters"
	print "---------------------------------------"
	print str(llf)

	parameterFileString = sys.argv[1]
	sequence_type = parameterFileString[-7:]
	sequence_type_pdf = parameterFileString[-7:-4] + ".pdf"

	#-------------Parameters-----------------------

	parameter_output_filename = "outputs/estimated_parameters" + sequence_type
	parameter_output_file = open(parameter_output_filename, 'w')
	parameter_output_file.write("pi_k_ml\n")
	parameter_output_file.write(str(new_params[0]))
	parameter_output_file.write("a_ij_ml\n")
	parameter_output_file.write(str(new_params[1]))
	parameter_output_file.write("e_k_ml\n")
	parameter_output_file.write(str(new_params[2]))

	likelihood_output_filename = "outputs/likelihoods" + sequence_type
	likelihood_output_file = open(likelihood_output_filename, 'w')
	likelihood_output_file.write("Log-Likelihood with Initial Parameters\n")
	likelihood_output_file.write(str(llf) + '\n')
	likelihood_output_file.write("Log-Likelihood with Estimated Parameters\n")
	likelihood_output_file.write(str(new_log_lk) + '\n')

	#-----------Decodings initial------------------
	decodings_initial_output_filename = "outputs/decodings_initial" + sequence_type
	decodings_initial_output_file = open(decodings_initial_output_filename, 'w')
	decodings_initial_output_file.write("Viterbi Decoding | Posterior Decoding | Posterior Mean\n")
		
	#CALCULATE THIS SHIT"
	post_mean, post_decode = posterior_mean(old_params, X, Q, x)
	v = viterbi(old_params, X, Q, x)
	for i in xrange(len(post_decode)):
		decodings_initial_output_file.write( str(v[i]) + "\t" + str(post_decode[i]) + "\t" + str(post_mean[i]) + "\n")

	plot_initial_output_filename = "outputs/plot_initial" + sequence_type_pdf
	plot_initial_output_file = open(plot_initial_output_filename, 'w')
	#PLOT THIS SHIT"

	#-----------Decodings estimated----------------
	decodings_estimated_output_filename = "outputs/decodings_estimated" + sequence_type
	decodings_estimated_output_file = open(decodings_estimated_output_filename, 'w')
	decodings_estimated_output_file.write("Viterbi Decoding | Posterior Decoding | Posterior Mean\n")
	#CALCULATE THIS SHIT"

	plot_estimated_output_filename = "outputs/plot_estimated" + sequence_type_pdf
	plot_estimated_output_file = open(plot_estimated_output_filename, 'w')
	#PLOT THIS SHIT"

if __name__ == '__main__':
	main()
