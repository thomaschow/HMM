import sys

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
			emitProb[int(temp[0])] = (float(temp[1]), float(temp[2]))	

	parameters = [margProb, transProb, emitProb]
	observations = ['I','D']
	hidden_states = [1, 2, 3, 4]

	return parameters, observations, hidden_states, emissions 

#forward decoding
def forward(parameters, observed_states, hidden_states, emissions):
	f = []
	
	marg = parameters[0]
	trans = parameters[1]
	emit = parameters[2]
	f.append({})
	for state in hidden_states:
		f[0][state] = (emit[state][observations.index(emissions[0])] * marg[state])
	
	summation = 0
	for t in range(1, len(emissions)):
		f.append({})
		for j in hidden_states:
			for i in hidden_states:
				summation = summation + f[t-1][i] * trans[i][j]
			f[t][j] = (emit[j][observations.index(emissions[t])] * summation)
			summation = 0 	
	print len(f)
	for prob in f:
		print prob 

def backward(parameters, observation_space, hidden_states, emissions):
	
	b = [0 for x in xrange(len(emissions))] 	
	marg = parameters[0]
	trans = parameters[1]
	emit = parameters[2]
	
	b[len(emissions) - 1] = {} 
	
	for state in hidden_states:
		b[len(emissions) - 1][state] = 1
	summation = 0
	for t in xrange(len(emissions) - 2, 0, -1):
		b[t] = {}
		for i in hidden_states:
			for j in hidden_states:
				summation = summation + b[t+1][j] * trans[i][j]* emit[j][observation_space.index(emissions[t+1])]
			b[t][i] =  summation
			summation = 0 	
	print len(b)
	for i in range(10 ):
		print b[i] 
parameters, observations, hidden_states, emissions = readParams()

forward(parameters, observations, hidden_states, emissions)


