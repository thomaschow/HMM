import sys, math

#This method reads in the initial parameters of ps5 and returns 4 arguments: parameters, observed_space, hidden_space, emissions.
#parameters[0] = marginal probabilities, parameters[1] = transition probabilities, parameters[2] = emission probabilities

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

parameters, observation_space, hidden_space, emissions = readParams()

logf, llf = forward(parameters, observation_space, hidden_space, emissions)
logb, llb = backward(parameters, observation_space, hidden_space, emissions)

print llf
print llb
