import sys

def readParams(): 
	parameterFile = open(sys.argv[1], 'r')
	sequenceFile = open(sys.argv[2], 'r')
	temp = sequenceFile.read().split(">Sequence 2")
	seq1 = temp[0].split(">Sequence1\n")[0]
	seq2 = temp[1]
	for line in seq1:
		print line
		break
	margProb = {}
	transProb = {}
	emitProb = {}
	temp = parameterFile.read().split("# Transition Probabilities")
	marg = filter(None,temp[0].split('\n'))
	print marg
	temp = temp[1].split("# Emission Probabilities")
	trans = filter(None,temp[0].split('\n'))
	emit = filter(None,temp[1].split('\n'))
	for line in marg:
		if line[0] != "#":
			temp = line.split()
			margProb[temp[0]] = temp[1]
	k = 1
	for line in trans:
		if line[0] != "#":
			temp = line.split()
			transProb[k] = temp
			k+=1
	print emit
	for line in emit:
		if line[2].isdigit():
			temp = line.split()
			emitProb[temp[0]] = (temp[1], temp[2])	


	parameters = [margProb, transProb, emitProb]
	observations = set(['I','D'])
	hidden_states = set(['1', '2', '3', '4'])

	return parameters, observations, hidden_states

#forward decoding
def forward(parameters, L, observations, hidden_states, ):
	f = {}
	marg = parameters[0]
	trans = parameters[1]
	emit = parameters[2]


readParams()	
