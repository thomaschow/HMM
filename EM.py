"""
X is observed data
Q is unobserved (latents) states
theta is parameters
"""


def runEM(data, iterations):
	
	likelihood = 0.0
	log_likelihood = 0.0
	old_params = None
	L = 1000000
	hidden_states = {1,2,3,4}
	length = len(hidden_states)

	"""
	Read data

	Initialize theta to arbitrary values using data (stationary, transition, emission)

	e_i_x = {"I": , "D": } #4 hidden states (rows) by 2 observed states (cols)
	a_ij = #4 hidden states by 4 hidden states
	pi_i = #4 hidden states
	theta_0 = (pi_i, a_ij, e_i_x)

	"""

	old_params = pi_i, a_ij, e_i_x
	for i in range(iterations):
		new_params, likelihood = EM(X, x, Q, old_params)
		old_params = new_params

def EM(X, x, Q, old_params):

	post_params = e_step(params, X, x, Q)
	new_params = m_step(post_params, X, x, Q)
	old_params = new_params

	return new_params, likelihood


	"""
	Determine log-likelihood and print
	"""

def e_step(params, X, x, Q):
	f = []
	b = []
	for l in range(L-1):
		for i in range(length):
			for j in range(length):
				f = forwards(params, X, Q, x)
				b = backwards(params, X, Q, x)
				likelihood = sum(f[L-1])

	"""stationary"""
	Pi_k = []
	for i in range(length):
		Pi_k.append(0.0)
	for k in range(length):
		Pi_k[k] = f[0][k] * b[0][k] / likelihood

	"""transition"""
	A_ij = []
	for i in range(length):
		A = []
		for j in range(length):
			A.append(0.0)
		A_ij.append(A)
	for t in range(L-1):
		for i in range(length):
			for j in range(length):
				A_ij[i][j] = f[t][i] * b[t+1][j] * a[i][j] * e[j][t+1] / likelihood

	"""emission"""
	E_k = []
	for i in range(length):
		E = []
		for j in range(len(X))):
			E.append(0.0)
		E_k.append(E)
	for k in range(length):
		for t in range(L-1):
			if x[t] == 'I':
				E_k[k][0] = f[t][k] * b[t][k] / likelihood
			elif x[t] == 'D':
				E_k[k][1] = f[t][k] * b[t][k] / likelihood

	posterior_params = (Pi_k, A_ij, E_k)
	return posterior_params

def m_step(params, X, x, Q):
	pi_k_ml = []
	for i in range(length):
		pi_k_ml.append(0.0)

	a_ij_ml = []
	for i in range(length):
		A = []
		for j in range(length):
			A.append(0.0)
		a_ij_ml.append(A)

	e_k_ml = []
	for i in range(length):
		E = []
		for j in range(len(X))):
			E.append(0.0)
		e_k_ml.append(E)

	Pi_k, A_ij, E_k = params[0], params[1], params[2]

	for k in range(length):
		pi_k_ml[k] = Pi_k[k] / sum(Pi_k[j] for j in range(length))
		e_k_ml[k][0] = E_k[k][0] / sum(E_k[k][sig] for sig in range(len(X)))
		e_k_ml[k][1] = E_k[k][1] / sum(E_k[k][sig] for sig in range(len(X)))
	a_ij_ml[i][j] = A_ij[i][j] / sum(A_ij[i][r] for r in range(length))

	max_likelihood_params = (pi_k_ml, a_ij_ml, e_k_ml)
	return max_likelihood_params

	"""for i in range(15):
		repeat

		while abs((theta_new - theta_old) > threshold):
			repeat
	"""


print runEM(data, 15)

