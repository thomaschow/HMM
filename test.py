
def run_SS(iterations):
	a=1
	b=2
	for i in range(iterations):
		s = sum_squares(a,b)
		print s
		b = s
	return s

def sum_squares(a, b):
	return a*a + b*b

print run_SS(2)