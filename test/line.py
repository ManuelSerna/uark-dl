# Input: two Cartesian points A and B (tuples)
# Return: coefficients a and b for line eq y=ax+b that goes through A and B
def line_2_pts(A, B):
	a = (B[1]-A[1])/(B[0]-A[0])
	b = A[1] - a * A[0]
	return a, b

X = (3, 2)
Y = (2, 5)

print(line_2_pts(X, Y))
