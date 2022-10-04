# Jerod Junkins - Fourier
# Collaborators: Brendan Schwartz

import math
import numpy as np


# f is a function which accepts a float and returns a float.
# a and b are floats representing the endpoints of the large interval.
# num_subintervals represents the number of subintervals for Simpson's Rule,
# assumed to be even.

def simpsons_rule(f, a, b, num_subintervals):
    if num_subintervals % 2 != 0:
        raise ValueError("Number of subintervals must be even!")
        
    delta_x = (b-a) / num_subintervals
    
    total = f(b) + f(a)
    
    for i in range(1, num_subintervals):
        if i % 2 == 0:
            coef = 2
        else:
            coef = 4
            
        total += coef * f(a + i * delta_x)
        
    return total * delta_x / 3



# f is a function which accepts a float and returns a float.
# N is an int representing the order of the Fourier expansion. 
# num_subintervals represents the number of sub_intervals to use
# the Simpson's Rule integration to calculte the Fourier coefficients.

def calculate_fourier_coefs(f, N, num_subintervals = 10000):
    a0 = simpsons_rule(f, -math.pi, math.pi, num_subintervals)/math.pi
    
    a_values = []
    b_values = []
    
    for i in range(1, N+1):
        a_values.append(simpsons_rule(lambda x: f(x)*math.cos(i*x), -math.pi, math.pi, num_subintervals) / math.pi)
        b_values.append(simpsons_rule(lambda x: f(x)*math.sin(i*x), -math.pi, math.pi, num_subintervals) / math.pi)
        
    return a0, tuple(a_values), tuple(b_values)


# fourier_coefs is a tuple with three entries, of the form specified above
# that can be returned by calculate_fourier_coefs
# x is a 1-D numpy array consisting of x values at which the Fourier
# approximation is to be evaluated.

def calculate_fourier_approx(fourier_coefs, x):
    y_values = np.full(x.shape, fourier_coefs[0]/2.)
    
    for i in range(len(fourier_coefs[1])):
        y_values += np.cos((i+1)*x) * fourier_coefs[1][i]
        y_values += np.sin((i+1)*x) * fourier_coefs[2][i]
    
    return y_values