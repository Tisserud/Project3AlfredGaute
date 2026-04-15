import numpy as np
import matplotlib.pyplot as plt

from test import RungeKutta45, diff_func_1a

def BVP_solver(
        f,
        x_init : float,
        x_end : float,
        y_boundaries : tuple,
        b0 : float,
        b1 : float,
        tol : float = 0.001,
) -> tuple:
    y_iterations=[]
    x_iterations=[]
    b=[b0,b1]
    x, y, n = RungeKutta45(x_init,x_end,np.array((y_boundaries[0],b0)),f)
    y_iterations.append(y)
    x_iterations.append(x)
    x, y, n = RungeKutta45(x_init,x_end,np.array((y_boundaries[0],b1)),f)
    y_iterations.append(y)
    x_iterations.append(x)

    i=1
    def err(i):
        return y_iterations[i][:,0][-1] - y_boundaries[1]
    
    while abs(err(i)) > tol:
        b_new=(b[i-1]*err(i)-b[i]*err(i-1))/(err(i)-err(i-1))
        print(b_new)
        b.append(b_new)
        # z_new=(z[i-1]*g(z[i])-z[i]*g(z[i-1]))/(g(z[i])-g(z[i-1]))
        x, y, n = RungeKutta45(x_init,x_end,np.array((y_boundaries[0],b_new)),f)
        y_iterations.append(y)
        x_iterations.append(x)
        i+=1

    
    
    return x_iterations, y_iterations


x_init = 0
x_end = 2*np.pi
y_boundaries=(0,0)
b0=100
b1=9000000


result = BVP_solver(diff_func_1a,x_init,x_end,y_boundaries,b0,b1)


plt.figure("BVP solution 1f")
plt.plot(result[0][-1],result[1][-1][:,0])
plt.show()