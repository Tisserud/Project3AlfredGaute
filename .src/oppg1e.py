
import numpy as np
import matplotlib.pyplot as plt

def sekantMetode(
        g, 
        z0 : float,
        z1 : float,
        tol : float,
) -> float:
    z = [z0,z1]
    i=1
    while abs(z[i]-z[i-1])>tol:
        z_new=(z[i-1]*g(z[i])-z[i]*g(z[i-1]))/(g(z[i])-g(z[i-1]))
        z.append(z_new)
        i+=1
    return z[-1]

def test_g(x):
    return x + np.sin(x) + np.cos(x)

root = sekantMetode(test_g,4,5,0.1)
print(root)
x=np.linspace(-5,5,100)
plt.figure("oppgave 1e")
plt.plot(x,test_g(x))
plt.axvline(root,0,1)
plt.grid()
plt.show()