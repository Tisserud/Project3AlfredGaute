import numpy as np
import matplotlib.pyplot as plt

def RungeKutta45(
        x_init : float,
        x_end : float,
        y_init : np.ndarray[np.float64],
        f,
        h0 : float =0.1,
        tol : float = 0.001,
        alpha : float = 0.8
                 ) -> tuple:
    
    x  =  np.zeros(int(np.ceil((x_end-x_init)/h0)))
    y  =  np.zeros((int(np.ceil((x_end-x_init)/h0)),len(y_init)))
    h  =  h0
    x[0] = x_init
    y[0] = y_init
    z = 0

    n  =  0
    k1  =  f(x_init,y_init)

    while x_end-x[n]>0:
       

        # Gjør arrayet større i tilfellet at gjetningen på størreslen var for liten
        if len(x)==n+1:
            
            sizeInc=int(np.ceil((x_end-x[n])/h))
            x=np.concatenate((x,np.zeros(sizeInc)))
            y=np.concatenate((y,np.zeros((sizeInc,len(y_init)))))

        
        h = min(h,x_end-x[n])
        k2 = f(x[n]+h/2,y[n]+h*k1/2)
        k3 = f(x[n]+3/4*h,y[n]+3/4*h*k2)
        y[n+1] = (y[n]+h/9*(2*k1+3*k2+4*k3))
        x[n+1] = (x[n]+h)
        k4 = f(x[n]+h,y[n+1])
        z = y[n]+h/24*(7*k1+6*k2+8*k3+3*k4)
        err_est = np.linalg.norm(y[n+1]-z)
        if err_est<tol:
            n += 1
            k1 = k4
        h=alpha*h*(tol/err_est)**(1/3)
    
    # Fjerner nuller bakerst i arrayet i tilfellet array ble for stort
    x =  np.trim_zeros(x,trim="b")
    y = np.trim_zeros(y,trim="b")


    return x, y, n
    
    
def diff_func_1a(
        x : float,
        y : np.ndarray[np.float64]
    )  -> np.ndarray[np.float64]:

    out=np.zeros_like(y)

    out[0]=y[1]
    out[1]=-4*np.sin(2*x)

    return out

x_init = 0
x_end = 2*np.pi
y_init = np.array([0,2])
h0 = 0.1

x , y , n= RungeKutta45(x_init,x_end,y_init,diff_func_1a,h0,tol= 1e-7, alpha=0.8)


plt.figure("Oppgave 1c")
plt.plot(x,y[:,0])




#### oppgave 1d

def anal_solution(x):
    return np.sin(2*x)

tols= np.logspace(1,-8,100)
alphas= np.linspace(0.1,1,100)
error= np.zeros_like(tols)


for i in range(len(tols)):
    x , y , n= RungeKutta45(x_init,x_end,y_init,diff_func_1a,h0,tol = tols[i], alpha=0.8)
    
    error[i] = np.linalg.norm(y[:,0]-anal_solution(x))
    
Iterations = np.zeros_like(alphas)
for i in range(len(alphas)):
    x , y , n= RungeKutta45(x_init,x_end,y_init,diff_func_1a,h0, alpha=alphas[i])
    print(alphas[i],n)
    Iterations[i] = n
    
    
    

plt.figure("Error/tols")
plt.plot(tols,error)
plt.xscale("log")
plt.yscale("log")

plt.figure("Iterations/alphas")
plt.plot(alphas, Iterations)
plt.show()
