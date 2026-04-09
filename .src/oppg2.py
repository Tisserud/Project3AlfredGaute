# %% {"tags" : ["imports"]}
import numpy as np
from numpy.linalg import inv
from scipy.integrate import solve_bvp

l = 1
epsilon = 1
# %% [markdown] 
"""
## Oppgave 2

### 2a
"""
# %% [code]

def toRealVector(complexMatrix : np.ndarray[np.ndarray[np.complex128]]) -> np.ndarray[np.float64]:
    arr = np.zeros(8)
    rav = np.ravel(complexMatrix, 'C')

    for i in range(4):
        arr[i] = rav[i].real
        arr[i + 4] = rav[i].imag

    return arr



def toComplexMatrix(realVector : np.ndarray) -> np.ndarray[np.ndarray[np.complex128]]:
    matrix =  np.zeros(4, dtype=np.complex128)

    for i in range(4):
        matrix[i] = realVector[i] + realVector[i +4] * 1j

    return matrix.reshape((2,2))


a = np.array([[1+5j, 2+6j],[3+7j, 4+8j]])

print(toRealVector(a))
print(toComplexMatrix(toRealVector(a)))


# %% [markdown]
"""
### 2b
"""

def toComponentVectors(vec : np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return vec[:8], vec[8:16], vec[16:24], vec[24:32]


def toLongVector(
    m1 : np.ndarray,
    m2 : np.ndarray,
    m3 : np.ndarray,
    m4 : np.ndarray
    ) -> np.ndarray:

    return np.concatenate((m1, m2, m3, m4))

a = np.array([1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8])

m1,m2,m3,m4 = toComponentVectors(a)
b = toLongVector(m1, m2, m3, m4)

print(m1,'\n',m2,'\n',m3,'\n',m4,'\n')
print(b)


# %% [markdown]
"""
### 2c
"""

# %% [code]
def oppg2c(
    gamma : np.ndarray[np.ndarray[np.complex128]],
    gammaTilde : np.ndarray[np.ndarray[np.complex128]],
    omega : np.ndarray[np.ndarray[np.complex128]],
    omegaTilde : np.ndarray[np.ndarray[np.complex128]]
    ) -> np.ndarray[np.float64]:
    
    return toLongVector(
        toRealVector(gamma),
        toRealVector(gammaTilde),
        toRealVector(omega),
        toRealVector(omegaTilde))
        
def oppg2cInverse(
    vec : np.ndarray[np.float64]
    ) -> tuple[
        np.ndarray[np.ndarray[np.complex128]],
        np.ndarray[np.ndarray[np.complex128]],
        np.ndarray[np.ndarray[np.complex128]],
        np.ndarray[np.ndarray[np.complex128]]
    ]:

    m1, m2, m3, m4 = toComponentVectors(vec)
    gamma = toComplexMatrix(m1)
    gammaTilde = toComplexMatrix(m2)
    omega = toComplexMatrix(m3)
    omegaTilde = toComplexMatrix(m4)

    return gamma, gammaTilde, omega, omegaTilde

# %% [markdown]
"""
$$
\vec{v} = \\
    Re(\gamma_{11}) Re(\gamma_{12}) Re(\gamma_{21}) Re(\gamma_{22}) Im(\gamma_{11}) Im(\gamma_{12}) Im(\gamma_{21})  Im (\gamma_{22}) \\
    Re(\tilde\gamma_{11}) Re(\tilde\gamma_{12}) Re(\tilde\gamma_{21}) Re(\tilde\gamma_{22}) Im(\tilde\gamma_{11}) Im(\tilde\gamma_{12}) Im(\tilde\gamma_{21})  Im (\tilde\gamma_{22}) \\
    Re(\omega_{11}) Re(\omega_{12}) Re(\omega_{21}) Re(\omega_{22}) Im(\omega_{11}) Im(\omega_{12}) Im(\omega_{21})  Im (\omega_{22}) \\ 
    Re(\tilde\omega_{11}) Re(\tilde\omega_{12}) Re(\tilde\omega_{21}) Re(\tilde\omega_{22}) Im(\tilde\omega_{11}) Im(\tilde\omega_{12}) Im(\tilde\omega_{21})  Im (\tilde\omega_{22}) \\
$$

### 2d
"""
# %% [code]
def deriv(vec : np.ndarray[np.float64], epsilon : float):
    gamma, gammaTilde, omega, omegaTilde = oppg2cInverse(vec)

    N = inv(1 - gamma @ gammaTilde)
    N_tilde = inv(1 - gammaTilde @ gamma)
    
    omegaDeriv = -2j * (epsilon + 0.01j) - 2 * omega @ N_tilde @ gammaTilde @ omega
    omegaTildeDeriv = -2j * (epsilon + 0.01j) - 2 * omegaTilde @ N @ gamma @ omegaTilde

    return oppg2c(omega, omegaTilde, omegaDeriv, omegaTildeDeriv)


# %% [markdown]
"""### 2e
"""

# %% [code]
def stepFunc(x : np.ndarray, vec : np.ndarray):
    return np.apply_along_axis(deriv, 1, vec, epsilon)

# %% [markdown]
"""
### 2f
"""

# %% [code]
def bc(
    v_left : np.ndarray,
    v_right : np.ndarray,
    ):

    gamma_left, gammaTilde_left, omega_left, omegaTilde_left = oppg2cInverse(v_left)
    gamma_right, gammaTilde_right, omega_right, omegaTilde_right = oppg2cInverse(v_right)

    return oppg2c(
        omega_left - 1/(3*l) * gamma_left,
        omegaTilde_left - 1/(3*l) * gammaTilde_left,
        omega_right + 1/(3*l) * gamma_right,
        omegaTilde_right + 1/(3*l) * gammaTilde_right)
# %%
l = 1
m = 101
x = np.linspace(0,l,m)
y = np.zeros((32,m))
for i in range(3):
    epsilon = i
    sol = solve_bvp(stepFunc, bc, x, y, verbose = 2)
    print(sol)
# %%
