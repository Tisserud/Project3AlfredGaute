# %% {"tags" : ["imports"]}
import numpy as np


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
    ) -> tuple[np.ndarray[np.ndarray[np.complex128]]]:

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
    Re(\gamma_{11}) Re(\gamma_{12}) Re(\gamma_{21}) Re(\gamma_{22}) Im(\gamma_{11}) Im(\gamma_{12}) Im(\gamma_{21})  Im (\gamma_{22}) \\
    Re(\omega_{11}) Re(\omega_{12}) Re(\omega_{21}) Re(\omega_{22}) Im(\omega_{11}) Im(\omega_{12}) Im(\omega_{21})  Im (\omega_{22}) \\ 
    Re(\omega_{11}) Re(\omega_{12}) Re(\omega_{21}) Re(\omega_{22}) Im(\omega_{11}) Im(\omega_{12}) Im(\omega_{21})  Im (\omega_{22}) \\
$$
"""
# %%
