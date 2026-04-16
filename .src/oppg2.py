# %% {"tags" : ["imports"]}
import numpy as np
from numpy.linalg import inv
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
from tqdm import trange
# %% [markdown] 
"""
## Oppgave 2

### 2a
"""
# %% [code]

def toRealVector(complexMatrix : np.ndarray[tuple[int,int], np.dtype[np.complex128]]) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    return np.concatenate((complexMatrix.real.flatten(), complexMatrix.imag.flatten()))



def toComplexMatrix(realVector : np.ndarray[tuple[int], np.dtype[np.float64]]) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
    return (realVector[:4] + realVector[4:] * 1j).reshape((2,2))


a = np.array([[1+5j, 2+6j],[3+7j, 4+8j]])

print(toRealVector(a))
print(toComplexMatrix(toRealVector(a)))


# %% [markdown]
"""
### 2b
"""

def toComponentVectors(vec : np.ndarray[tuple[int], np.dtype[np.float64]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
def to_32_component_vector(
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
        
def to_2D_matrices(
    vec : np.ndarray[np.float64]
    ) -> tuple[
        np.ndarray[np.ndarray[np.complex128]],
        np.ndarray[np.ndarray[np.complex128]],
        np.ndarray[np.ndarray[np.complex128]],
        np.ndarray[np.ndarray[np.complex128]]
    ]:

    m1, m2, m3, m4 = toComponentVectors(vec)
    return toComplexMatrix(m1),toComplexMatrix(m2),toComplexMatrix(m3),toComplexMatrix(m4)

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
    gamma, gammaTilde, omega, omegaTilde = to_2D_matrices(vec)

    N = inv(np.identity(2) - gamma @ gammaTilde)
    N_tilde = inv(np.identity(2) - gammaTilde @ gamma)
    
    omegaDeriv = -2j * (epsilon + 0.01j)*gamma - 2 * omega @ N_tilde @ gammaTilde @ omega
    omegaTildeDeriv = -2j * (epsilon + 0.01j)*gammaTilde- 2 * omegaTilde @ N @ gamma @ omegaTilde

    return to_32_component_vector(omega, omegaTilde, omegaDeriv, omegaTildeDeriv)


# %% [markdown]
"""### 2e
"""

# %% [code]
def get_fun(epsilon : float) -> callable:
    def fun(x : np.ndarray, vec : np.ndarray):
        return np.apply_along_axis(deriv, 0, vec, epsilon)
    return fun
# %% [markdown]
"""
### 2f
"""

# %% [code]
def get_bc_normal_metal(l : float) -> callable:
    def bc(
        v_left : np.ndarray,
        v_right : np.ndarray,
        ):

        gamma_left, gamma_left_tilde, omega_left, omega_left_tilde = to_2D_matrices(v_left)
        gamma_right, gamma_right_tilde, omega_right, omega_right_tilde = to_2D_matrices(v_right)

        return to_32_component_vector(
            omega_left - 1/(3*l) * gamma_left,
            omega_left_tilde - 1/(3*l) * gamma_left_tilde,
            omega_right + 1/(3*l) * gamma_right,
            omega_right_tilde + 1/(3*l) * gamma_right_tilde)
    return bc
# %% [markdown]
"""
### 2g
"""

# %%
l = 1
m = 101
x = np.linspace(0,l,m)
y = np.zeros((32,m))

results = dict()
for eps in range(3):
    results[eps] = solve_bvp(get_fun(eps), get_bc_normal_metal(l), x, y, verbose = 2)
    print(not results[eps].sol(x).any())

# %% [markdown]
"""
### 2h
"""
# %%
def normDensity(vec : np.ndarray) -> float:
    gamma, gammaTilde, _, _ = to_2D_matrices(vec)

    N = inv(np.identity(2) - gamma @ gammaTilde)
    N_tilde = inv(np.identity(2) - gammaTilde @ gamma)

    return (np.trace(N) + np.trace(N_tilde)).real/2 - 1

for eps in range(3):
    density = np.apply_along_axis(normDensity, 0, results[eps].sol(x))
    plt.plot(x, density, label = f'epsilon = {eps}')

plt.xlabel('x')
plt.ylabel('d')
plt.grid()
plt.title("Normal metals, energy density")
plt.legend()
plt.show()
# %% [markdown]
"""
### 2i
"""
# %%
def get_ricatti(l, epsilon, phi_L, phi_R):
        def theta(sigma, epsilon):
            return np.atanh(sigma/(epsilon + 0.01j))
        
        def s(sigma, epsilon):
            return np.sinh(theta(sigma, epsilon))
        
        def c(sigma, epsilon):
            return np.cosh(theta(sigma, epsilon))
        
        s_minus_c_minus = s(-1, epsilon)/(1 + c(-1, epsilon))
        s_plus_c_plus = s(1, epsilon)/(1 + c(1, epsilon))

        gamma_L = np.asarray(([0, s_plus_c_plus], [s_minus_c_minus, 0])) * np.exp(1j * phi_L)
        gamma_L_tilde = np.asarray(([0, s_minus_c_minus], [s_plus_c_plus, 0])) * np.exp(-1j * phi_L)
        gamma_R = np.asarray(([0, s_plus_c_plus], [s_minus_c_minus, 0])) * np.exp(1j * phi_R)
        gamma_R_tilde = np.asarray(([0, s_minus_c_minus], [s_plus_c_plus, 0])) * np.exp(-1j * phi_R)

        return gamma_L, gamma_L_tilde, gamma_R, gamma_R_tilde

def get_bc_superconductor(l : float, epsilon : float, phi_L : float, phi_R : float):

    gamma_L, gamma_L_tilde, gamma_R, gamma_R_tilde = get_ricatti(l, epsilon, phi_L, phi_R)

    I = np.identity(2)
    def bc(
        v_left : np.ndarray,
        v_right : np.ndarray,
        ):

        gamma_left, gamma_left_tilde, omega_left, omega_left_tilde = to_2D_matrices(v_left)
        gamma_right, gamma_right_tilde, omega_right, omega_right_tilde = to_2D_matrices(v_right)

        N_L = inv(I - gamma_L @ gamma_L_tilde)
        N_L_tilde = inv(I - gamma_L_tilde @ gamma_L)
        N_R = inv(I - gamma_R @ gamma_R_tilde)
        N_R_tilde = inv(I - gamma_R_tilde @ gamma_R)

        return to_32_component_vector(
            omega_left + 1/(3*l) * (I - gamma_left @ gamma_L_tilde) @ N_L @ (gamma_L - gamma_left),
            omega_left_tilde + 1/(3*l) * (I - gamma_left_tilde @ gamma_L) @ N_L_tilde @ (gamma_L_tilde - gamma_left_tilde),
            omega_right - 1/(3*l) * (I - gamma_right @ gamma_R_tilde) @ N_R @ (gamma_R - gamma_right),
            omega_right_tilde - 1/(3*l) * (I - gamma_right_tilde @ gamma_R) @ N_R_tilde @ (gamma_R_tilde - gamma_right_tilde)
            )
    
    return bc

# %% [markdown]
"""
### 2j
"""
# %%
l = 1
m = 101
x = np.linspace(0,l,m)
y = np.zeros((32,m))
epsilon = 2
phi_L = phi_R = 0
res = solve_bvp(get_fun(epsilon), get_bc_superconductor(l, epsilon, phi_L, phi_R), x, y, verbose = 2)

density = np.apply_along_axis(normDensity, 0, res.sol(x))
plt.plot(x, density, label = f'epsilon = {epsilon}')
plt.xlabel('x')
plt.ylabel('d')
plt.grid()
plt.title("Superconductor, energy density")
plt.legend()
plt.show()

# %% [markdown]
"""
### 2k
"""

# %%
for l in [0.5, 1, 2]:
    results = []
    m = 101
    x = np.linspace(0,l,m)
    y = np.zeros((32,m))
    epsilon_arr = np.linspace(2, 0, m)
    phi_L = phi_R = 0

    for i in trange(m, desc = f"l = {l}"):
        res= solve_bvp(get_fun(epsilon_arr[i]), get_bc_superconductor(l, epsilon_arr[i], phi_L, phi_R), x, y)
        results.append(res)
        y = res.sol(x)
    plt.plot(epsilon_arr, [np.apply_along_axis(normDensity, 0, res.sol(l/2)) for res in results], label = f"l = {l}")
plt.legend()
plt.grid()
plt.show()
