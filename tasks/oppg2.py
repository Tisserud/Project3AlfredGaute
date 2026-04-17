# %%
import numpy as np
from numpy.linalg import inv
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
from tqdm import trange
from scipy.integrate import simpson

I = np.identity(2)
# %% [markdown] 
"""
## Oppgave 2

### 2a
"""
# %% [code]

def toRealVector(
    complexMatrix : np.ndarray[tuple[int,int], np.dtype[np.complex128]]
    )->np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Convert complex 2x2 matrix to real 1x8 array, return array."""

    return np.concatenate((complexMatrix.real.flatten(), complexMatrix.imag.flatten()))



def toComplexMatrix(
    realVector : np.ndarray[tuple[int], np.dtype[np.float64]]
    )->np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
    """Convert real 1x8 vector to complex 2x2 matrix, return matrix."""

    return (realVector[:4]+realVector[4:]*1j).reshape((2,2))


def test2a() -> None:
    """Tests if toRealVector and toComplexMatrix are inverse functions."""

    for i in range(999):
        A = np.random.random(8)
        B = toRealVector(toComplexMatrix(A))
        assert not np.any(A-B)
    print("test2a passed")

test2a()
# %% [markdown]
"""
### 2b
"""

def toComponentVectors(
    vec : np.ndarray[tuple[int], np.dtype[np.float64]]
    )->tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split 1x8 array in four arrays, return tuple of the arrays."""

    return vec[:8], vec[8:16], vec[16:24], vec[24:32]


def toLongVector(
    m1 : np.ndarray,
    m2 : np.ndarray,
    m3 : np.ndarray,
    m4 : np.ndarray
    ) -> np.ndarray:
    """Concatenate 4 arrays, return concatenated array"""

    return np.concatenate((m1, m2, m3, m4))

def test2b() -> None:
    """Tests if toLongVector and toComponentVector are inverse."""
    for i in range(999):
        A = np.random.random(32)
        B = toComponentVectors(toLongVector(A))
        assert not np.any(A - B)
    print("test2b passed")
# %% [markdown]
"""
### 2c
"""

# %% [code]
def to_32_component_vector(
    gamma : np.ndarray[tuple[int, int], np.dtype[np.complex128]],
    gammaTilde : np.ndarray[tuple[int, int], np.dtype[np.complex128]],
    omega : np.ndarray[tuple[int, int], np.dtype[np.complex128]],
    omegaTilde : np.ndarray[tuple[int, int], np.dtype[np.complex128]]
    )->np.ndarray[int, np.dtype[np.float64]]:
    """Convert 4 complex matrices to real array of length 32."""

    return toLongVector(
        toRealVector(gamma),
        toRealVector(gammaTilde),
        toRealVector(omega),
        toRealVector(omegaTilde))
        
def to_2D_matrices(
    vec : np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> tuple[
        np.ndarray[tuple[int, int], np.dtype[np.complex128]],
        np.ndarray[tuple[int, int], np.dtype[np.complex128]],
        np.ndarray[tuple[int, int], np.dtype[np.complex128]],
        np.ndarray[tuple[int, int], np.dtype[np.complex128]]
    ]:
    """Convert real array of lenght 32 to 4 comples matrices."""

    m1, m2, m3, m4 = toComponentVectors(vec)
    return toComplexMatrix(m1),toComplexMatrix(m2),toComplexMatrix(m3),toComplexMatrix(m4)

def test2c():
    for i in range(999):
        A = np.random(32)
        B = to_32_component_vector(toComplexMatrix(A))
        assert not np.any(A - B)
    print("test2c passed")
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
    """Calculates the derivates of the vectorized Riccati matrices and their derivates.
    
    Parameters
    ----------
    vec : Real vectorized form of the Riccati matrices with their derivates.
    epsilon : dimensionless energy $(E - E_F)\Delta$
    Returns
    -------
    dx_vec : the derivative of vec.
    """
    gamma, gammaTilde, omega, omegaTilde = to_2D_matrices(vec)

    N = inv(I - gamma @ gammaTilde)
    N_tilde = inv(I - gammaTilde @ gamma)
    
    omegaDeriv = -2j * (epsilon + 0.01j)*gamma - 2 * omega @ N_tilde @ gammaTilde @ omega
    omegaTildeDeriv = -2j * (epsilon + 0.01j)*gammaTilde- 2 * omegaTilde @ N @ gamma @ omegaTilde

    return to_32_component_vector(omega, omegaTilde, omegaDeriv, omegaTildeDeriv)


# %% [markdown]
"""### 2e
"""

# %% [code]
def get_fun(epsilon : float) -> callable:
    """Returns function compatible with solve_bvp with parameter epsilon.
    
    Parameters
    ----------
    epsilon : the unitless energy of the system.
    
    Returns
    -------
    fun : function calculating derivates compatible with solve_bvp"""
    def fun(
        x : np.ndarray, 
        matrix : np.ndarray[tuple[int, int], np.dtype[np.float64]]
        )->np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Do derivate of all vectors in matrix, return derivative

        Parameters
        ----------
        x : argument required by solve_bvp, unused.
        matrix : (32, m) array containing the Riccati matrices and their derivates at m positions.

        Returns
        ----------
        dx_matrix : (32, m) array with the derivates of matrix.
        """
        return np.apply_along_axis(deriv, 0, matrix, epsilon)
    return fun
# %% [markdown]
"""
### 2f
"""

# %% [code]
def get_bc_normal_metal(l : float) -> callable:
    """Get bc function for normal metal compatible with solve_bvp with parameter l
    
    Parameters
    ----------
    l : the lenght of the interfacing metal.
    Returns
    -------
    bc : function calculating boundary condition residuals compatible with solve_bvp."""
    def bc(
        v_left : np.ndarray,
        v_right : np.ndarray,
        )->np.ndarray[int, np.dtype[np.float64]]:
        """Get residuals from left and right boundary conditions
        
        Parameters
        ----------
        v_left : Riccati matricies and their derivates at x = 0
        v_right : Riccati matrices and their derivates at x = l
        Returns
        -------
        residuals : residuals to optimize."""
        
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
    """Returns the density D/D_0 as function of vectorized Riccati matrices"""
    gamma, gammaTilde, _, _ = to_2D_matrices(vec)
    
    N = inv(I - gamma @ gammaTilde)
    N_tilde = inv(I - gammaTilde @ gamma)

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
def get_ricatti(l : float, epsilon : float, phi_L : float, phi_R :float):
        """Get the riccati matrices at the boundary of the interfacing metal.
        
        Paramenters
        -----------
        l :             length of the interfacing metal.
        epsilon :       normalized energy.
        phi_L :         the phase of the superconducting metal to the left of the interfacing metal.
        phi_R :         the phase of the superconducting metal to the right of the interfacing metal.

        Returns
        -------
        gamma_L :       Riccati gamma matrix for the superconducting metal to the left of the interfacing metal.
        gamma_L_tilde : Riccati gamma ~ matric for the superconducting metal to the left of the interfacing metal.
        gamma_R :       Riccati gamma matrix for the superconducting metal to the right of the interfacing metal.
        gamma_R_tilde : Riccati gamma ~ matrix for the superconducting metal to the right of the interfacing metal.
        """
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


def get_bc_superconductor(
        l : float,
        epsilon : float,
        phi_L : float,
        phi_R : float
        )->callable:
    """Get bc function for superconductor compatible with solve_bvp with parameters
    
    Parameters
    ----------
    l :             length of the interfacing metal.
    epsilon :       normalized energy.
    phi_L :         the phase of the superconducting metal to the left of the interfacing metal.
    phi_R :         the phase of the superconducting metal to the right of the interfacing metal.

    Returns
    -------
    bc : function calculating boundary condition residuals compatible with solve_bvp."""
    
    gamma_L, gamma_L_tilde, gamma_R, gamma_R_tilde = get_ricatti(l, epsilon, phi_L, phi_R)

    def bc(
        v_left : np.ndarray[int, np.dtype[np.float64]],
        v_right : np.ndarray[int, np.dtype[np.float64]],
        )->np.ndarray[int, np.dtype[np.float64]]:
        """Get residuals from left and right boundary conditions
        
        Parameters
        ----------
        v_left : Riccati matricies and their derivates at x = 0
        v_right : Riccati matrices and their derivates at x = l
        Returns
        -------
        residuals : residuals to optimize."""

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
results_per_l = []
for l in [0.5, 1, 2]:
    results = []
    m = 51
    x = np.linspace(0,l,m)
    y = np.zeros((32,m))
    epsilon_arr = np.linspace(2, 0, m)
    phi_L = phi_R = 0

    for i in trange(m, desc = f"l = {l}"):
        res= solve_bvp(get_fun(epsilon_arr[i]), get_bc_superconductor(l, epsilon_arr[i], phi_L, phi_R), x, y)
        results.append(res)
        y = res.sol(x)
    plt.plot(epsilon_arr, [np.apply_along_axis(normDensity, 0, res.sol(l/2)) for res in results], label = f"l = {l}")
    results_per_l.append(results)
plt.legend()
plt.grid()
plt.show()

# %% [markdown]
"""
### 2l
"""

# %%
results_for_1 = results_per_l[1]

def current_density(vec : np.ndarray[tuple[int], np.dtype[np.float64]]) -> float:
    """Returns current density integrand as function of Riccati matrices.
    
    Paramenters
    -----------
    vec : (32,) array with Riccati matrices and their derivates at a given position and energy.

    Returns
    -------
    curr_density : the current density at given position.
    """

    gamma, gamma_tilde, omega, omega_tilde = to_2D_matrices(vec)

    N = inv(I - gamma @ gamma_tilde)
    N_tilde = inv (I - gamma @ gamma_tilde)

    dx_N = N @ (omega @ gamma_tilde + gamma @ omega_tilde) @ N
    dx_N_tilde = N_tilde @ (omega_tilde @ gamma + gamma_tilde @ omega) @ N_tilde

    g = np.block([[2*N-I, 2 * N @gamma], [-2*N_tilde@gamma_tilde, -2*N_tilde+I]])
    dx_g = 2* np.block([[dx_N, N @omega + dx_N @gamma], [-N_tilde @omega_tilde - dx_N_tilde @ gamma_tilde, -dx_N_tilde]])

    rho = np.asarray([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]])

    return np.trace(rho @ (g @ dx_g - dx_g @ g)).real


res_2 = results_for_1[0]
res_1_5 = results_for_1[m//4]
res_1 = results_for_1[m//2]
res_0_5 = results_for_1[3*m//4]
res_0 = results_for_1[-1]

x = np.linspace(0, 1, m)

for idx, res in enumerate((res_0, res_0_5, res_1, res_1_5, res_2)):
    
    plt.plot(x, np.apply_along_axis(current_density, 0, res.sol(x)), label = f"epsilon = {idx/2}")
plt.legend()
plt.grid()
plt.show()
# %% [markdown]
"""
### 2m
"""
# %%
l = 1
m = 51
x = np.linspace(0,l,m)
y = np.zeros((32,m))
epsilon_arr = np.linspace(2, 0, m)
phi_L = 1
phi_R = 0

results = []
for i in trange(m, desc = f"l = {l}"):
    res= solve_bvp(get_fun(epsilon_arr[i]), get_bc_superconductor(l, epsilon_arr[i], phi_L, phi_R), x, y)
    results.append(res)
    y = res.sol(x)

# %%
plt.plot(epsilon_arr, [np.apply_along_axis(current_density, 0, res.sol(l/2)) for res in results], label = f"l = {l}")
plt.legend()
plt.grid()
plt.show()
# %%
for idx, res in enumerate(results):
    plt.plot(x , np.apply_along_axis(current_density, 0, res.sol(x)), label = f"epsilon = {epsilon_arr[i]}")
plt.show()

# %% [markdown]
"""
### 2n
"""
# %%
n = 10
m = 51

l = 1
x = np.linspace(0,l,m)
y = np.zeros((32,m))

phi_R = 0
phi_L_arr = np.linspace(0, 2*np.pi, n)
epsilon_arr = np.linspace(2, 0, m)

results_per_phi = []
for i in range(n):
    results = []
    for j in trange(m, desc = f"delta_phi = {phi_L_arr[i]}"):
        res= solve_bvp(get_fun(epsilon_arr[j]), get_bc_superconductor(l, epsilon_arr[j], phi_L_arr[i], phi_R), x, y)
        results.append(res)
        y = res.sol(x)
    results_per_phi.append(results)

# %%
j_phi = np.array([[np.apply_along_axis(current_density, 0, res.sol(l/2)) for res in results] for results in results_per_phi])
I_phi = -simpson(j_phi, epsilon_arr)

plt.plot(phi_L_arr, I_phi)
plt.show()