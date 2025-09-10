#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
This script solves a 2D Fredholm integral equation of the second kind using three different iterative methods:
1. Gradient Descent
2. Two-step Gradient Descent
3. Stabilized Biconjugate Gradient (BiCGStab)

The equation is discretized using the Nyström method with midpoint quadrature on a uniform grid.
The performance of each method is compared in terms of relative error and number of matrix-vector multiplications.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define problem parameters
x1_center = 50
x2_center = 50
H = 10  # Side length of the square domain
N = 10  # Number of subdivisions along each axis

# Calculate domain boundaries
x1_left = x1_center - H / 2
x1_right = x1_center + H / 2
x2_left = x2_center - H / 2
x2_right = x2_center + H / 2

# Calculate step size and cell centers
h = H / N
x1_centers = [i for i in np.arange(x1_left + h / 2, x1_right, h)]
x2_centers = [i for i in np.arange(x2_left + h / 2, x2_right, h)]

# Create grid of cell centers
squares_centers = [(i, j) for j in x2_centers for i in x1_centers]

# Define kernel function (Green's function for Laplace equation in 2D)
def K(x, y):
    """Green's function for Laplace equation in 2D."""
    return 1 / (4 * np.pi * np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2))

# Construct system matrix using Nyström method
A = []
for i in range(N**2):
    row = []
    for j in range(N**2):
        if i == j:
            # Diagonal elements (identity part of the equation)
            row.append(0)
        else:
            # Off-diagonal elements (integral operator part)
            row.append(K(squares_centers[i], squares_centers[j]) * h**2)
    A.append(row)

# Form the complete matrix (I + K)
A = np.eye(N**2) + np.array(A)

# Define right-hand side function
f = [np.sin(c[0]) + np.cos(c[1]) for c in squares_centers]

def relative_error(x_true, x_sol):
    """Calculate relative error between true and approximate solutions."""
    return np.sum(np.abs(x_true - x_sol)) / np.sum(np.abs(x_true))

def gradient_descent(A, f, epsilon=0.0001):
    """
    Solve linear system using gradient descent method.
    
    Parameters:
    A: System matrix
    f: Right-hand side vector
    epsilon: Convergence tolerance
    
    Returns:
    u: Solution vector
    counter: Number of matrix-vector multiplications performed
    """
    N = len(f)
    u_prev = np.zeros(N)
    A_star = A.conj().T  # Hermitian transpose
    counter = 0
    
    while True:
        # Compute residual
        r = np.dot(A, u_prev) - f
        A_star_r = np.dot(A_star, r)
        
        # Compute step size
        numerator = np.linalg.norm(A_star_r)**2
        denominator = np.linalg.norm(np.dot(A, A_star_r))**2
        counter += 3  # Count matrix-vector multiplications
        
        # Update solution
        u_next = u_prev - (numerator / denominator) * A_star_r
        
        # Check convergence
        if np.linalg.norm(u_next - u_prev) / np.linalg.norm(f) < epsilon:
            return u_next, counter
        
        u_prev = u_next

def gradient_descent_twostep(A, f, epsilon=0.0001):
    """
    Solve linear system using two-step gradient descent method.
    
    Parameters:
    A: System matrix
    f: Right-hand side vector
    epsilon: Convergence tolerance
    
    Returns:
    u: Solution vector
    counter: Number of matrix-vector multiplications performed
    """
    N = len(f)
    u_prev = np.zeros(N)
    A_star = A.conj().T  # Hermitian transpose
    
    # Initial step
    r_prev = np.dot(A, u_prev) - f
    A_star_r = np.dot(A_star, r_prev)
    numerator = np.linalg.norm(A_star_r)**2
    denominator = np.linalg.norm(np.dot(A, A_star_r))**2
    u_curr = u_prev - (numerator / denominator) * A_star_r
    counter = 3  # Count matrix-vector multiplications
    
    # Check initial convergence
    if np.linalg.norm(u_curr - u_prev) / np.linalg.norm(f) < epsilon:
        return u_curr, counter
    
    # Main iteration loop
    while True:
        r_curr = np.dot(A, u_curr) - f
        delta_r = np.linalg.norm(r_curr - r_prev)**2
        A_star_r = np.dot(A_star, r_curr)
        counter += 2  # Count matrix-vector multiplications
        
        # Set up and solve 2x2 system for parameters
        left = np.array([
            [delta_r, np.linalg.norm(A_star_r)**2],
            [np.linalg.norm(A_star_r)**2, np.linalg.norm(np.dot(A, A_star_r))**2]
        ])
        right = np.array([0, np.linalg.norm(A_star_r)**2])
        a, g = np.linalg.solve(left, right)
        
        # Update solution
        u_next = u_curr - a * (u_curr - u_prev) - g * A_star_r
        
        # Check convergence
        if np.linalg.norm(u_next - u_curr) / np.linalg.norm(f) < epsilon:
            return u_next, counter
        
        # Update variables for next iteration
        u_prev = u_curr
        u_curr = u_next
        r_prev = r_curr

def stabilized_gradient(A, f, epsilon=0.0001):
    """
    Solve linear system using BiCGStab method.
    
    Parameters:
    A: System matrix
    f: Right-hand side vector
    epsilon: Convergence tolerance
    
    Returns:
    u: Solution vector
    counter: Number of matrix-vector multiplications performed
    """
    N = len(f)
    u_prev = np.zeros(N)
    r_prev = f - np.dot(A, u_prev)
    r_tilda = r_prev  # Initial shadow residual
    
    # Initialize parameters
    ro_prev = 1
    alpha_prev = 1
    omega_prev = 1
    
    # Initialize vectors
    v_prev = np.zeros(N)
    p_prev = np.zeros(N)
    s_prev = np.zeros(N)
    
    counter = 1  # Count matrix-vector multiplications
    
    while True:
        # BiCGStab iteration
        ro_curr = np.sum(np.conj(r_tilda) * r_prev)
        beta_curr = (ro_curr / ro_prev) * (alpha_prev / omega_prev)
        p_curr = r_prev + beta_curr * (p_prev - omega_prev * v_prev)
        v_curr = np.dot(A, p_curr)
        alpha_curr = ro_curr / np.sum(np.conj(r_tilda) * v_curr)
        s_curr = r_prev - alpha_curr * v_curr
        t_curr = np.dot(A, s_curr)
        counter += 2  # Count matrix-vector multiplications
        
        omega_curr = np.sum(t_curr * s_curr) / np.sum(t_curr * t_curr)
        u_curr = u_prev + omega_curr * s_curr + alpha_curr * p_curr
        r_curr = s_curr - omega_curr * t_curr
        
        # Check convergence
        if np.linalg.norm(u_curr - u_prev) / np.linalg.norm(f) < epsilon:
            return u_curr, counter
        
        # Update variables for next iteration
        u_prev = u_curr
        r_prev = r_curr
        ro_prev, alpha_prev, omega_prev = ro_curr, alpha_curr, omega_curr
        v_prev, p_prev, s_prev = v_curr, p_curr, s_curr

# Create results table
table = pd.DataFrame(columns=[
    "Number of points", 
    "Gradient Descent (error)", 
    "Gradient Descent (multiplications)", 
    "Two-step Gradient Descent (error)", 
    "Two-step Gradient Descent (multiplications)", 
    "BiCGStab (error)", 
    "BiCGStab (multiplications)"
])

# Solve for N=10
true_solution = np.linalg.solve(A, f)
gd_solution, gd_dots = gradient_descent(A, f)
gd2_solution, gd2_dots = gradient_descent_twostep(A, f)
bi_solution, bi_dots = stabilized_gradient(A, f)

# Add results to table
table.loc[len(table)] = {
    "Number of points": N, 
    "Gradient Descent (error)": relative_error(true_solution, gd_solution), 
    "Gradient Descent (multiplications)": gd_dots, 
    "Two-step Gradient Descent (error)": relative_error(true_solution, gd2_solution), 
    "Two-step Gradient Descent (multiplications)": gd2_dots, 
    "BiCGStab (error)": relative_error(true_solution, bi_solution), 
    "BiCGStab (multiplications)": bi_dots
}

# Plot solutions for N=10
_, ax = plt.subplots(3, 1, figsize=(10, 20))

cax = ax[0].imshow(gd_solution.reshape(N, N), interpolation="nearest", 
                   extent=[x1_centers[0], x1_centers[-1], x2_centers[0], x2_centers[-1]])
ax[0].set_xticks(x1_centers)
ax[0].set_yticks(x2_centers)
ax[0].set_title("Gradient Descent Solution")
plt.colorbar(cax, ax=ax[0])

cax = ax[1].imshow(gd2_solution.reshape(N, N), interpolation="nearest", 
                   extent=[x1_centers[0], x1_centers[-1], x2_centers[0], x2_centers[-1]])
ax[1].set_xticks(x1_centers)
ax[1].set_yticks(x2_centers)
ax[1].set_title("Two-step Gradient Descent Solution")
plt.colorbar(cax, ax=ax[1])

cax = ax[2].imshow(bi_solution.reshape(N, N), interpolation="nearest", 
                   extent=[x1_centers[0], x1_centers[-1], x2_centers[0], x2_centers[-1]])
ax[2].set_xticks(x1_centers)
ax[2].set_yticks(x2_centers)
ax[2].set_title("BiCGStab Solution")
plt.colorbar(cax, ax=ax[2])

plt.tight_layout()
plt.show()

# Repeat for N=20
N = 20
h = H / N
x1_centers = [i for i in np.arange(x1_left + h / 2, x1_right, h)]
x2_centers = [i for i in np.arange(x2_left + h / 2, x2_right, h)]
squares_centers = [(i, j) for j in x2_centers for i in x1_centers]

A = []
for i in range(N**2):
    row = []
    for j in range(N**2):
        if i == j:
            row.append(0)
        else:
            row.append(K(squares_centers[i], squares_centers[j]) * h**2)
    A.append(row)

A = np.eye(N**2) + np.array(A)
f = [np.sin(c[0]) + np.cos(c[1]) for c in squares_centers]

true_solution = np.linalg.solve(A, f)
gd_solution, gd_dots = gradient_descent(A, f)
gd2_solution, gd2_dots = gradient_descent_twostep(A, f)
bi_solution, bi_dots = stabilized_gradient(A, f)

table.loc[len(table)] = {
    "Number of points": N, 
    "Gradient Descent (error)": relative_error(true_solution, gd_solution), 
    "Gradient Descent (multiplications)": gd_dots, 
    "Two-step Gradient Descent (error)": relative_error(true_solution, gd2_solution), 
    "Two-step Gradient Descent (multiplications)": gd2_dots, 
    "BiCGStab (error)": relative_error(true_solution, bi_solution), 
    "BiCGStab (multiplications)": bi_dots
}

# Plot solutions for N=20
_, ax = plt.subplots(3, 1, figsize=(10, 20))

cax = ax[0].imshow(gd_solution.reshape(N, N), interpolation="nearest", 
                   extent=[x1_centers[0], x1_centers[-1], x2_centers[0], x2_centers[-1]])
ax[0].set_xticks(x1_centers)
ax[0].set_xticklabels(x1_centers, rotation=90)
ax[0].set_yticks(x2_centers)
ax[0].set_title("Gradient Descent Solution (N=20)")
plt.colorbar(cax, ax=ax[0])

cax = ax[1].imshow(gd2_solution.reshape(N, N), interpolation="nearest", 
                   extent=[x1_centers[0], x1_centers[-1], x2_centers[0], x2_centers[-1]])
ax[1].set_xticks(x1_centers)
ax[1].set_xticklabels(x1_centers, rotation=90)
ax[1].set_yticks(x2_centers)
ax[1].set_title("Two-step Gradient Descent Solution (N=20)")
plt.colorbar(cax, ax=ax[1])

cax = ax[2].imshow(bi_solution.reshape(N, N), interpolation="nearest", 
                   extent=[x1_centers[0], x1_centers[-1], x2_centers[0], x2_centers[-1]])
ax[2].set_xticks(x1_centers)
ax[2].set_xticklabels(x1_centers, rotation=90)
ax[2].set_yticks(x2_centers)
ax[2].set_title("BiCGStab Solution (N=20)")
plt.colorbar(cax, ax=ax[2])

plt.tight_layout()
plt.show()

# Repeat for N=30
N = 30
h = H / N
x1_centers = [i for i in np.arange(x1_left + h / 2, x1_right, h)]
x2_centers = [i for i in np.arange(x2_left + h / 2, x2_right, h)]
squares_centers = [(i, j) for j in x2_centers for i in x1_centers]

A = []
for i in range(N**2):
    row = []
    for j in range(N**2):
        if i == j:
            row.append(0)
        else:
            row.append(K(squares_centers[i], squares_centers[j]) * h**2)
    A.append(row)

A = np.eye(N**2) + np.array(A)
f = [np.sin(c[0]) + np.cos(c[1]) for c in squares_centers]

true_solution = np.linalg.solve(A, f)
gd_solution, gd_dots = gradient_descent(A, f)
gd2_solution, gd2_dots = gradient_descent_twostep(A, f)
bi_solution, bi_dots = stabilized_gradient(A, f)

table.loc[len(table)] = {
    "Number of points": N, 
    "Gradient Descent (error)": relative_error(true_solution, gd_solution), 
    "Gradient Descent (multiplications)": gd_dots, 
    "Two-step Gradient Descent (error)": relative_error(true_solution, gd2_solution), 
    "Two-step Gradient Descent (multiplications)": gd2_dots, 
    "BiCGStab (error)": relative_error(true_solution, bi_solution), 
    "BiCGStab (multiplications)": bi_dots
}

# Plot solutions for N=30
_, ax = plt.subplots(3, 1, figsize=(20, 40))

cax = ax[0].imshow(gd_solution.reshape(N, N), interpolation="nearest", 
                   extent=[x1_centers[0], x1_centers[-1], x2_centers[0], x2_centers[-1]])
ax[0].set_xticks(x1_centers)
ax[0].set_xticklabels(x1_centers, rotation=90)
ax[0].set_yticks(x2_centers)
ax[0].set_title("Gradient Descent Solution (N=30)")
plt.colorbar(cax, ax=ax[0])

cax = ax[1].imshow(gd2_solution.reshape(N, N), interpolation="nearest", 
                   extent=[x1_centers[0], x1_centers[-1], x2_centers[0], x2_centers[-1]])
ax[1].set_xticks(x1_centers)
ax[1].set_xticklabels(x1_centers, rotation=90)
ax[1].set_yticks(x2_centers)
ax[1].set_title("Two-step Gradient Descent Solution (N=30)")
plt.colorbar(cax, ax=ax[1])

cax = ax[2].imshow(bi_solution.reshape(N, N), interpolation="nearest", 
                   extent=[x1_centers[0], x1_centers[-1], x2_centers[0], x2_centers[-1]])
ax[2].set_xticks(x1_centers)
ax[2].set_xticklabels(x1_centers, rotation=90)
ax[2].set_yticks(x2_centers)
ax[2].set_title("BiCGStab Solution (N=30)")
plt.colorbar(cax, ax=ax[2])

plt.tight_layout()
plt.show()

# Repeat for N=40
N = 40
h = H / N
x1_centers = [i for i in np.arange(x1_left + h / 2, x1_right, h)]
x2_centers = [i for i in np.arange(x2_left + h / 2, x2_right, h)]
squares_centers = [(i, j) for j in x2_centers for i in x1_centers]

A = []
for i in range(N**2):
    row = []
    for j in range(N**2):
        if i == j:
            row.append(0)
        else:
            row.append(K(squares_centers[i], squares_centers[j]) * h**2)
    A.append(row)

A = np.eye(N**2) + np.array(A)
f = [np.sin(c[0]) + np.cos(c[1]) for c in squares_centers]

true_solution = np.linalg.solve(A, f)
gd_solution, gd_dots = gradient_descent(A, f)
gd2_solution, gd2_dots = gradient_descent_twostep(A, f)
bi_solution, bi_dots = stabilized_gradient(A, f)

table.loc[len(table)] = {
    "Number of points": N,
    "Gradient Descent (error)": relative_error(true_solution, gd_solution),
    "Gradient Descent (multiplications)": gd_dots,
    "Two-step Gradient Descent (error)": relative_error(true_solution, gd2_solution),
    "Two-step Gradient Descent (multiplications)": gd2_dots,
    "BiCGStab (error)": relative_error(true_solution, bi_solution),
    "BiCGStab (multiplications)": bi_dots
}

# Display results table
print(table)

# Plot solutions for N=40
_, ax = plt.subplots(3, 1, figsize=(20, 40))

cax = ax[0].imshow(gd_solution.reshape(N, N), interpolation="nearest", 
                   extent=[x1_centers[0], x1_centers[-1], x2_centers[0], x2_centers[-1]])
ax[0].set_xticks(x1_centers)
ax[0].set_xticklabels(x1_centers, rotation=90)
ax[0].set_yticks(x2_centers)
ax[0].set_title("Gradient Descent Solution (N=40)")
plt.colorbar(cax, ax=ax[0])

cax = ax[1].imshow(gd2_solution.reshape(N, N), interpolation="nearest", 
                   extent=[x1_centers[0], x1_centers[-1], x2_centers[0], x2_centers[-1]])
ax[1].set_xticks(x1_centers)
ax[1].set_xticklabels(x1_centers, rotation=90)
ax[1].set_yticks(x2_centers)
ax[1].set_title("Two-step Gradient Descent Solution (N=40)")
plt.colorbar(cax, ax=ax[1])

cax = ax[2].imshow(bi_solution.reshape(N, N), interpolation="nearest", 
                   extent=[x1_centers[0], x1_centers[-1], x2_centers[0], x2_centers[-1]])
ax[2].set_xticks(x1_centers)
ax[2].set_xticklabels(x1_centers, rotation=90)
ax[2].set_yticks(x2_centers)
ax[2].set_title("BiCGStab Solution (N=40)")
plt.colorbar(cax, ax=ax[2])

plt.tight_layout()
plt.show()

