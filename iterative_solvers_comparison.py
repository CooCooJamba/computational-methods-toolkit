import numpy as np
import matplotlib.pyplot as plt
import time


def create_system(a, b, N):
    """
    Create a discretized linear system from an integral equation.
    
    Parameters:
    a (float): Lower bound of the integration interval
    b (float): Upper bound of the integration interval
    N (int): Number of discretization points
    
    Returns:
    tuple: (A, x) where A is the system matrix and x is the grid points
    """
    h = np.abs(b - a) / N
    # Create midpoints of each subinterval
    x = np.array([i * h + h / 2 + a for i in range(N)])
    y = x.copy()
    
    # Create the matrix representation of the integral operator
    A = []
    for i in x:
        row = []
        for j in y:
            row.append(K(i, j) * h)
        A.append(row)
    
    # Add identity matrix to convert to Fredholm equation of second kind
    return np.array(A) + np.eye(N), x


def u_x(x):
    """Exact solution function for validation."""
    return 0.75 * x


def K(x, y):
    """Kernel function of the integral equation."""
    return x * y


def f_x(x):
    """Right-hand side function of the integral equation."""
    return x


def relative_error(u, x):
    """
    Calculate the relative error between numerical and exact solutions.
    
    Parameters:
    u (array): Numerical solution
    x (array): Grid points
    
    Returns:
    float: Relative error
    """
    error_numerator = []
    error_denominator = []
    for i in range(len(x)):
        error_numerator.append(np.abs(u_x(x[i]) - u[i]))
        error_denominator.append(np.abs(u_x(x[i])))
    
    return sum(error_numerator) / sum(error_denominator)


def simple_iteration(A, f, epsilon=0.0001):
    """
    Solve linear system using simple iteration method.
    
    Parameters:
    A (matrix): System matrix
    f (vector): Right-hand side vector
    epsilon (float): Convergence tolerance
    
    Returns:
    tuple: (solution, iterations, time) 
    """
    start = time.time()
    # Normalize the system for better convergence
    max_value = max(np.max(np.abs(A)), np.max(np.abs(f)))
    A = A / max_value
    f = f / max_value
    
    N = len(f)
    I = np.eye(N)
    B = I - A  # Iteration matrix
    
    u_prev = np.zeros(N)
    iterations = 1
    
    while True:
        u_next = np.dot(B, u_prev) + f
        
        # Check convergence
        if np.linalg.norm(u_next - u_prev) / np.linalg.norm(f) < epsilon:
            end = time.time()
            return u_next, iterations, end - start
        
        u_prev = u_next
        iterations += 1


def gradient_descent(A, f, epsilon=0.0001):
    """
    Solve linear system using gradient descent method.
    
    Parameters:
    A (matrix): System matrix
    f (vector): Right-hand side vector
    epsilon (float): Convergence tolerance
    
    Returns:
    tuple: (solution, iterations, time) 
    """
    start = time.time()
    N = len(f)
    u_prev = np.zeros(N)
    A_star = A.conj().T  # Conjugate transpose
    iterations = 1
    
    while True:
        # Compute residual
        r = np.dot(A, u_prev) - f
        A_star_r = np.dot(A_star, r)
        
        # Compute step size
        numerator = np.linalg.norm(A_star_r) ** 2
        denominator = np.linalg.norm(np.dot(A, A_star_r)) ** 2
        step_size = numerator / denominator
        
        # Update solution
        u_next = u_prev - step_size * A_star_r
        
        # Check convergence
        if np.linalg.norm(u_next - u_prev) / np.linalg.norm(f) < epsilon:
            end = time.time()
            return u_next, iterations, end - start
        
        u_prev = u_next
        iterations += 1


def gradient_descent_twostep(A, f, epsilon=0.0001):
    """
    Solve linear system using two-step gradient descent method.
    
    Parameters:
    A (matrix): System matrix
    f (vector): Right-hand side vector
    epsilon (float): Convergence tolerance
    
    Returns:
    tuple: (solution, iterations, time) 
    """
    start = time.time()
    N = len(f)
    u_prev = np.zeros(N)
    A_star = A.conj().T  # Conjugate transpose
    iterations = 1
    
    # Initial step
    r_prev = np.dot(A, u_prev) - f
    A_star_r = np.dot(A_star, r_prev)
    numerator = np.linalg.norm(A_star_r) ** 2
    denominator = np.linalg.norm(np.dot(A, A_star_r)) ** 2
    u_curr = u_prev - (numerator / denominator) * A_star_r
    
    while True:
        r_curr = np.dot(A, u_curr) - f
        delta_r = np.linalg.norm(r_curr - r_prev) ** 2
        A_star_r = np.dot(A_star, r_curr)
        
        # Set up and solve the 2x2 system for parameters a and g
        left_matrix = np.array([
            [delta_r, np.linalg.norm(A_star_r) ** 2],
            [np.linalg.norm(A_star_r) ** 2, np.linalg.norm(np.dot(A, A_star_r)) ** 2]
        ])
        right_vector = np.array([0, np.linalg.norm(A_star_r) ** 2])
        a, g = np.linalg.solve(left_matrix, right_vector)
        
        # Update solution
        u_next = u_curr - a * (u_curr - u_prev) - g * A_star_r
        
        # Check convergence
        if np.linalg.norm(u_next - u_curr) / np.linalg.norm(f) < epsilon:
            end = time.time()
            return u_next, iterations, end - start
        
        # Prepare for next iteration
        u_prev = u_curr
        u_curr = u_next
        r_prev = r_curr
        iterations += 1


def first_type_dot_product(u, v):
    """Dot product with conjugation of the first argument."""
    return np.sum(np.conjugate(u) * v)


def second_type_dot_product(u, v):
    """Standard dot product without conjugation."""
    return np.sum(u * v)


def biconjugate_gradient_stabilized(A, f, epsilon=0.0001):
    """
    Solve linear system using BiCGSTAB method.
    
    Parameters:
    A (matrix): System matrix
    f (vector): Right-hand side vector
    epsilon (float): Convergence tolerance
    
    Returns:
    tuple: (solution, iterations, time) 
    """
    start = time.time()
    N = len(f)
    
    # Initialize variables
    u_prev = np.zeros(N)
    r_prev = f - np.dot(A, u_prev)
    r_tilda = r_prev  # Shadow residual
    ro_prev = alpha_prev = omega_prev = 1
    v_prev = np.zeros(N)
    p_prev = np.zeros(N)
    s_prev = np.zeros(N)
    iterations = 1
    
    while True:
        ro_curr = first_type_dot_product(r_tilda, r_prev)
        beta_curr = (ro_curr / ro_prev) * (alpha_prev / omega_prev)
        
        # Update search direction
        p_curr = r_prev + beta_curr * (p_prev - omega_prev * v_prev)
        v_curr = np.dot(A, p_curr)
        alpha_curr = ro_curr / first_type_dot_product(r_tilda, v_curr)
        
        # Compute intermediate residual
        s_curr = r_prev - alpha_curr * v_curr
        t_curr = np.dot(A, s_curr)
        omega_curr = second_type_dot_product(t_curr, s_curr) / second_type_dot_product(t_curr, t_curr)
        
        # Update solution and residual
        u_curr = u_prev + omega_curr * s_curr + alpha_curr * p_curr
        r_curr = s_curr - omega_curr * t_curr
        
        # Check convergence
        if np.linalg.norm(u_curr - u_prev) / np.linalg.norm(f) < epsilon:
            end = time.time()
            return u_curr, iterations, end - start
        
        # Prepare for next iteration
        iterations += 1
        u_prev = u_curr
        r_prev = r_curr
        ro_prev, alpha_prev, omega_prev = ro_curr, alpha_curr, omega_curr
        v_prev, p_prev, s_prev = v_curr, p_curr, s_curr


def main():
    """Main function to compare iterative solvers."""
    # Define range of system sizes to test
    N_values = np.arange(100, 501, 10)
    
    # Test with two different tolerance values
    for epsilon in [0.0001, 0.00001]:
        rel_err = []
        iter_counts = []
        times = []
        
        for n in N_values:
            # Create system
            A, x_grid = create_system(0, 1, n)
            f = f_x(x_grid)
            
            # Solve with different methods
            si_solve, si_iter, si_time = simple_iteration(A, f, epsilon)
            gd_solve, gd_iter, gd_time = gradient_descent(A, f, epsilon)
            gd2_solve, gd2_iter, gd2_time = gradient_descent_twostep(A, f, epsilon)
            bgs_solve, bgs_iter, bgs_time = biconjugate_gradient_stabilized(A, f, epsilon)
            
            # Store results
            rel_err.append((
                relative_error(si_solve, x_grid),
                relative_error(gd_solve, x_grid),
                relative_error(gd2_solve, x_grid),
                relative_error(bgs_solve, x_grid)
            ))
            
            iter_counts.append((si_iter, gd_iter, gd2_iter, bgs_iter))
            times.append((si_time, gd_time, gd2_time, bgs_time))
        
        # Plot results
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 8))
        
        # Iteration count plot
        axes[0].plot(N_values, [i[0] for i in iter_counts], label="Simple Iteration")
        axes[0].plot(N_values, [i[1] for i in iter_counts], label="Gradient Descent")
        axes[0].plot(N_values, [i[2] for i in iter_counts], label="Two-step Gradient Descent")
        axes[0].plot(N_values, [i[3] for i in iter_counts], label="BiCGSTAB")
        axes[0].legend()
        axes[0].set_xlabel("N value")
        axes[0].set_ylabel("Iteration count")
        axes[0].set_title("Iteration Count of Methods")
        
        # Execution time plot
        axes[1].plot(N_values, [t[0] for t in times], label="Simple Iteration")
        axes[1].plot(N_values, [t[1] for t in times], label="Gradient Descent")
        axes[1].plot(N_values, [t[2] for t in times], label="Two-step Gradient Descent")
        axes[1].plot(N_values, [t[3] for t in times], label="BiCGSTAB")
        axes[1].legend()
        axes[1].set_xlabel("N value")
        axes[1].set_ylabel("Time (s)")
        axes[1].set_title("Method Execution Time")
        
        # Relative error plot
        axes[2].plot(N_values, [err[0] for err in rel_err], label="Simple Iteration")
        axes[2].plot(N_values, [err[1] for err in rel_err], label="Gradient Descent")
        axes[2].plot(N_values, [err[2] for err in rel_err], label="Two-step Gradient Descent")
        axes[2].plot(N_values, [err[3] for err in rel_err], label="BiCGSTAB")
        axes[2].legend()
        axes[2].set_xlabel("N value")
        axes[2].set_ylabel("Relative error")
        axes[2].set_title("Relative Error of Methods")
        
        fig.suptitle(f'Tolerance: {epsilon}')
        plt.show()


if __name__ == "__main__":
    main()

