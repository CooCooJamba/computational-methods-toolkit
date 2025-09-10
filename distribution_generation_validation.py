#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
This script generates samples from various probability distributions (Gaussian, Chi-Squared, 
F, and Student's t) and validates them against their theoretical definitions using 
goodness-of-fit tests and visual comparisons.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, chi2, f, t
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def generate_gaussian_distribution(k):
    """
    Generate and validate a Gaussian (normal) distribution using the Central Limit Theorem.
    
    Parameters:
    k (int): Number of uniform random variables to sum for CLT approximation
    """
    # Parameters for uniform distribution
    a = -1
    b = 1
    n_samples = 1000
    
    # Generate uniform random samples
    samples = np.random.uniform(a, b, (k, n_samples))
    sums = np.array(samples.sum(axis=0))
    
    # Plot histogram of first uniform distribution
    bins_count = int(round(1 + np.log2(samples[0].size)))
    plt.hist(samples[0], bins=bins_count, edgecolor='w', density=True, range=(a, b))
    plt.title(f'Uniform Distribution (k={k})')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()
    
    # Calculate sample mean and standard deviation
    mu = np.mean(sums)
    sigma = np.sqrt(1 / (sums.size - 1) * sum((sums - mu) ** 2))
    
    # Plot histogram of sums with normal distribution overlay
    bins_density = plt.hist(sums, bins=bins_count, edgecolor='black', 
                           facecolor='None', density=True)[:-1]
    plt.plot(sorted(sums), norm.pdf(sorted(sums), mu, sigma), color='c', label='Normal Fit')
    plt.axvline(x=mu, color='b', label='Mean')
    plt.axvline(x=mu + sigma, color='r', label='Mean ± Std')
    plt.axvline(x=mu - sigma, color='r')
    plt.title(f'Sum of {k} Uniform Variables (Gaussian Approximation)')
    plt.xlabel('Sum Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
    # Chi-squared goodness-of-fit test
    bins = plt.hist(sums, bins=bins_count, edgecolor='black', facecolor='None')[:-1]
    plt.clf()
    
    # Calculate expected frequencies under normal distribution
    Pi = np.array([norm.cdf(bins[1][i + 1], loc=mu, scale=sigma) - 
                   norm.cdf(bins[1][i], loc=mu, scale=sigma) 
                   for i in range(bins[1].size - 1)])
    nP = Pi * sums.size
    chi2_stat = sum((bins[0] - nP) ** 2 / nP)
    critical_chi2 = chi2.ppf(0.95, bins[0].size - 3)
    
    print(f'Chi-squared statistic: {chi2_stat:.4f} < Critical value: {critical_chi2:.4f} - ' 
          f'{"Pass" if chi2_stat < critical_chi2 else "Fail"}')
    
    # R² calculation for normality check using linear regression on log-densities
    x = [((bins_density[1][i] + bins_density[1][i + 1]) / 2 - mu) ** 2 
         for i in range(bins_density[1].size - 1)]
    x = np.array(x).reshape(-1, 1)
    y = np.log(bins_density[0])
    
    model = LinearRegression()
    model.fit(x, y)
    predictions = model.predict(x)
    
    plt.plot(x, predictions, label='Linear Fit')
    plt.plot(x, y, 'go', label='Actual Data')
    plt.title(f'Normality Check (R²) for k={k}')
    plt.xlabel('(x - μ)²')
    plt.ylabel('log(Density)')
    plt.grid()
    plt.legend()
    plt.show()
    
    r_squared = r2_score(y, predictions)
    print(f'R² score for normality: {r_squared:.4f}')


def generate_chi_squared_distribution(k):
    """
    Generate and validate a Chi-Squared distribution with k degrees of freedom.
    
    Parameters:
    k (int): Degrees of freedom for the chi-squared distribution
    """
    n_samples = 1000
    
    # Generate standard normal samples
    samples = np.random.normal(size=(k, n_samples))
    
    # Plot histogram of first normal distribution
    plt.hist(samples[0], edgecolor='w', density=True)
    plt.title('Standard Normal Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()
    
    # Transform to chi-squared distribution
    samples = np.array([(row - np.mean(row)) ** 2 / np.std(row) ** 2 for row in samples])
    sums = np.array(samples.sum(axis=0))
    
    bins_count = int(round(1 + np.log2(sums.size)))
    
    # Plot histogram with theoretical chi-squared distribution
    bins_density = plt.hist(sums, bins=bins_count, edgecolor='black', 
                           facecolor='None', density=True)[:-1]
    plt.plot(sorted(sums), chi2.pdf(sorted(sums), k), color='c', label=f'χ²({k})')
    plt.title(f'Chi-Squared Distribution (k={k})')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
    # Chi-squared goodness-of-fit test
    bins = plt.hist(sums, bins=bins_count, edgecolor='black', facecolor='None')[:-1]
    plt.clf()
    
    Pi = np.array([chi2.cdf(bins[1][i + 1], df=k) - 
                   chi2.cdf(bins[1][i], df=k) 
                   for i in range(bins[1].size - 1)])
    nP = Pi * sums.size
    chi2_stat = sum((bins[0] - nP) ** 2 / nP)
    critical_chi2 = chi2.ppf(0.95, bins[0].size - 2)
    
    print(f'Chi-squared statistic: {chi2_stat:.4f} < Critical value: {critical_chi2:.4f} - ' 
          f'{"Pass" if chi2_stat < critical_chi2 else "Fail"}')
    
    # R² calculation for distribution fit
    bin_centers = np.array([(bins_density[1][i + 1] + bins_density[1][i]) / 2 
                           for i in range(bins_density[1].size - 1)])
    theoretical_density = chi2.pdf(bin_centers, k)
    
    plt.plot(bins_density[0], bins_density[0], 'g', label='Perfect Fit')
    plt.plot(bins_density[0], theoretical_density, 'bo', label='Theoretical')
    plt.title(f'Chi-Squared Fit Quality (k={k})')
    plt.xlabel('Empirical Density')
    plt.ylabel('Theoretical Density')
    plt.grid()
    plt.legend()
    plt.show()
    
    r_squared = r2_score(bins_density[0], theoretical_density)
    print(f'R² score for chi-squared fit: {r_squared:.4f}')


def generate_f_distribution():
    """
    Generate and validate an F distribution using chi-squared distributions.
    """
    n_samples = 1000
    df1, df2 = 5, 10  # Degrees of freedom
    
    # Generate chi-squared random variables
    chi2_1 = chi2.rvs(df=df1, size=n_samples)
    chi2_2 = chi2.rvs(df=df2, size=n_samples)
    
    # Create F distribution
    f_values = (chi2_1 / df1) / (chi2_2 / df2)
    
    bins_count = int(round(1 + np.log2(f_values.size)))
    
    # Plot component chi-squared distributions
    plt.hist(chi2_1, edgecolor='w', density=True)
    plt.title(f'Chi-Squared Distribution (df={df1})')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()
    
    plt.hist(chi2_2, edgecolor='w', density=True)
    plt.title(f'Chi-Squared Distribution (df={df2})')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()
    
    # Plot F distribution with theoretical overlay
    bins_density = plt.hist(f_values, bins=bins_count, edgecolor='black', 
                           facecolor='None', density=True)[:-1]
    plt.plot(sorted(f_values), f.pdf(sorted(f_values), df1, df2), color='c', 
             label=f'F({df1},{df2})')
    plt.title(f'F Distribution (df1={df1}, df2={df2})')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
    # Chi-squared goodness-of-fit test
    bins = plt.hist(f_values, bins=bins_count, edgecolor='black', facecolor='None')[:-1]
    plt.clf()
    
    Pi = np.array([f.cdf(bins[1][i + 1], dfn=df1, dfd=df2) - 
                   f.cdf(bins[1][i], dfn=df1, dfd=df2) 
                   for i in range(bins[1].size - 1)])
    nP = Pi * f_values.size
    chi2_stat = sum((bins[0] - nP) ** 2 / nP)
    critical_chi2 = chi2.ppf(0.95, bins[0].size - 3)
    
    print(f'Chi-squared statistic: {chi2_stat:.4f} < Critical value: {critical_chi2:.4f} - ' 
          f'{"Pass" if chi2_stat < critical_chi2 else "Fail"}')
    
    # R² calculation for distribution fit
    bin_centers = np.array([(bins_density[1][i + 1] + bins_density[1][i]) / 2 
                           for i in range(bins_density[1].size - 1)])
    theoretical_density = f.pdf(bin_centers, df1, df2)
    
    plt.plot(bins_density[0], bins_density[0], 'g', label='Perfect Fit')
    plt.plot(bins_density[0], theoretical_density, 'bo', label='Theoretical')
    plt.title(f'F Distribution Fit Quality (df1={df1}, df2={df2})')
    plt.xlabel('Empirical Density')
    plt.ylabel('Theoretical Density')
    plt.grid()
    plt.legend()
    plt.show()
    
    r_squared = r2_score(bins_density[0], theoretical_density)
    print(f'R² score for F distribution fit: {r_squared:.4f}')


def generate_students_t_distribution(k):
    """
    Generate and validate a Student's t distribution with k degrees of freedom.
    
    Parameters:
    k (int): Degrees of freedom for the t-distribution
    """
    n_samples = 500
    
    # Generate standard normal samples
    samples = np.random.normal(size=(k, n_samples))
    
    # Create t-distribution values
    t_values = np.array([row[0] / np.sqrt(np.sum(row[1:] ** 2) / k) for row in samples.T])
    
    bins_count = int(round(1 + np.log2(t_values.size)))
    
    # Plot histogram of first normal distribution
    plt.hist(samples[0], edgecolor='w', density=True)
    plt.title('Standard Normal Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()
    
    # Plot t-distribution with theoretical overlay
    bins_density = plt.hist(t_values, bins=bins_count, edgecolor='black', 
                           facecolor='None', density=True)[:-1]
    plt.plot(sorted(t_values), t.pdf(sorted(t_values), k), color='c', label=f't({k})')
    plt.title(f"Student's t-Distribution (df={k})")
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
    # Chi-squared goodness-of-fit test
    bins = plt.hist(t_values, bins=bins_count, edgecolor='black', facecolor='None')[:-1]
    plt.clf()
    
    Pi = np.array([t.cdf(bins[1][i + 1], k) - 
                   t.cdf(bins[1][i], k) 
                   for i in range(bins[1].size - 1)])
    nP = Pi * t_values.size
    chi2_stat = sum((bins[0] - nP) ** 2 / nP)
    critical_chi2 = chi2.ppf(0.95, bins[0].size - 3)
    
    print(f'Chi-squared statistic: {chi2_stat:.4f} < Critical value: {critical_chi2:.4f} - ' 
          f'{"Pass" if chi2_stat < critical_chi2 else "Fail"}')
    
    # R² calculation for distribution fit
    bin_centers = np.array([(bins_density[1][i + 1] + bins_density[1][i]) / 2 
                           for i in range(bins_density[1].size - 1)])
    theoretical_density = t.pdf(bin_centers, k)
    
    plt.plot(bins_density[0], bins_density[0], 'g', label='Perfect Fit')
    plt.plot(bins_density[0], theoretical_density, 'bo', label='Theoretical')
    plt.title(f"Student's t-Distribution Fit Quality (df={k})")
    plt.xlabel('Empirical Density')
    plt.ylabel('Theoretical Density')
    plt.grid()
    plt.legend()
    plt.show()
    
    r_squared = r2_score(bins_density[0], theoretical_density)
    print(f'R² score for t-distribution fit: {r_squared:.4f}')


def main():
    """Main function to execute all distribution generation and validation tests."""
    np.random.seed(123)  # Set random seed for reproducibility
    
    print("=== Gaussian Distribution Tests ===")
    for k in [1, 2, 5]:
        print(f"\nTesting Gaussian approximation with k={k} uniform variables:")
        generate_gaussian_distribution(k)
    
    print("\n=== Chi-Squared Distribution Tests ===")
    for k in [2, 3, 10]:
        print(f"\nTesting Chi-Squared distribution with df={k}:")
        generate_chi_squared_distribution(k)
    
    print("\n=== F Distribution Test ===")
    generate_f_distribution()
    
    print("\n=== Student's t-Distribution Test ===")
    generate_students_t_distribution(3)


if __name__ == "__main__":
    main()

