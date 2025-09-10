# Computational Methods & Data Analysis Toolkit
A collection of Python implementations demonstrating various computational methods, statistical analyses, and machine learning applications across different domains.

## 📁 Repository Structure
### 1. BTC Price Analysis (btc_price_analysis.py)
Time series analysis of Bitcoin prices featuring:
Moving average calculations (SMA 20 & 50 days)
Autocorrelation function analysis
Logarithmic transformations and growth rate calculations
Statistical identification of significant price movement periods

### 2. Distribution Generation & Validation (distribution_generation_validation.py)
Statistical distribution simulation and validation system:
Gaussian distribution via Central Limit Theorem
Chi-Squared distribution generation
F-distribution construction from chi-squared components
Student's t-distribution simulation
Goodness-of-fit testing with χ² and R² metrics
Visual comparison of empirical vs theoretical distributions

### 3. Integral Equation Solvers (integral_equation_solvers_comparison.py)
Numerical solution of 2D Fredholm integral equations using:
Gradient Descent: Basic iterative optimization
Two-step Gradient Descent: Enhanced convergence variant
BiCGStab: Stabilized Biconjugate Gradient method
Nyström discretization with midpoint quadrature
Performance comparison across grid sizes (N=10 to N=40)

### 4. Iterative Linear Solvers Comparison (iterative_solvers_comparison.py)
Benchmarking framework for linear system solvers:
Simple Iteration: Basic fixed-point method
Gradient Descent: First-order optimization
Two-step Gradient Descent: Accelerated version
BiCGSTAB: Krylov subspace method
Comprehensive evaluation of iteration count, execution time, and relative error

### 5. Hybrid Movie Recommendation System (movie_recommendation_system.py)
Machine learning recommendation engine featuring:
Collaborative Filtering: SVD-based matrix factorization
Content-Based Filtering: Genre-based cosine similarity
Hybrid Approach: Combined recommendation strategy
Evaluation metrics: RMSE, Precision, and Recall

## 🛠️ Technical Features
Numerical Methods: Advanced iterative algorithms for linear systems
Statistical Analysis: Distribution fitting and validation techniques
Time Series Analysis: Financial data processing and transformation
Machine Learning: Hybrid recommendation systems with evaluation
Visualization: Comprehensive plotting and result presentation
Performance Benchmarking: Comparative analysis of algorithmic efficiency

## 📊 Applications
Financial Analysis: Cryptocurrency price modeling and trend detection
Statistical Simulation: Probability distribution generation and validation
Computational Mathematics: Integral equation solving and linear algebra
Recommendation Systems: Collaborative and content-based filtering
Algorithm Development: Iterative method implementation and optimization
