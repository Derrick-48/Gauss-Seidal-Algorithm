# Gauss-Seidel Method: Solving Systems of Linear Equations in Python

## Overview

This project demonstrates the application of the Gauss-Seidel iterative method to solve systems of linear equations using Python. The Gauss-Seidel method is a numerical technique that provides approximate solutions to linear systems, commonly used in computational problems where direct methods are computationally expensive.

## Table of Contents

- [Introduction](#introduction)
- [Understanding the Gauss-Seidel Method](#understanding-the-gauss-seidel-method)
- [Python Implementation](#python-implementation)
- [Usage](#usage)
- [Convergence Considerations](#convergence-considerations)
- [References](#references)

## Introduction

In many scientific and engineering applications, we encounter systems of linear equations that need to be solved efficiently. The Gauss-Seidel method offers an iterative approach to approximate the solutions of such systems, making it particularly useful for large-scale problems where traditional direct methods may be impractical.

## Understanding the Gauss-Seidel Method

The Gauss-Seidel method iteratively refines guesses for the solution vector of a linear system \( A\mathbf{x} = \mathbf{b} \). Starting with an initial guess, the method updates each component of the solution vector using the latest available values, leading to improved approximations with each iteration.

### Mathematical Formulation

Given a system of linear equations represented as \( A\mathbf{x} = \mathbf{b} \), where \( A \) is a square matrix of coefficients, \( \mathbf{x} \) is the vector of unknowns, and \( \mathbf{b} \) is the right-hand side vector, the Gauss-Seidel update for the \( i \)-th component at iteration \( k+1 \) is:

\[ x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j=1}^{i-1} a_{ij} x_j^{(k+1)} - \sum_{j=i+1}^{n} a_{ij} x_j^{(k)} \right) \]

Here, \( a_{ii} \) are the diagonal elements of \( A \), and the sums account for the contributions of other variables.

## Python Implementation

Below is a Python implementation of the Gauss-Seidel method:

```python
import numpy as np

def gauss_seidel(A, b, tolerance=1e-10, max_iterations=1000):
    n = len(b)
    x = np.zeros_like(b, dtype=np.double)
    for k in range(max_iterations):
        x_new = np.copy(x)
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tolerance:
            return x_new
        x = x_new
    raise Exception("Gauss-Seidel method did not converge")

# Example usage:
A = np.array([[4, 1, 2],
              [3, 5, 1],
              [1, 1, 3]], dtype=np.double)
b = np.array([4, 7, 3], dtype=np.double)

solution = gauss_seidel(A, b)
print("Solution:", solution)




```
## Usage
### Define the Coefficient Matrix 
𝐴
A and Vector 
𝑏
b: Ensure that 
𝐴
A is a square matrix (same number of rows and columns) and that the dimensions of 
𝐴
A and 
𝑏
b are compatible.

Set Parameters: Adjust the tolerance and max_iterations parameters as needed. The tolerance determines the convergence criterion, while max_iterations sets a limit on the number of iterations to prevent infinite loops.

Call the Function: Use the gauss_seidel function with your defined 
𝐴
A and 
𝑏
b to compute the solution.

Handle Exceptions: Be prepared to handle exceptions in case the method does not converge within the specified number of iterations.

Convergence Considerations
The convergence of the Gauss-Seidel method depends on the properties of the coefficient matrix 
𝐴
A. The method is guaranteed to converge if 
𝐴
A is either:

Diagonally Dominant: Each diagonal element 
𝑎
𝑖
𝑖
a 
ii
​
  is greater than or equal to the sum of the absolute values of the other elements in the same row, i.e.,

∣
𝑎
𝑖
𝑖
∣
≥
∑
𝑗
≠
𝑖
∣
𝑎
𝑖
𝑗
∣
∣a 
ii
​
 ∣≥ 
j

=i
∑
​
 ∣a 
ij
​
 ∣
for all 
𝑖
i, with strict inequality for at least one row.

Symmetric and Positive Definite: 
𝐴
A is equal to its transpose (
𝐴
=
𝐴
𝑇
A=A 
T
 ) and all its eigenvalues are positive.

If these conditions are not met, the method may still converge, but convergence is not guaranteed. It's essential to analyze the properties of 
𝐴
A before applying the Gauss-Seidel method.

References
For further reading and a deeper understanding of the Gauss-Seidel method, consider the following resources:

- Gauss–Seidel Method - Wikipedia

- Gauss-Seidel Method - Mathematics LibreTexts

- Chapter 8: Gauss-Seidel Method | Introduction to Matrix Algebra

These references provide comprehensive explanations, examples, and insights into the Gauss-Seidel method and its applications.