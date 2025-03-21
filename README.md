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
