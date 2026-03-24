"""
This module serves as the main entry point for executing the polynomial regression with cross-validation example. It generates synthetic data, evaluates polynomial degrees, and visualizes the results.

Usage example::

    python -m src.pkg.main
"""

import numpy as np
from .polyCV import PolynomialCV

if __name__ == "__main__":

    # Generate synthetic data
    n = 100
    X = np.linspace(-10, 10, n).reshape(-1, 1)
    f = lambda x: 2 * x**3 + x**2 + 4 * x
    seed = 39
    np.random.seed(seed)
    y = (f(X) + np.random.randn(100, 1) * 200).ravel()

    # Perform polynomial regression with cross-validation, evaluate degrees, and plot results
    poly_cv = PolynomialCV(X, y, degrees=range(1, 10), cv=5, random_seed=39)
    poly_cv.evaluate_degrees()
    poly_cv.summary()
    poly_cv.refit_best()
    poly_cv.plot(true_func=f)