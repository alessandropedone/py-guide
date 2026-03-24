import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate


class PolynomialCV:
    """
    .. admonition:: Description
       A class for performing polynomial regression with cross-validation.

    :param X: Input features (2D array).
    :param y: Target values (1D array).
    :param degrees: Iterable of polynomial degrees to evaluate (default: range(1, 10)).
    :param cv: Number of cross-validation folds (default: 5).
    :param random_seed: Optional random seed for reproducibility.
    """

    def __init__(self, X, y, degrees=range(1, 10), cv=5, random_seed=None) -> None:
        self.X = X
        self.y = y
        self.degrees = degrees
        self.cv = cv
        if random_seed is not None:
            np.random.seed(random_seed)
        self.results = {}
        self.best_deg = None
        self.best_model = None

    def evaluate_degrees(self) -> int:
        """Evaluate polynomial degrees using cross-validation and store results."""
        for deg in self.degrees:
            model = make_pipeline(PolynomialFeatures(degree=deg), LinearRegression())
            cv_res = cross_validate(
                model,
                self.X,
                self.y,
                cv=self.cv,
                scoring=["r2", "neg_root_mean_squared_error"],
                return_train_score=True,
            )
            self.results[deg] = {
                "val_r2": cv_res["test_r2"].mean(),
                "train_r2": cv_res["train_r2"].mean(),
                "val_rmse": -cv_res["test_neg_root_mean_squared_error"].mean(),
                "val_r2_std": cv_res["test_r2"].std(),
            }
        self.best_deg = max(self.results, key=lambda d: self.results[d]["val_r2"])
        return self.best_deg

    def refit_best(self) -> make_pipeline:
        """Refit the best model on the entire dataset."""
        self.best_model = make_pipeline(
            PolynomialFeatures(degree=self.best_deg), LinearRegression()
        )
        self.best_model.fit(self.X, self.y)
        return self.best_model

    def plot(self, true_func=None):
        """
        Plot the data, the best polynomial fit, and optionally the true function.

        :param true_func: A callable representing the true function to plot (optional).
        """
        y_pred = self.best_model.predict(self.X)
        plt.scatter(self.X, self.y, s=15, color="steelblue", label="Data")
        plt.plot(
            self.X,
            y_pred,
            color="darkorange",
            linestyle="--",
            label=f"Degree {self.best_deg} fit",
        )
        if true_func is not None:
            plt.plot(
                self.X,
                true_func(self.X),
                color="green",
                linestyle=":",
                label="True curve",
            )
        plt.title(f"Best model (degree={self.best_deg})")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def summary(self):
        """Print a summary of the cross-validation results for each polynomial degree."""
        print(f"{'Deg':<5} {'Val R²':>8} {'±':>10} {'Train R²':>10} {'Val RMSE':>10}")
        print("-" * 47)
        for deg, r in self.results.items():
            marker = " <-- best" if deg == self.best_deg else ""
            print(
                f"{deg:<5} {r['val_r2']:>8.2f} {r['val_r2_std']:>10.2f} {r['train_r2']:>10.2f} {r['val_rmse']:>10.2f}{marker}"
            )
