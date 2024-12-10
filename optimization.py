# Beginning of optimization.py
# Last updated: Nov 23, 2024
# Purpose: Handles hyperparameter optimization using Bayesian methods and dynamic adjustment of the search space.

import os
import json
import warnings
import numpy as np
from typing import Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import logging
from models import create_model  # Ensure this is implemented and matches your architecture.
from utils import load_data, compute_complexity_factor  # Ensure these utilities exist.

# Suppress specific warnings from scikit-learn
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class HyperparameterOptimization:
    """
    Class for handling hyperparameter optimization tasks with dynamic search space adjustment.
    """

    def __init__(self, config_file: str = 'config.json', max_iterations=10, tolerance=0.05):
        """
        Initialize the optimization class and load configuration from the specified file.
        """
        self.cache = {}
        self.config = self._load_config(config_file)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _load_config(self, config_file: str) -> Dict:
        """
        Load configuration settings from a JSON file.
        """
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file '{config_file}' not found.")
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON in '{config_file}'.")

    def adjust_search_space(self, current_space: Dict, performance: float, threshold: float = 0.1) -> Dict:
        """
        Adjusts the hyperparameter search space dynamically based on model performance.
        """
        adjusted_space = current_space.copy()
        if performance > threshold:
            adjusted_space['learning_rate'] = Real(current_space['learning_rate'].low * 0.1,
                                                   current_space['learning_rate'].high * 10,
                                                   prior='log-uniform')
        else:
            adjusted_space['learning_rate'] = Real(current_space['learning_rate'].low,
                                                   current_space['learning_rate'].high * 0.1,
                                                   prior='log-uniform')
        return adjusted_space

    def parallel_bayesian_optimization(self, X_train, y_train, X_val, y_val, hyperparameters):
        """
        Parallel Bayesian optimization to determine optimal hyperparameters.
        """
        best_result = None
        previous_score = float('-inf')
        iterations = 0
        updated_hyperparameters = hyperparameters.copy()

        with ProcessPoolExecutor(max_workers=2) as executor:
            while iterations < self.max_iterations:
                futures = [
                    executor.submit(
                        self._execute_bayesian_optimization,
                        X_train, y_train, X_val, y_val,
                        self._perturb_hyperparameters(updated_hyperparameters)
                    ) for _ in range(2)
                ]

                current_results = []
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            current_results.append(result)
                    except Exception as e:
                        self.logger.error(f"Optimization iteration failed: {e}")

                if not current_results:
                    self.logger.warning("No valid results found in this iteration. Stopping.")
                    break

                best_current_result = max(current_results, key=lambda x: x[1])
                improvement = abs(best_current_result[1] - previous_score)

                if improvement < self.tolerance:
                    self.logger.info(f"Early stopping at iteration {iterations + 1}")
                    break

                previous_score = best_current_result[1]
                best_result = best_current_result
                updated_hyperparameters = best_current_result[0]

                iterations += 1
                self.logger.info(f"Iteration {iterations}: Best Score = {best_result[1]}")

        if best_result is None:
            self.logger.warning("No valid results after all iterations.")
            return {"status": "No valid results", "hyperparameters": updated_hyperparameters}

    def _execute_bayesian_optimization(self, X_train, y_train, X_val, y_val, hyperparameters):
        """
        Internal function for Bayesian optimization.
        """
        kernel_constant = hyperparameters.get("kernel_constant", 1.0)
        kernel_length_scale = max(hyperparameters.get("kernel_length_scale", 1.0), 1e-5)

        kernel = C(kernel_constant) * RBF(kernel_length_scale)
        gp_model = GaussianProcessRegressor(kernel=kernel)

        try:
            gp_model.fit(X_train, y_train)
            score = gp_model.score(X_val, y_val)
            quality_score = self._compute_quality_score(gp_model, X_val, y_val)

            return (
                {"kernel_constant": kernel_constant, "kernel_length_scale": kernel_length_scale},
                score,
                quality_score
            )
        except Exception as e:
            self.logger.error(f"Optimization error: {e}")
            return None

    def _perturb_hyperparameters(self, hyperparameters):
        """
        Perturb the hyperparameters for exploration.
        """
        perturbed = hyperparameters.copy()
        perturbed['kernel_constant'] *= np.random.uniform(0.9, 1.1)
        perturbed['kernel_length_scale'] *= np.random.uniform(0.9, 1.1)
        return perturbed

    def _compute_quality_score(self, model, X_val, y_val):
        """
        Compute quality score based on model predictions.
        """
        try:
            score = model.score(X_val, y_val)
            predictions = model.predict(X_val)
            mse = np.mean((predictions - y_val) ** 2)
            quality_score = 0.7 * score - 0.3 * mse
            return quality_score
        except Exception as e:
            self.logger.error(f"Quality score computation error: {e}")
            return -np.inf

    def perform_optimization(self, X_train, y_train, X_val, y_val):
        """
        Manages the overall optimization process, including adjusting the search space.
        """
        # Ensure initial model setup
        model = create_model(self.config['model_type'], self.config['model_params'])

        # Initial Bayesian optimization
        best_params, best_score, best_quality_score = self.parallel_bayesian_optimization(
            X_train, y_train, X_val, y_val, self.config['num_optimization_steps']
        )

        # Adjust search space based on results
        adjusted_search_space = self.adjust_search_space(param_space, best_score)

        # Re-run optimization with adjusted space
        best_params, best_score, best_quality_score = self.parallel_bayesian_optimization(
            X_train, y_train, X_val, y_val, self.config['num_optimization_steps']
        )

        # Final model creation and saving
        best_model = create_model(self.config['model_type'], best_params)
        model_save_path = os.path.join(self.config['model_save_dir'], 'best_model.pkl')
        best_model.save(model_save_path)

        return best_params, best_score, best_quality_score


# Example usage
def main():
    np.random.seed(42)
    X_train = np.random.rand(100, 5)
    y_train = np.random.rand(100)
    X_val = np.random.rand(20, 5)
    y_val = np.random.rand(20)

    hyperparameters = {
        "kernel_constant": 1.0,
        "kernel_length_scale": 1.0
    }

    optimizer = HyperparameterOptimization(max_iterations=10, tolerance=0.01)
    best_result = optimizer.perform_optimization(X_train, y_train, X_val, y_val)

    print("Best Result:", best_result)


if __name__ == "__main__":
    main()
