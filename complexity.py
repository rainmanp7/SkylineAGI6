# complexity.py updated 12/10/2024

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Callable
import logging
from knowledge_base import TieredKnowledgeBase
from assimilation_memory_module import AssimilationMemoryModule

@dataclass
class ModelConfig:
    """
    Configuration for a machine learning model.
    """
    model_class: Any
    default_params: Dict[str, Any]
    complexity_level: str
    suggested_iterations: int
    suggested_metric: Callable
    quality_score: float


class EnhancedModelSelector:
    """
    Enhanced model selector based on the 9-tier complexity system.
    """
    def __init__(self, knowledge_base: TieredKnowledgeBase, assimilation_module: AssimilationMemoryModule):
        """
        Initialize the enhanced model selector.
        """
        self.knowledge_base = knowledge_base
        self.assimilation_module = assimilation_module
        self.complexity_tiers = self._validate_tier_ranges({
            # Example tiers
            'easy': (1000, 1200, mean_squared_error),
            'imp': (1201, 1400, mean_squared_error),
            'norm': (1401, 1600, mean_absolute_error),
            'ods': (1601, 1800, mean_absolute_error),
            'hard': (1801, 2000, mean_absolute_error),
            'para': (2001, 2200, r2_score),
            'vice': (2201, 2400, r2_score),
            'zeta': (2401, 2600, r2_score),
            'tetr': (2601, 2800, r2_score),
            # Add remaining tiers as necessary
        })
        self.model_configs = self._configure_models()

    def _validate_tier_ranges(self, tier_ranges: Dict[str, Tuple[int, int, Callable]]) -> Dict[str, Tuple[int, int, Callable]]:
        """
        Validate the tier ranges to ensure non-overlapping complexity ranges.
        """
        sorted_tiers = sorted(tier_ranges.items(), key=lambda x: x[1][0])
        for i in range(len(sorted_tiers) - 1):
            if sorted_tiers[i][1][1] >= sorted_tiers[i + 1][1][0]:
                raise ValueError("Overlapping tier ranges detected")
        return tier_ranges

    def _configure_models(self) -> Dict[str, ModelConfig]:
        """
        Configure the models for each complexity range.
        """
        return {
            'easy': ModelConfig(LinearRegression, {}, 'easy', 100, mean_squared_error, 0.8),
            'imp': ModelConfig(Ridge, {'alpha': 1.0}, 'imp', 200, mean_squared_error, 0.85),
            'norm': ModelConfig(Lasso, {'alpha': 1.0}, 'norm', 300, mean_absolute_error, 0.9),
            'ods': ModelConfig(RandomForestRegressor, {'n_estimators': 50}, 'ods', 400, mean_absolute_error, 0.92),
            'hard': ModelConfig(RandomForestRegressor, {'n_estimators': 100}, 'hard', 500, mean_absolute_error, 0.95),
            'para': ModelConfig(GradientBoostingRegressor, {'n_estimators': 100}, 'para', 600, r2_score, 0.98),
            # Add remaining model configurations
        }

    def _get_tier(self, complexity_factor: float) -> str:
        """
        Determine the tier for a given complexity factor.
        """
        for tier, (min_comp, max_comp, _) in self.complexity_tiers.items():
            if min_comp <= complexity_factor <= max_comp:
                return tier
        return 'easy'  # Default to 'easy'

    def choose_model_and_config(
        self, complexity_factor: float, custom_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Callable, int]:
        """
        Enhanced model selection based on the 9-tier complexity system.
        """
        try:
            complexity_factor = max(1000, min(6600, complexity_factor))  # Clamp value
            tier = self._get_tier(complexity_factor)
            config = self.model_configs[tier]
            params = config.default_params.copy()
            if custom_params:
                params.update(custom_params)
            model = config.model_class(**params)
            return model, config.suggested_metric, config.suggested_iterations
        except Exception as e:
            logging.error(f"Error in enhanced model selection: {str(e)}", exc_info=True)
            return LinearRegression(), mean_squared_error, 100  # Fallback

    def get_tier_details(self, tier: str) -> Dict[str, Any]:
        """
        Get details for a specific tier.
        """
        try:
            if tier in self.model_configs:
                config = self.model_configs[tier]
                min_comp, max_comp, metric = self.complexity_tiers[tier]
                return {
                    'complexity_range': (min_comp, max_comp),
                    'model_class': config.model_class.__name__,
                    'default_params': config.default_params,
                    'iterations': config.suggested_iterations,
                    'metric': metric.__name__,
                }
            else:
                raise ValueError("Tier not found")
        except Exception as e:
            logging.error(f"Error in getting tier details: {str(e)}", exc_info=True)
            return {}


# Example usage:
# knowledge_base = TieredKnowledgeBase()
# assimilation_module = AssimilationMemoryModule()
# model_selector = EnhancedModelSelector(knowledge_base, assimilation_module)
# model, metric, iterations = model_selector.choose_model_and_config(1500)
