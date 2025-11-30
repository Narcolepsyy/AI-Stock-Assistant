"""
ML-based tool selector for intelligent tool selection.

This module provides the high-level interface for using ML to select tools
based on user queries. It handles:
- Loading the trained model
- Embedding queries
- Predicting relevant tools
- Confidence filtering
- Fallback to rule-based selection
- Performance tracking and statistics
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import threading

from app.core.config import (
    ML_MODEL_PATH,
    ML_CONFIDENCE_THRESHOLD,
    ML_MAX_TOOLS,
    ML_TOOL_SELECTION_ENABLED
)
from app.services.ml.embedder import QueryEmbedder
from app.services.ml.classifier import ToolClassifier

logger = logging.getLogger(__name__)


class MLToolSelector:
    """
    High-level ML tool selector with caching, fallback, and monitoring.
    
    Features:
    - Lazy model loading (only loads when first needed)
    - Query embedding with caching
    - Confidence-based filtering
    - Fallback to rule-based when confidence is low
    - Statistics tracking for monitoring
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for efficient resource usage."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the ML tool selector (lazy loading)."""
        if self._initialized:
            return
            
        self.embedder: Optional[QueryEmbedder] = None
        self.classifier: Optional[ToolClassifier] = None
        self.model_loaded = False
        self.model_path = ML_MODEL_PATH
        self.confidence_threshold = ML_CONFIDENCE_THRESHOLD
        self.max_tools = ML_MAX_TOOLS
        self._load_lock = threading.Lock()
        
        # Statistics tracking
        self.stats = {
            'total_predictions': 0,
            'fallback_count': 0,
            'ml_count': 0,  # Track ML usage
            'avg_confidence': 0.0,  # Running mean of top predicted probability
            'avg_prediction_time_ms': 0.0,
            'tools_predicted': defaultdict(int),
            'confidence_distribution': defaultdict(int),  # bins: 0-0.3, 0.3-0.5, 0.5-0.7, 0.7-0.9, 0.9-1.0
        }
        self.stats_lock = threading.Lock()
        
        self._initialized = True
        logger.info("MLToolSelector initialized (lazy loading enabled)")
    
    def _load_model(self) -> bool:
        """
        Load the trained model and embedder (lazy loading).
        
        Returns:
            True if successful, False otherwise
        """
        if self.model_loaded:
            return True
            
        try:
            with self._load_lock:
                if self.model_loaded:
                    return True

                logger.info(f"Loading ML model from {self.model_path}...")
                
                # Check if model file exists
                model_file = Path(self.model_path)
                if not model_file.exists():
                    logger.error(f"Model file not found: {self.model_path}")
                    return False
                
                # Initialize embedder
                self.embedder = QueryEmbedder()
                logger.info("Query embedder initialized")
                
                # Load classifier
                self.classifier = ToolClassifier.load(self.model_path)
                logger.info(f"Classifier loaded with {len(self.classifier.tool_names)} tools")
                
                self.model_loaded = True
                logger.info("✅ ML model loaded successfully!")
                return True
            
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}", exc_info=True)
            self.model_loaded = False
            return False
    
    def predict_tools(
        self,
        query: str,
        available_tools: Optional[List[str]] = None,
        return_probabilities: bool = False,
        use_hybrid: bool = True  # Deprecated parameter, kept for backward compatibility
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Predict relevant tools for a query using ML classifier.
        
        Args:
            query: User query text
            available_tools: Optional list of available tools to filter by
            return_probabilities: Whether to return tool probabilities
            use_hybrid: Deprecated, kept for backward compatibility
            
        Returns:
            Tuple of (selected_tools, probabilities_dict)
            - selected_tools: List of tool names above confidence threshold
            - probabilities_dict: Dict of tool -> probability (empty if not requested)
        """
        start_time = time.time()

        if not ML_TOOL_SELECTION_ENABLED:
            logger.info("ML tool selection disabled via config, returning empty tool list")
            prediction_time_ms = (time.time() - start_time) * 1000
            self._update_stats([], {}, prediction_time_ms, 0.0, method='ml_disabled')
            return [], {}
        
        # Lazy load model if needed
        if not self.model_loaded:
            if not self._load_model():
                logger.warning("ML model not loaded, returning empty tool list")
                prediction_time_ms = (time.time() - start_time) * 1000
                self._update_stats([], {}, prediction_time_ms, 0.0, method='ml_unavailable')
                return [], {}
        
        try:
            # Step 1: Try ML classifier first
            logger.debug(f"Embedding query: {query[:100]}...")
            query_embedding = self.embedder.embed(query)
            
            logger.debug("Predicting tool probabilities with ML...")
            tool_probs = self.classifier.predict_proba(query_embedding)
            if not tool_probs:
                logger.info("ML returned no tool probabilities, returning empty tool list")
                prediction_time_ms = (time.time() - start_time) * 1000
                self._update_stats([], {}, prediction_time_ms, 0.0, method='ml_no_candidates')
                return [], {}
            
            # Filter by available tools if specified
            if available_tools:
                tool_probs = {
                    tool: prob 
                    for tool, prob in tool_probs.items() 
                    if tool in available_tools
                }
                if not tool_probs:
                    logger.info("No available tools matched the classifier output, returning empty tool list")
                    prediction_time_ms = (time.time() - start_time) * 1000
                    self._update_stats([], {}, prediction_time_ms, 0.0, method='ml_no_candidates')
                    return [], {}
            
            # Derive confidence metrics
            top_confidence = max(tool_probs.values()) if tool_probs else 0.0
            LOW_CONFIDENCE_THRESHOLD = 0.5
            if top_confidence < LOW_CONFIDENCE_THRESHOLD:
                logger.info(
                    f"ML confidence too low ({top_confidence:.2f} < {LOW_CONFIDENCE_THRESHOLD}), "
                    "returning empty list to let model decide with all tools"
                )
                prediction_time_ms = (time.time() - start_time) * 1000
                self._update_stats([], {}, prediction_time_ms, top_confidence, method='ml_low_confidence')
                return [], {}

            # Filter by configured confidence threshold
            filtered_tools = {
                tool: prob 
                for tool, prob in tool_probs.items() 
                if prob >= self.confidence_threshold
            }
            if not filtered_tools:
                logger.info(
                    "No tools passed the configured confidence threshold, returning empty list"
                )
                prediction_time_ms = (time.time() - start_time) * 1000
                self._update_stats([], {}, prediction_time_ms, top_confidence, method='ml_below_threshold')
                return [], {}
            
            # Sort by probability and limit to max_tools
            sorted_tools = sorted(
                filtered_tools.items(),
                key=lambda x: x[1],
                reverse=True
            )[:self.max_tools]
            
            selected_tools = [tool for tool, _ in sorted_tools]
            probabilities = dict(sorted_tools)
            
            # Update statistics
            prediction_time_ms = (time.time() - start_time) * 1000
            self._update_stats(
                selected_tools, 
                probabilities, 
                prediction_time_ms, 
                top_confidence,
                method='ml'
            )
            
            probabilities_out = probabilities if return_probabilities else {}
            
            logger.info(
                f"ML predicted {len(selected_tools)} tools in {prediction_time_ms:.1f}ms "
                f"(top confidence: {top_confidence:.2f}): {selected_tools}"
            )
            
            return selected_tools, probabilities_out
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}", exc_info=True)
            prediction_time_ms = (time.time() - start_time) * 1000
            self._update_stats([], {}, prediction_time_ms, 0.0, method='ml_error')
            return [], {}
    
    def should_use_ml(self) -> bool:
        """
        Check if ML tool selection should be used.
        
        Returns:
            True if ML is enabled and model is loaded/loadable
        """
        if not ML_TOOL_SELECTION_ENABLED:
            return False
        
        if not self.model_loaded:
            return self._load_model()
        
        return True
    
    def _update_stats(
        self,
        selected_tools: List[str],
        probabilities: Dict[str, float],
        prediction_time_ms: float,
        confidence_metric: float,
        method: str = 'ml'
    ):
        """Update statistics tracking."""
        with self.stats_lock:
            n = self.stats['total_predictions']
            
            # Update running averages
            self.stats['avg_confidence'] = (
                (self.stats['avg_confidence'] * n + confidence_metric) / (n + 1)
            )
            self.stats['avg_prediction_time_ms'] = (
                (self.stats['avg_prediction_time_ms'] * n + prediction_time_ms) / (n + 1)
            )
            self.stats['total_predictions'] += 1
            
            # Track method usage
            if method == 'ml':
                self.stats['ml_count'] += 1
            else:
                self.stats['fallback_count'] += 1
            
            # Track tool usage
            for tool in selected_tools:
                self.stats['tools_predicted'][tool] += 1
            
            # Track confidence distribution
            if confidence_metric < 0.3:
                bucket = '0.0-0.3'
            elif confidence_metric < 0.5:
                bucket = '0.3-0.5'
            elif confidence_metric < 0.7:
                bucket = '0.5-0.7'
            elif confidence_metric < 0.9:
                bucket = '0.7-0.9'
            else:
                bucket = '0.9-1.0'
            self.stats['confidence_distribution'][bucket] += 1
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        with self.stats_lock:
            total = self.stats['total_predictions']
            return {
                'total_predictions': total,
                'ml_count': self.stats['ml_count'],
                'fallback_count': self.stats['fallback_count'],
                'ml_rate': self.stats['ml_count'] / total if total > 0 else 0.0,
                'fallback_rate': self.stats['fallback_count'] / total if total > 0 else 0.0,
                'avg_confidence': round(self.stats['avg_confidence'], 3),
                'avg_prediction_time_ms': round(self.stats['avg_prediction_time_ms'], 2),
                'tools_predicted': dict(self.stats['tools_predicted']),
                'confidence_distribution': dict(self.stats['confidence_distribution']),
                'model_loaded': self.model_loaded,
                'ml_enabled': ML_TOOL_SELECTION_ENABLED,
            }
    
    def record_fallback(self):
        """Record that fallback to rule-based was used."""
        with self.stats_lock:
            self.stats['fallback_count'] += 1
    
    def reset_stats(self):
        """Reset statistics (useful for testing/monitoring)."""
        with self.stats_lock:
            self.stats = {
                'total_predictions': 0,
                'fallback_count': 0,
                'ml_count': 0,
                'avg_confidence': 0.0,
                'avg_prediction_time_ms': 0.0,
                'tools_predicted': defaultdict(int),
                'confidence_distribution': defaultdict(int),
            }
        logger.info("Statistics reset")


# Global singleton instance
_ml_selector = MLToolSelector()


def get_ml_tool_selector() -> MLToolSelector:
    """
    Get the global ML tool selector instance.
    
    Returns:
        MLToolSelector singleton instance
    """
    return _ml_selector


def predict_tools_ml(
    query: str,
    available_tools: Optional[List[str]] = None,
    return_probabilities: bool = False
) -> Tuple[List[str], Dict[str, float]]:
    """
    Convenience function for ML tool prediction.
    
    Args:
        query: User query text
        available_tools: Optional list of available tools to filter by
        return_probabilities: Whether to return tool probabilities
        
    Returns:
        Tuple of (selected_tools, probabilities_dict)
    """
    selector = get_ml_tool_selector()
    return selector.predict_tools(query, available_tools, return_probabilities)


def get_ml_stats() -> Dict:
    """
    Get ML tool selector statistics.
    
    Returns:
        Dictionary with statistics
    """
    selector = get_ml_tool_selector()
    return selector.get_stats()
