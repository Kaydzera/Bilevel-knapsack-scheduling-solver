"""Logging system for the bilevel optimization solver.

This module provides structured logging for tracking algorithm performance,
including runtime metrics, node statistics, bound quality, and debugging info.
"""

import logging
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class BnBLogger:
    """Logger for branch-and-bound algorithm with performance metrics.
    
    Tracks:
    - Standard log messages (debug, info, warning, error)
    - Performance metrics (nodes explored, pruned, runtime)
    - Bound quality and progression
    - Instance characteristics
    """
    
    def __init__(self, log_dir: str = "logs", instance_name: str = "default"):
        """Initialize the logger.
        
        Args:
            log_dir: Directory for log files
            instance_name: Name of the problem instance being solved
        """
        self.instance_name = instance_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{instance_name}_{self.timestamp}"
        
        # Initialize metrics dictionary
        self.metrics = {
            "instance_name": instance_name,
            "timestamp": self.timestamp,
            "start_time": None,
            "end_time": None,
            "total_runtime": None,
            "nodes_explored": 0,
            "nodes_pruned": 0,
            "nodes_evaluated": 0,  # leaf nodes where scheduling was solved
            "best_bound_updates": [],  # track incumbent improvements
            "bound_computations": [],  # track bound quality over time
            "pruning_reasons": {},  # categorize why nodes were pruned
        }
        
        # Setup file logging
        self._setup_file_logger()
        
        # Setup metrics logger (for structured data)
        self._setup_metrics_logger()
        
        self.logger.info(f"Initialized logger for instance: {instance_name}")
        self.logger.info(f"Run ID: {self.run_id}")
    
    def _setup_file_logger(self):
        """Setup standard file logger for text messages."""
        log_file = self.log_dir / f"{self.run_id}.log"
        
        # Create logger
        self.logger = logging.getLogger(f"bnb_{self.run_id}")
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # File handler with detailed format
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler with simpler format (optional)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_metrics_logger(self):
        """Setup metrics file for structured performance data."""
        self.metrics_file = self.log_dir / f"{self.run_id}_metrics.json"
    
    def start_run(self, problem_data: Optional[Dict[str, Any]] = None):
        """Mark the start of a solver run.
        
        Args:
            problem_data: Dictionary with problem characteristics
                         (n_items, budget, machines, etc.)
        """
        self.metrics["start_time"] = time.time()
        if problem_data:
            self.metrics["problem_data"] = problem_data
        self.logger.info("=" * 60)
        self.logger.info("Starting branch-and-bound solver")
        if problem_data:
            self.logger.info(f"Problem: {problem_data}")
        self.logger.info("=" * 60)
    
    def end_run(self, final_result: Optional[Dict[str, Any]] = None):
        """Mark the end of a solver run and save metrics.
        
        Args:
            final_result: Dictionary with final solution info
        """
        self.metrics["end_time"] = time.time()
        self.metrics["total_runtime"] = self.metrics["end_time"] - self.metrics["start_time"]
        
        if final_result:
            self.metrics["final_result"] = final_result
        
        # Save metrics to JSON
        self._save_metrics()
        
        self.logger.info("=" * 60)
        self.logger.info("Branch-and-bound completed")
        self.logger.info(f"Total runtime: {self.metrics['total_runtime']:.3f} seconds")
        self.logger.info(f"Nodes explored: {self.metrics['nodes_explored']}")
        self.logger.info(f"Nodes pruned: {self.metrics['nodes_pruned']}")
        self.logger.info(f"Nodes evaluated: {self.metrics['nodes_evaluated']}")
        if self.metrics['nodes_explored'] > 0:
            prune_rate = 100 * self.metrics['nodes_pruned'] / self.metrics['nodes_explored']
            self.logger.info(f"Pruning rate: {prune_rate:.2f}%")
        self.logger.info("=" * 60)
    
    def log_node_visit(self, node_info: Dict[str, Any]):
        """Log visiting a node in the search tree.
        
        Args:
            node_info: Dict with node details (depth, occurrences, budget, etc.)
        """
        self.metrics["nodes_explored"] += 1
        self.logger.debug(f"Node {self.metrics['nodes_explored']}: {node_info}")
    
    def log_node_pruned(self, reason: str, node_info: Optional[Dict[str, Any]] = None):
        """Log pruning a node.
        
        Args:
            reason: Why the node was pruned (e.g., "bound", "infeasible")
            node_info: Optional dict with node details
        """
        self.metrics["nodes_pruned"] += 1
        
        # Track pruning reasons
        if reason not in self.metrics["pruning_reasons"]:
            self.metrics["pruning_reasons"][reason] = 0
        self.metrics["pruning_reasons"][reason] += 1
        
        if node_info:
            self.logger.debug(f"Pruned ({reason}): {node_info}")
        else:
            self.logger.debug(f"Node pruned: {reason}")
    
    def log_node_evaluated(self, makespan: float, node_info: Optional[Dict[str, Any]] = None):
        """Log evaluating a leaf node (solving scheduling problem).
        
        Args:
            makespan: Makespan obtained from scheduling
            node_info: Optional dict with node details
        """
        self.metrics["nodes_evaluated"] += 1
        self.logger.debug(f"Evaluated leaf node: makespan={makespan}")
        if node_info:
            self.logger.debug(f"  Node: {node_info}")
    
    def log_incumbent_update(self, new_incumbent: float, selection: list, 
                            node_count: Optional[int] = None):
        """Log finding a new best solution.
        
        Args:
            new_incumbent: New best makespan
            selection: Job selection that achieved this makespan
            node_count: Number of nodes explored when found
        """
        update_info = {
            "incumbent": new_incumbent,
            "selection": selection,
            "node_count": node_count or self.metrics["nodes_explored"],
            "timestamp": time.time() - self.metrics["start_time"]
        }
        self.metrics["best_bound_updates"].append(update_info)
        
        self.logger.info("=" * 50)
        self.logger.info(f"NEW INCUMBENT: {new_incumbent}")
        self.logger.info(f"Selection: {selection}")
        self.logger.info(f"Found at node: {node_count or self.metrics['nodes_explored']}")
        self.logger.info("=" * 50)
    
    def log_bound_computation(self, bound_value: float, bound_type: str,
                             node_depth: int, computation_time: Optional[float] = None):
        """Log computing an upper bound.
        
        Args:
            bound_value: The bound value computed
            bound_type: Type of bound (e.g., "exact_ip", "fractional", "dynamic_prog")
            node_depth: Depth in the tree where bound was computed
            computation_time: Time taken to compute bound (seconds)
        """
        bound_info = {
            "value": bound_value,
            "type": bound_type,
            "depth": node_depth,
            "time": computation_time,
            "node_count": self.metrics["nodes_explored"]
        }
        self.metrics["bound_computations"].append(bound_info)
        
        if computation_time:
            self.logger.debug(f"Bound computed: {bound_value:.2f} ({bound_type}) "
                            f"at depth {node_depth} in {computation_time:.4f}s")
        else:
            self.logger.debug(f"Bound computed: {bound_value:.2f} ({bound_type}) "
                            f"at depth {node_depth}")
    
    def log_branching_decision(self, decision: str, node_info: Dict[str, Any]):
        """Log a branching decision (for future branching rule experiments).
        
        Args:
            decision: Description of branching decision
            node_info: Node information
        """
        self.logger.debug(f"Branching decision: {decision}")
        self.logger.debug(f"  Node: {node_info}")
    
    def _save_metrics(self):
        """Save metrics dictionary to JSON file."""
        # Convert to JSON-serializable format
        metrics_copy = self.metrics.copy()
        
        # Remove non-serializable objects from final_result
        if 'final_result' in metrics_copy and 'best_schedule' in metrics_copy['final_result']:
            schedule = metrics_copy['final_result']['best_schedule']
            if isinstance(schedule, dict) and 'model' in schedule:
                # Remove Gurobi model object
                schedule_clean = schedule.copy()
                del schedule_clean['model']
                metrics_copy['final_result']['best_schedule'] = schedule_clean
        
        # Save to file
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics_copy, f, indent=2)
        
        self.logger.info(f"Metrics saved to: {self.metrics_file}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics dictionary.
        
        Returns:
            Dictionary with all tracked metrics
        """
        return self.metrics.copy()
    
    def debug(self, msg: str):
        """Log debug message."""
        self.logger.debug(msg)
    
    def info(self, msg: str):
        """Log info message."""
        self.logger.info(msg)
    
    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)
    
    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)


def create_logger(instance_name: str = "default", log_dir: str = "logs") -> BnBLogger:
    """Factory function to create a BnBLogger.
    
    Args:
        instance_name: Name of the problem instance
        log_dir: Directory for log files
        
    Returns:
        Configured BnBLogger instance
    """
    return BnBLogger(log_dir=log_dir, instance_name=instance_name)
