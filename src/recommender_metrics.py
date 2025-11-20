"""
Recommender System Evaluation Metrics.
Implements Precision@K, Recall@K, MAP@K, MRR@K for medicine recommendation.
"""
from typing import List, Set, Dict, Tuple
import numpy as np


class RecommenderMetrics:
    """
    Evaluation metrics for recommender systems.

    Metrics:
    - Precision@K: Proportion of relevant items in top K recommendations
    - Recall@K: Proportion of relevant items retrieved in top K
    - MAP@K (Mean Average Precision): Average of precision at each relevant item
    - MRR@K (Mean Reciprocal Rank): Reciprocal of rank of first relevant item
    """

    def __init__(self):
        """Initialize metrics accumulator."""
        self.reset()

    def reset(self):
        """Reset all accumulated statistics."""
        self.all_precisions = {k: [] for k in [5, 10, 20, 50]}
        self.all_recalls = {k: [] for k in [5, 10, 20, 50]}
        self.all_maps = {k: [] for k in [5, 10, 20, 50]}
        self.all_mrrs = {k: [] for k in [5, 10, 20, 50]}
        self.num_samples = 0

    @staticmethod
    def precision_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
        """
        Calculate Precision@K.

        Args:
            recommended: Ranked list of recommended items
            relevant: Set of ground truth relevant items
            k: Top K items to consider

        Returns:
            Precision@K score (0.0 to 1.0)
        """
        if k <= 0 or not recommended:
            return 0.0

        top_k = recommended[:k]
        relevant_in_top_k = sum(1 for item in top_k if item in relevant)

        return relevant_in_top_k / k

    @staticmethod
    def recall_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
        """
        Calculate Recall@K.

        Args:
            recommended: Ranked list of recommended items
            relevant: Set of ground truth relevant items
            k: Top K items to consider

        Returns:
            Recall@K score (0.0 to 1.0)
        """
        if not relevant or k <= 0 or not recommended:
            return 0.0

        top_k = recommended[:k]
        relevant_in_top_k = sum(1 for item in top_k if item in relevant)

        return relevant_in_top_k / len(relevant)

    @staticmethod
    def average_precision_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
        """
        Calculate Average Precision@K.

        AP@K = (1/min(K, |relevant|)) * sum(P@i * rel(i))
        where P@i is precision at position i, rel(i) is 1 if item at i is relevant

        Args:
            recommended: Ranked list of recommended items
            relevant: Set of ground truth relevant items
            k: Top K items to consider

        Returns:
            Average Precision@K score (0.0 to 1.0)
        """
        if not relevant or k <= 0 or not recommended:
            return 0.0

        top_k = recommended[:k]
        score = 0.0
        num_hits = 0.0

        for i, item in enumerate(top_k):
            if item in relevant:
                num_hits += 1
                score += num_hits / (i + 1)

        if num_hits == 0:
            return 0.0

        return score / min(len(relevant), k)

    @staticmethod
    def reciprocal_rank_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
        """
        Calculate Reciprocal Rank@K.

        RR@K = 1 / rank of first relevant item (within top K)

        Args:
            recommended: Ranked list of recommended items
            relevant: Set of ground truth relevant items
            k: Top K items to consider

        Returns:
            Reciprocal Rank@K score (0.0 to 1.0)
        """
        if not relevant or k <= 0 or not recommended:
            return 0.0

        top_k = recommended[:k]

        for i, item in enumerate(top_k):
            if item in relevant:
                return 1.0 / (i + 1)

        return 0.0

    def evaluate_single(
        self,
        recommended: List[str],
        relevant: Set[str]
    ) -> Dict[int, Dict[str, float]]:
        """
        Evaluate a single recommendation against ground truth.

        Args:
            recommended: Ranked list of recommended items
            relevant: Set of ground truth relevant items

        Returns:
            Dictionary with metrics for each K value
        """
        results = {}

        for k in [5, 10, 20, 50]:
            results[k] = {
                'precision': self.precision_at_k(recommended, relevant, k),
                'recall': self.recall_at_k(recommended, relevant, k),
                'map': self.average_precision_at_k(recommended, relevant, k),
                'mrr': self.reciprocal_rank_at_k(recommended, relevant, k)
            }

        return results

    def update(self, recommended: List[str], relevant: Set[str]):
        """
        Update accumulated metrics with a new sample.

        Args:
            recommended: Ranked list of recommended items
            relevant: Set of ground truth relevant items
        """
        metrics = self.evaluate_single(recommended, relevant)

        for k in [5, 10, 20, 50]:
            self.all_precisions[k].append(metrics[k]['precision'])
            self.all_recalls[k].append(metrics[k]['recall'])
            self.all_maps[k].append(metrics[k]['map'])
            self.all_mrrs[k].append(metrics[k]['mrr'])

        self.num_samples += 1

    def get_aggregate_metrics(self) -> Dict[str, Dict[int, float]]:
        """
        Get aggregate metrics across all samples.

        Returns:
            Dictionary with mean metrics for each K value
        """
        results = {
            'precision': {},
            'recall': {},
            'map': {},
            'mrr': {},
            'num_samples': self.num_samples
        }

        for k in [5, 10, 20, 50]:
            results['precision'][k] = np.mean(self.all_precisions[k]) if self.all_precisions[k] else 0.0
            results['recall'][k] = np.mean(self.all_recalls[k]) if self.all_recalls[k] else 0.0
            results['map'][k] = np.mean(self.all_maps[k]) if self.all_maps[k] else 0.0
            results['mrr'][k] = np.mean(self.all_mrrs[k]) if self.all_mrrs[k] else 0.0

        return results

    def print_aggregate_summary(self, prefix: str = ""):
        """
        Print aggregate metrics summary.

        Args:
            prefix: Prefix for output (e.g., "With KG" or "Without KG")
        """
        metrics = self.get_aggregate_metrics()

        print(f"\n{'=' * 80}")
        print(f"{prefix} Recommender Metrics")
        print('=' * 80)
        print(f"Total samples: {metrics['num_samples']}\n")

        # Print table header
        print(f"{'Metric':<15} {'@5':<12} {'@10':<12} {'@20':<12} {'@50':<12}")
        print('─' * 80)

        # Print each metric
        for metric_name in ['precision', 'recall', 'map', 'mrr']:
            display_name = metric_name.upper() if metric_name in ['map', 'mrr'] else metric_name.capitalize()
            row = f"{display_name:<15}"
            for k in [5, 10, 20, 50]:
                row += f" {metrics[metric_name][k]:.4f}      "
            print(row)

        print('=' * 80)


def compare_recommenders(
    recommendations_without_kg: List[List[str]],
    recommendations_with_kg: List[List[str]],
    ground_truths: List[Set[str]]
) -> Dict[str, Dict]:
    """
    Compare two recommender systems.

    Args:
        recommendations_without_kg: List of ranked recommendation lists (without KG)
        recommendations_with_kg: List of ranked recommendation lists (with KG)
        ground_truths: List of ground truth relevant item sets

    Returns:
        Dictionary with metrics for both systems
    """
    metrics_without_kg = RecommenderMetrics()
    metrics_with_kg = RecommenderMetrics()

    for rec_no_kg, rec_kg, gt in zip(recommendations_without_kg, recommendations_with_kg, ground_truths):
        metrics_without_kg.update(rec_no_kg, gt)
        metrics_with_kg.update(rec_kg, gt)

    return {
        'without_kg': metrics_without_kg.get_aggregate_metrics(),
        'with_kg': metrics_with_kg.get_aggregate_metrics(),
        'metrics_objects': {
            'without_kg': metrics_without_kg,
            'with_kg': metrics_with_kg
        }
    }


if __name__ == "__main__":
    # Test the metrics
    print("Testing Recommender Evaluation Metrics")
    print("=" * 80)

    # Test case 1: Perfect recommendations
    print("\nTest 1: Perfect Recommendations")
    recommended = ["药A", "药B", "药C", "药D", "药E"]
    relevant = {"药A", "药B", "药C"}

    metrics = RecommenderMetrics()
    result = metrics.evaluate_single(recommended, relevant)

    print(f"Recommended: {recommended}")
    print(f"Relevant: {relevant}")
    for k in [5, 10]:
        print(f"\n@{k}:")
        print(f"  Precision: {result[k]['precision']:.4f}")
        print(f"  Recall: {result[k]['recall']:.4f}")
        print(f"  MAP: {result[k]['map']:.4f}")
        print(f"  MRR: {result[k]['mrr']:.4f}")

    # Test case 2: Partial match
    print("\n\nTest 2: Partial Match (1st relevant at rank 3)")
    recommended2 = ["药X", "药Y", "药A", "药B", "药Z"]
    relevant2 = {"药A", "药B", "药C"}

    result2 = metrics.evaluate_single(recommended2, relevant2)

    print(f"Recommended: {recommended2}")
    print(f"Relevant: {relevant2}")
    for k in [5, 10]:
        print(f"\n@{k}:")
        print(f"  Precision: {result2[k]['precision']:.4f}")
        print(f"  Recall: {result2[k]['recall']:.4f}")
        print(f"  MAP: {result2[k]['map']:.4f}")
        print(f"  MRR: {result2[k]['mrr']:.4f}")

    # Test case 3: Aggregate metrics
    print("\n\nTest 3: Aggregate Metrics")
    agg_metrics = RecommenderMetrics()

    test_cases = [
        (["药A", "药B", "药C", "药D", "药E"], {"药A", "药B"}),
        (["药X", "药Y", "药A", "药B", "药C"], {"药A", "药B", "药C"}),
        (["药A", "药X", "药Y", "药Z", "药W"], {"药A"}),
    ]

    for rec, rel in test_cases:
        agg_metrics.update(rec, rel)

    agg_metrics.print_aggregate_summary("Test")
