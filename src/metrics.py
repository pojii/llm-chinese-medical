"""
Evaluation metrics for medicine prediction.
Calculates precision, recall, and F1 scores.
"""
from typing import List, Set, Dict, Tuple
import re


class MedicineEvaluationMetrics:
    """
    Calculate evaluation metrics for medicine predictions.
    Supports both exact and partial matching for Chinese medicine names.
    """

    def __init__(self):
        """Initialize metrics calculator."""
        self.reset()

    def reset(self):
        """Reset all accumulated statistics."""
        self.total_tp = 0  # True Positives
        self.total_fp = 0  # False Positives
        self.total_fn = 0  # False Negatives
        self.total_tn = 0  # True Negatives (not typically used)
        self.sample_scores = []

    @staticmethod
    def normalize_medicine_name(name: str) -> str:
        """
        Normalize medicine name for comparison.

        Args:
            name: Medicine name

        Returns:
            Normalized name
        """
        # Remove whitespace
        name = name.strip()
        # Remove common suffixes/prefixes for better matching
        # e.g., "片", "胶囊", "颗粒", "丸", "液" etc.
        # Keep them for now to maintain specificity
        return name

    @staticmethod
    def extract_medicines_from_text(text: str) -> Set[str]:
        """
        Extract medicine names from prediction text.

        Args:
            text: Prediction text that may contain multiple medicines

        Returns:
            Set of medicine names
        """
        if not text:
            return set()

        # Remove parenthetical content and other annotations
        import re
        text = re.sub(r'\([^)]*\)', '', text)  # Remove (...)
        text = re.sub(r'（[^）]*）', '', text)  # Remove （...）
        text = re.sub(r'\[[^\]]*\]', '', text)  # Remove [...]

        # Common delimiters for Chinese text
        delimiters = ['、', '，', ',', '；', ';', '和', '或', '以及', '\n']

        # Split by delimiters
        medicines = [text]
        for delimiter in delimiters:
            new_medicines = []
            for med in medicines:
                new_medicines.extend(med.split(delimiter))
            medicines = new_medicines

        # Clean and normalize
        medicines = [
            MedicineEvaluationMetrics.normalize_medicine_name(m)
            for m in medicines
        ]

        # Filter out empty strings and common phrases
        exclude_phrases = {
            '基于LLM直接推荐', '基于知识图谱推荐', '示例', '推荐',
            '建议', '可以', '使用', '服用', '等', ''
        }
        medicines = [
            m for m in medicines
            if m and m not in exclude_phrases and len(m) > 1
        ]

        return set(medicines)

    @staticmethod
    def calculate_similarity(name1: str, name2: str) -> float:
        """
        Calculate similarity between two medicine names.

        Args:
            name1: First medicine name
            name2: Second medicine name

        Returns:
            Similarity score (0.0 to 1.0)
        """
        if name1 == name2:
            return 1.0

        # Check if one contains the other
        if name1 in name2 or name2 in name1:
            return 0.8

        # Calculate character-level overlap
        set1 = set(name1)
        set2 = set(name2)

        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union

    def match_medicines(
        self,
        predicted: Set[str],
        ground_truth: Set[str],
        similarity_threshold: float = 0.6
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        Match predicted medicines with ground truth using similarity.

        Args:
            predicted: Set of predicted medicine names
            ground_truth: Set of ground truth medicine names
            similarity_threshold: Minimum similarity for a match

        Returns:
            Tuple of (true_positives, false_positives, false_negatives)
        """
        tp = set()  # Correctly predicted
        fp = set()  # Incorrectly predicted
        fn = set()  # Missed predictions

        matched_gt = set()

        # Try to match each prediction
        for pred in predicted:
            best_match = None
            best_score = 0.0

            for gt in ground_truth:
                if gt in matched_gt:
                    continue

                score = self.calculate_similarity(pred, gt)
                if score > best_score:
                    best_score = score
                    best_match = gt

            if best_score >= similarity_threshold:
                tp.add(pred)
                matched_gt.add(best_match)
            else:
                fp.add(pred)

        # Find missed ground truths
        fn = ground_truth - matched_gt

        return tp, fp, fn

    def calculate_metrics(
        self,
        predicted: Set[str],
        ground_truth: Set[str],
        similarity_threshold: float = 0.6
    ) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score.

        Args:
            predicted: Set of predicted medicine names
            ground_truth: Set of ground truth medicine names
            similarity_threshold: Minimum similarity for a match

        Returns:
            Dictionary with precision, recall, f1, tp, fp, fn counts
        """
        tp, fp, fn = self.match_medicines(predicted, ground_truth, similarity_threshold)

        tp_count = len(tp)
        fp_count = len(fp)
        fn_count = len(fn)

        # Calculate metrics
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp_count,
            'fp': fp_count,
            'fn': fn_count,
            'tp_items': list(tp),
            'fp_items': list(fp),
            'fn_items': list(fn)
        }

    def update(
        self,
        predicted: Set[str],
        ground_truth: Set[str],
        similarity_threshold: float = 0.6
    ):
        """
        Update accumulated statistics with a new sample.

        Args:
            predicted: Set of predicted medicine names
            ground_truth: Set of ground truth medicine names
            similarity_threshold: Minimum similarity for a match
        """
        metrics = self.calculate_metrics(predicted, ground_truth, similarity_threshold)

        self.total_tp += metrics['tp']
        self.total_fp += metrics['fp']
        self.total_fn += metrics['fn']

        self.sample_scores.append(metrics)

    def get_aggregate_metrics(self) -> Dict[str, float]:
        """
        Get aggregate metrics across all samples.

        Returns:
            Dictionary with aggregate precision, recall, f1
        """
        # Micro-averaged metrics (aggregate counts then calculate)
        precision = self.total_tp / (self.total_tp + self.total_fp) if (self.total_tp + self.total_fp) > 0 else 0.0
        recall = self.total_tp / (self.total_tp + self.total_fn) if (self.total_tp + self.total_fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Macro-averaged metrics (average individual scores)
        if self.sample_scores:
            macro_precision = sum(s['precision'] for s in self.sample_scores) / len(self.sample_scores)
            macro_recall = sum(s['recall'] for s in self.sample_scores) / len(self.sample_scores)
            macro_f1 = sum(s['f1'] for s in self.sample_scores) / len(self.sample_scores)
        else:
            macro_precision = macro_recall = macro_f1 = 0.0

        return {
            'micro_precision': precision,
            'micro_recall': recall,
            'micro_f1': f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'total_tp': self.total_tp,
            'total_fp': self.total_fp,
            'total_fn': self.total_fn,
            'num_samples': len(self.sample_scores)
        }

    def print_metrics_summary(self, metrics: Dict[str, float], prefix: str = ""):
        """
        Print a formatted summary of metrics.

        Args:
            metrics: Metrics dictionary
            prefix: Prefix for the output
        """
        print(f"\n{prefix}Evaluation Metrics:")
        print(f"{'─' * 60}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  True Positives (TP):  {metrics['tp']}")
        print(f"  False Positives (FP): {metrics['fp']}")
        print(f"  False Negatives (FN): {metrics['fn']}")

        if metrics.get('tp_items'):
            print(f"\nCorrect Predictions: {', '.join(metrics['tp_items'])}")
        if metrics.get('fp_items'):
            print(f"False Predictions: {', '.join(metrics['fp_items'])}")
        if metrics.get('fn_items'):
            print(f"Missed Predictions: {', '.join(metrics['fn_items'])}")

    def print_aggregate_summary(self):
        """Print aggregate metrics summary."""
        metrics = self.get_aggregate_metrics()

        print("\n" + "=" * 80)
        print("AGGREGATE EVALUATION METRICS")
        print("=" * 80)

        print(f"\n📊 Micro-Averaged Metrics (aggregate counts):")
        print(f"{'─' * 60}")
        print(f"  Precision: {metrics['micro_precision']:.4f}")
        print(f"  Recall:    {metrics['micro_recall']:.4f}")
        print(f"  F1 Score:  {metrics['micro_f1']:.4f}")

        print(f"\n📈 Macro-Averaged Metrics (average per sample):")
        print(f"{'─' * 60}")
        print(f"  Precision: {metrics['macro_precision']:.4f}")
        print(f"  Recall:    {metrics['macro_recall']:.4f}")
        print(f"  F1 Score:  {metrics['macro_f1']:.4f}")

        print(f"\n🔢 Overall Confusion Matrix:")
        print(f"{'─' * 60}")
        print(f"  True Positives (TP):  {metrics['total_tp']}")
        print(f"  False Positives (FP): {metrics['total_fp']}")
        print(f"  False Negatives (FN): {metrics['total_fn']}")
        print(f"  Total Samples:        {metrics['num_samples']}")


def compare_prediction_methods(
    predictions_without_kg: List[str],
    predictions_with_kg: List[str],
    ground_truths: List[List[str]]
) -> Dict[str, Dict]:
    """
    Compare two prediction methods against ground truth.

    Args:
        predictions_without_kg: List of predictions without KG
        predictions_with_kg: List of predictions with KG
        ground_truths: List of ground truth medicine lists

    Returns:
        Dictionary with metrics for both methods
    """
    metrics_without_kg = MedicineEvaluationMetrics()
    metrics_with_kg = MedicineEvaluationMetrics()

    for pred_no_kg, pred_kg, gt in zip(predictions_without_kg, predictions_with_kg, ground_truths):
        # Extract medicines from predictions
        pred_no_kg_set = MedicineEvaluationMetrics.extract_medicines_from_text(pred_no_kg)
        pred_kg_set = MedicineEvaluationMetrics.extract_medicines_from_text(pred_kg)
        gt_set = set(gt) if gt else set()

        # Update metrics
        metrics_without_kg.update(pred_no_kg_set, gt_set)
        metrics_with_kg.update(pred_kg_set, gt_set)

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
    print("Testing Medicine Evaluation Metrics")
    print("=" * 80)

    # Test case 1: Exact match
    print("\nTest 1: Exact Match")
    metrics = MedicineEvaluationMetrics()
    predicted = {"银翘解毒片", "连花清瘟胶囊"}
    ground_truth = {"银翘解毒片", "板蓝根颗粒"}

    result = metrics.calculate_metrics(predicted, ground_truth)
    metrics.print_metrics_summary(result, "Test 1 - ")

    # Test case 2: Partial match
    print("\n\nTest 2: Partial Match")
    metrics2 = MedicineEvaluationMetrics()
    predicted2 = {"银翘解毒片", "板蓝根"}
    ground_truth2 = {"银翘解毒片", "板蓝根颗粒"}

    result2 = metrics2.calculate_metrics(predicted2, ground_truth2)
    metrics2.print_metrics_summary(result2, "Test 2 - ")

    # Test case 3: Extract from text
    print("\n\nTest 3: Extract from Prediction Text")
    text = "银翘解毒片、连花清瘟胶囊 (基于知识图谱推荐)"
    extracted = MedicineEvaluationMetrics.extract_medicines_from_text(text)
    print(f"Input text: {text}")
    print(f"Extracted medicines: {extracted}")

    # Test case 4: Aggregate metrics
    print("\n\nTest 4: Aggregate Metrics")
    agg_metrics = MedicineEvaluationMetrics()

    samples = [
        ({"银翘解毒片"}, {"银翘解毒片", "板蓝根颗粒"}),
        ({"藿香正气丸", "保和丸"}, {"藿香正气丸"}),
        ({"安神补脑液"}, {"安神补脑液", "天王补心丹"}),
    ]

    for pred, gt in samples:
        agg_metrics.update(pred, gt)

    agg_metrics.print_aggregate_summary()
