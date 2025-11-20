"""
Compare PTM Herb Recommender: Baseline vs RAG-Enhanced
Side-by-side comparison with same random seed for fair evaluation.
"""
import os
import time
import random
import numpy as np
from typing import Dict, List

# Import baseline (no RAG)
from ptm_herb_recommender import (
    PTMHerbDataset as BaselineDataset,
    PTMHerbPredictor as BaselinePredictor,
    HerbRecommenderMetrics as BaselineMetrics
)

# Import RAG-enhanced
from ptm_herb_recommender_with_rag import (
    PTMHerbDataset as RAGDataset,
    PTMHerbPredictorWithRAG as RAGPredictor,
    HerbRecommenderMetrics as RAGMetrics,
    HerbKnowledgeRAG
)


class PTMComparison:
    """Compare baseline and RAG-enhanced herb recommenders."""

    def __init__(
        self,
        prescriptions_path: str = "../data/PTM/data/prescriptions.txt",
        knowledge_path: str = "../data/herb-knowledge.csv",
        model: str = "deepseek-chat",
        random_seed: int = 42
    ):
        """
        Initialize comparison.

        Args:
            prescriptions_path: Path to PTM prescriptions
            knowledge_path: Path to herb knowledge CSV
            model: Model name for DeepSeek API
            random_seed: Random seed for reproducibility
        """
        self.prescriptions_path = prescriptions_path
        self.knowledge_path = knowledge_path
        self.model = model
        self.random_seed = random_seed

        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        print(f"🎲 Random seed set to: {random_seed}")

    def run_comparison(
        self,
        num_samples: int = 20,
        vocab_size: int = 200,
        max_recommendations: int = 30,
        rag_top_k: int = 10
    ):
        """
        Run comparison between baseline and RAG.

        Args:
            num_samples: Number of test samples
            vocab_size: Herb vocabulary size
            max_recommendations: Max herbs to recommend
            rag_top_k: RAG knowledge retrieval top-K
        """
        print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║              PTM Herb Recommender Comparison                               ║
║                   Baseline vs RAG-Enhanced                                 ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
        """)

        # Check API keys
        if not os.environ.get("DEEPSEEK_API_KEY"):
            print("❌ Error: DEEPSEEK_API_KEY not set")
            return

        if not os.environ.get("OPENAI_API_KEY"):
            print("❌ Error: OPENAI_API_KEY not set (required for RAG)")
            return

        print("Configuration:")
        print(f"  Data path: {self.prescriptions_path}")
        print(f"  Knowledge path: {self.knowledge_path}")
        print(f"  Model: {self.model}")
        print(f"  Random seed: {self.random_seed}")
        print(f"  Test samples: {num_samples}")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Max recommendations: {max_recommendations}")
        print(f"  RAG top-K: {rag_top_k}")

        # Load dataset (same for both)
        print("\n" + "=" * 80)
        print("Loading Dataset")
        print("=" * 80)
        dataset = BaselineDataset(self.prescriptions_path)

        # Get herb vocabulary
        herb_vocabulary = dataset.get_top_herbs(vocab_size)
        print(f"\nHerb vocabulary: {len(herb_vocabulary)} herbs")

        # Select random samples with fixed seed
        print(f"\n🎲 Selecting {num_samples} random samples (seed={self.random_seed})...")
        all_indices = list(range(len(dataset)))
        random.shuffle(all_indices)
        sample_indices = all_indices[:num_samples]
        print(f"Selected sample indices: {sample_indices[:10]}{'...' if num_samples > 10 else ''}")

        # Initialize baseline predictor
        print("\n" + "=" * 80)
        print("Initializing BASELINE Predictor (No RAG)")
        print("=" * 80)
        baseline_predictor = BaselinePredictor(
            herb_vocabulary=herb_vocabulary,
            model=self.model
        )
        baseline_metrics = BaselineMetrics()

        # Initialize RAG system and predictor
        print("\n" + "=" * 80)
        print("Initializing RAG System")
        print("=" * 80)
        rag_system = HerbKnowledgeRAG(self.knowledge_path)

        print("\n" + "=" * 80)
        print("Initializing RAG-ENHANCED Predictor")
        print("=" * 80)
        rag_predictor = RAGPredictor(
            herb_vocabulary=herb_vocabulary,
            rag_system=rag_system,
            model=self.model
        )
        rag_metrics = RAGMetrics()

        # Run evaluation on same samples
        print("\n" + "=" * 80)
        print("Running Side-by-Side Comparison")
        print("=" * 80)

        baseline_time = 0
        rag_time = 0

        for i, idx in enumerate(sample_indices, 1):
            print(f"\n{'=' * 80}")
            print(f"Sample {i}/{num_samples} (Index: {idx})")
            print('=' * 80)

            sample = dataset.get_sample(idx)
            symptoms = sample['symptoms']
            ground_truth = sample['ground_truth']

            print(f"\n📝 Symptoms: {symptoms[:80]}{'...' if len(symptoms) > 80 else ''}")
            print(f"✅ Ground Truth: {len(ground_truth)} herbs")

            # Baseline prediction
            print(f"\n🔵 BASELINE (No RAG)...")
            start = time.time()
            baseline_predictions = baseline_predictor.predict(symptoms, top_k=max_recommendations)
            baseline_time += time.time() - start
            print(f"   Top 5: {baseline_predictions[:5]}")
            print(f"   Time: {time.time() - start:.2f}s")

            # RAG prediction
            print(f"\n🟢 RAG-ENHANCED...")
            start = time.time()
            rag_predictions = rag_predictor.predict(symptoms, top_k=max_recommendations)
            rag_time += time.time() - start
            print(f"   Top 5: {rag_predictions[:5]}")
            print(f"   Time: {time.time() - start:.2f}s")

            # Update metrics
            baseline_metrics.update(baseline_predictions, ground_truth)
            rag_metrics.update(rag_predictions, ground_truth)

            # Show sample-level comparison
            baseline_sample_metrics = baseline_metrics.evaluate_single(baseline_predictions, ground_truth)
            rag_sample_metrics = rag_metrics.evaluate_single(rag_predictions, ground_truth)

            print(f"\n📊 Sample Metrics Comparison:")
            print(f"{'Metric':<10} {'Baseline':<12} {'RAG':<12} {'Δ':<10}")
            print('─' * 50)
            for k in [5, 10, 20]:
                b_p = baseline_sample_metrics[k]['precision']
                r_p = rag_sample_metrics[k]['precision']
                delta_p = r_p - b_p
                print(f"P@{k:<7} {b_p:<12.4f} {r_p:<12.4f} {delta_p:+.4f}")

        # Print final comparison
        print("\n\n" + "=" * 80)
        print("FINAL RESULTS COMPARISON")
        print("=" * 80)

        baseline_agg = baseline_metrics.get_aggregate_metrics()
        rag_agg = rag_metrics.get_aggregate_metrics()

        print(f"\nTotal samples: {num_samples}")
        print(f"Baseline total time: {baseline_time:.2f}s ({baseline_time/num_samples:.2f}s/sample)")
        print(f"RAG total time: {rag_time:.2f}s ({rag_time/num_samples:.2f}s/sample)")

        # Side-by-side metrics table
        print("\n" + "=" * 80)
        print("PRECISION@K Comparison")
        print("=" * 80)
        print(f"{'K':<6} {'Baseline':<15} {'RAG':<15} {'Δ (Improvement)':<20} {'% Change':<10}")
        print('─' * 80)

        for k in baseline_metrics.k_values:
            b_p = baseline_agg['precision'][k]
            r_p = rag_agg['precision'][k]
            delta = r_p - b_p
            pct_change = (delta / b_p * 100) if b_p > 0 else 0

            symbol = "🟢" if delta > 0 else "🔴" if delta < 0 else "⚪"
            print(f"{k:<6} {b_p:<15.4f} {r_p:<15.4f} {delta:+.4f} {symbol:<8} {pct_change:+.2f}%")

        print("\n" + "=" * 80)
        print("RECALL@K Comparison")
        print("=" * 80)
        print(f"{'K':<6} {'Baseline':<15} {'RAG':<15} {'Δ (Improvement)':<20} {'% Change':<10}")
        print('─' * 80)

        for k in baseline_metrics.k_values:
            b_r = baseline_agg['recall'][k]
            r_r = rag_agg['recall'][k]
            delta = r_r - b_r
            pct_change = (delta / b_r * 100) if b_r > 0 else 0

            symbol = "🟢" if delta > 0 else "🔴" if delta < 0 else "⚪"
            print(f"{k:<6} {b_r:<15.4f} {r_r:<15.4f} {delta:+.4f} {symbol:<8} {pct_change:+.2f}%")

        # F1 Score Comparison
        print("\n" + "=" * 80)
        print("F1 SCORE Comparison")
        print("=" * 80)
        print(f"{'K':<6} {'Baseline':<15} {'RAG':<15} {'Δ (Improvement)':<20} {'% Change':<10}")
        print('─' * 80)

        for k in baseline_metrics.k_values:
            b_p = baseline_agg['precision'][k]
            b_r = baseline_agg['recall'][k]
            b_f1 = 2 * (b_p * b_r) / (b_p + b_r) if (b_p + b_r) > 0 else 0

            r_p = rag_agg['precision'][k]
            r_r = rag_agg['recall'][k]
            r_f1 = 2 * (r_p * r_r) / (r_p + r_r) if (r_p + r_r) > 0 else 0

            delta = r_f1 - b_f1
            pct_change = (delta / b_f1 * 100) if b_f1 > 0 else 0

            symbol = "🟢" if delta > 0 else "🔴" if delta < 0 else "⚪"
            print(f"{k:<6} {b_f1:<15.4f} {r_f1:<15.4f} {delta:+.4f} {symbol:<8} {pct_change:+.2f}%")

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        # Find best K for each metric
        best_baseline_f1_k = max(baseline_metrics.k_values,
                                  key=lambda k: 2 * baseline_agg['precision'][k] * baseline_agg['recall'][k] /
                                                (baseline_agg['precision'][k] + baseline_agg['recall'][k])
                                                if (baseline_agg['precision'][k] + baseline_agg['recall'][k]) > 0 else 0)

        best_rag_f1_k = max(baseline_metrics.k_values,
                             key=lambda k: 2 * rag_agg['precision'][k] * rag_agg['recall'][k] /
                                           (rag_agg['precision'][k] + rag_agg['recall'][k])
                                           if (rag_agg['precision'][k] + rag_agg['recall'][k]) > 0 else 0)

        baseline_best_f1 = 2 * baseline_agg['precision'][best_baseline_f1_k] * baseline_agg['recall'][best_baseline_f1_k] / \
                           (baseline_agg['precision'][best_baseline_f1_k] + baseline_agg['recall'][best_baseline_f1_k]) \
                           if (baseline_agg['precision'][best_baseline_f1_k] + baseline_agg['recall'][best_baseline_f1_k]) > 0 else 0

        rag_best_f1 = 2 * rag_agg['precision'][best_rag_f1_k] * rag_agg['recall'][best_rag_f1_k] / \
                      (rag_agg['precision'][best_rag_f1_k] + rag_agg['recall'][best_rag_f1_k]) \
                      if (rag_agg['precision'][best_rag_f1_k] + rag_agg['recall'][best_rag_f1_k]) > 0 else 0

        print(f"""
🔵 BASELINE (No RAG):
   Best F1: {baseline_best_f1:.4f} @ K={best_baseline_f1_k}
   P@{best_baseline_f1_k}: {baseline_agg['precision'][best_baseline_f1_k]:.4f}
   R@{best_baseline_f1_k}: {baseline_agg['recall'][best_baseline_f1_k]:.4f}

🟢 RAG-ENHANCED:
   Best F1: {rag_best_f1:.4f} @ K={best_rag_f1_k}
   P@{best_rag_f1_k}: {rag_agg['precision'][best_rag_f1_k]:.4f}
   R@{best_rag_f1_k}: {rag_agg['recall'][best_rag_f1_k]:.4f}

📈 IMPROVEMENT:
   F1 Score: {rag_best_f1 - baseline_best_f1:+.4f} ({((rag_best_f1 - baseline_best_f1) / baseline_best_f1 * 100) if baseline_best_f1 > 0 else 0:+.2f}%)
   {'✅ RAG is better!' if rag_best_f1 > baseline_best_f1 else '⚠️ Baseline is better' if baseline_best_f1 > rag_best_f1 else '⚪ Same performance'}

⏱️  TIME:
   Baseline: {baseline_time:.2f}s total, {baseline_time/num_samples:.2f}s/sample
   RAG: {rag_time:.2f}s total, {rag_time/num_samples:.2f}s/sample
   Overhead: {rag_time - baseline_time:+.2f}s ({((rag_time - baseline_time) / baseline_time * 100) if baseline_time > 0 else 0:+.2f}%)
        """)

        print("\n" + "=" * 80)
        print("CONCLUSIONS")
        print("=" * 80)

        avg_improvement_p10 = rag_agg['precision'][10] - baseline_agg['precision'][10]
        avg_improvement_r10 = rag_agg['recall'][10] - baseline_agg['recall'][10]

        if avg_improvement_p10 > 0.01 and avg_improvement_r10 > 0.01:
            print("✅ RAG Enhancement provides significant improvement!")
            print(f"   P@10: +{avg_improvement_p10:.4f} ({avg_improvement_p10/baseline_agg['precision'][10]*100:+.2f}%)")
            print(f"   R@10: +{avg_improvement_r10:.4f} ({avg_improvement_r10/baseline_agg['recall'][10]*100:+.2f}%)")
            print("   Recommendation: Use RAG-enhanced version in production")
        elif avg_improvement_p10 > 0 or avg_improvement_r10 > 0:
            print("⚠️ RAG Enhancement provides modest improvement")
            print(f"   P@10: {avg_improvement_p10:+.4f}")
            print(f"   R@10: {avg_improvement_r10:+.4f}")
            print("   Recommendation: Use RAG if computational cost is acceptable")
        else:
            print("🔴 RAG Enhancement does not improve performance")
            print("   Recommendation: Stick with baseline or tune RAG parameters")

        print("\n✅ Comparison Complete!\n")


def main():
    """Main function."""
    comparison = PTMComparison(
        prescriptions_path="../data/PTM/data/prescriptions.txt",
        knowledge_path="../data/herb-knowledge.csv",
        model="deepseek-chat",
        random_seed=42  # Fixed seed for reproducibility
    )

    comparison.run_comparison(
        num_samples=20,       # Number of test samples
        vocab_size=200,       # Herb vocabulary size
        max_recommendations=30,
        rag_top_k=10         # RAG knowledge retrieval
    )


if __name__ == "__main__":
    main()
