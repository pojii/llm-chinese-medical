"""
PTM Herb Recommender - Symptom to Traditional Chinese Medicine Herbs
Baseline model without Knowledge Graph enhancement.
"""
import re
import os
import time
from typing import List, Set, Dict
from collections import Counter
import numpy as np


class PTMHerbDataset:
    """
    Dataset for PTM symptom→herb recommendation.
    Loads prescriptions.txt and extracts symptom-herb pairs.
    """

    def __init__(self, prescriptions_path: str = "../data/PTM/data/prescriptions.txt"):
        """
        Initialize dataset.

        Args:
            prescriptions_path: Path to prescriptions.txt
        """
        self.prescriptions_path = prescriptions_path
        self.samples = []
        self.all_herbs = set()
        self.herb_frequency = Counter()
        self.load_data()

    def load_data(self):
        """Load prescriptions and extract symptom→herb pairs."""
        print(f"Loading PTM dataset from: {self.prescriptions_path}")

        with open(self.prescriptions_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                # Split by tab: symptoms \t herbs
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue

                symptoms = parts[0].strip()
                herbs_text = parts[1].strip()

                # Extract herbs
                herbs = self._extract_herbs(herbs_text)

                if not symptoms or not herbs:
                    continue

                # Create sample
                self.samples.append({
                    'symptoms': symptoms,
                    'ground_truth': set(herbs),
                    'herbs_text': herbs_text
                })

                # Track herb frequency
                for herb in herbs:
                    self.all_herbs.add(herb)
                    self.herb_frequency[herb] += 1

        print(f"Loaded {len(self.samples)} samples")
        print(f"Unique herbs: {len(self.all_herbs)}")
        print(f"Average herbs per prescription: {np.mean([len(s['ground_truth']) for s in self.samples]):.2f}")

    def _extract_herbs(self, herbs_text: str) -> List[str]:
        """Extract herb names from prescription text."""
        herbs = []
        parts = herbs_text.strip().split()

        for part in parts:
            if not part:
                continue

            # Remove parentheses (preparation methods)
            herb_name = re.sub(r'[（(].*?[）)]', '', part)

            # Remove dosage patterns
            herb_name = re.sub(r'[一二三四五六七八九十百千半]+[钱两分厘克斤枚个粒片丸分两]$', '', herb_name)
            herb_name = re.sub(r'\d+[钱两分厘克斤枚个粒片丸分两]$', '', herb_name)

            # Remove "各"
            herb_name = re.sub(r'^各', '', herb_name)
            herb_name = re.sub(r'各$', '', herb_name)

            # Skip standalone numbers
            if re.match(r'^[一二三四五六七八九十百千半\d]+$', herb_name):
                continue

            herb_name = herb_name.strip()

            # Filter: must have Chinese characters and length >= 2
            if herb_name and len(herb_name) >= 2 and re.search(r'[\u4e00-\u9fff]', herb_name):
                if not re.match(r'^[\d一二三四五六七八九十百千半钱两分厘克斤枚个粒片丸]+$', herb_name):
                    herbs.append(herb_name)

        return herbs

    def get_sample(self, index: int) -> Dict:
        """Get sample by index."""
        return self.samples[index]

    def get_top_herbs(self, k: int = 100) -> List[str]:
        """Get top K most frequent herbs."""
        return [herb for herb, _ in self.herb_frequency.most_common(k)]

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.samples)


class PTMHerbPredictor:
    """
    Predict TCM herbs from symptoms using DeepSeek API.
    Uses constrained vocabulary to prevent hallucination.
    """

    def __init__(
        self,
        herb_vocabulary: List[str],
        api_key: str = None,
        model: str = "deepseek-chat"
    ):
        """
        Initialize predictor.

        Args:
            herb_vocabulary: List of valid herb names
            api_key: DeepSeek API key
            model: Model name
        """
        from openai import OpenAI

        self.herb_vocabulary = herb_vocabulary
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError("API key required. Set DEEPSEEK_API_KEY environment variable.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )

        print(f"Initialized predictor with {len(herb_vocabulary)} herbs vocabulary")

    def predict(self, symptoms: str, top_k: int = 30) -> List[str]:
        """
        Predict herbs from symptoms.

        Args:
            symptoms: Symptom description
            top_k: Number of herbs to recommend

        Returns:
            Ranked list of herb names
        """
        # Create herb vocabulary string (top 200 herbs to keep prompt manageable)
        herb_list_str = '、'.join(self.herb_vocabulary[:200])

        prompt = f"""你是一个专业的中医助手。根据患者的症状描述，从给定的中药列表中推荐适合的中药。

【患者症状】
{symptoms}

【可用中药列表】（只能从这个列表中选择）
{herb_list_str}

【要求】
1. 只能推荐列表中的中药，不要推荐列表外的药物
2. 按优先级从高到低排序
3. 每个中药用顿号（、）分隔
4. 只输出中药名称，不要任何解释
5. 尽量推荐10-20种中药

【推荐中药】："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是专业的中医师，擅长根据症状推荐中药配方。严格遵守给定的中药列表，不推荐列表外的药物。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                top_p=0.85,
                max_tokens=500
            )

            result_text = response.choices[0].message.content.strip()

            # Extract herbs
            herbs = self._extract_herbs(result_text)

            # Filter: only keep herbs in vocabulary
            valid_herbs = [h for h in herbs if h in self.herb_vocabulary]

            return valid_herbs[:top_k]

        except Exception as e:
            print(f"Error in prediction: {e}")
            return []

    def _extract_herbs(self, text: str) -> List[str]:
        """Extract herb names from response text."""
        if not text:
            return []

        # Remove parentheses
        text = re.sub(r'\([^)]*\)', '', text)
        text = re.sub(r'（[^）]*）', '', text)
        text = re.sub(r'\[[^\]]*\]', '', text)

        # Split by delimiters
        delimiters = ['、', '，', ',', '；', ';', '和', '或', '以及', '\n', '。']
        herbs = [text]
        for delimiter in delimiters:
            new_herbs = []
            for herb in herbs:
                new_herbs.extend(herb.split(delimiter))
            herbs = new_herbs

        # Clean up
        herbs = [h.strip() for h in herbs if h.strip()]

        # Remove prefixes/suffixes
        prefix_patterns = [
            r'^推荐', r'^建议', r'^使用', r'^可用',
            r'^\d+[\.\、]',  # Remove numbering
        ]

        cleaned_herbs = []
        for herb in herbs:
            for pattern in prefix_patterns:
                herb = re.sub(pattern, '', herb)
            herb = herb.strip()
            cleaned_herbs.append(herb)

        # Filter
        exclude = {'推荐', '建议', '使用', '等', '中药', '药物', ''}
        herbs = [h for h in cleaned_herbs if h not in exclude and len(h) >= 2]

        # Remove duplicates while preserving order
        seen = set()
        unique_herbs = []
        for herb in herbs:
            if herb not in seen:
                seen.add(herb)
                unique_herbs.append(herb)

        return unique_herbs


class HerbRecommenderMetrics:
    """Evaluation metrics for herb recommendation."""

    def __init__(self):
        """Initialize metrics."""
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self.k_values = [5, 10, 15, 20, 30]
        self.all_precisions = {k: [] for k in self.k_values}
        self.all_recalls = {k: [] for k in self.k_values}
        self.num_samples = 0

    @staticmethod
    def precision_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
        """Calculate Precision@K."""
        if k <= 0 or not recommended:
            return 0.0
        top_k = recommended[:k]
        relevant_in_top_k = sum(1 for item in top_k if item in relevant)
        return relevant_in_top_k / k

    @staticmethod
    def recall_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
        """Calculate Recall@K."""
        if not relevant or k <= 0 or not recommended:
            return 0.0
        top_k = recommended[:k]
        relevant_in_top_k = sum(1 for item in top_k if item in relevant)
        return relevant_in_top_k / len(relevant)

    def evaluate_single(self, recommended: List[str], relevant: Set[str]) -> Dict:
        """Evaluate single prediction."""
        results = {}
        for k in self.k_values:
            results[k] = {
                'precision': self.precision_at_k(recommended, relevant, k),
                'recall': self.recall_at_k(recommended, relevant, k)
            }
        return results

    def update(self, recommended: List[str], relevant: Set[str]):
        """Update accumulated metrics."""
        metrics = self.evaluate_single(recommended, relevant)
        for k in self.k_values:
            self.all_precisions[k].append(metrics[k]['precision'])
            self.all_recalls[k].append(metrics[k]['recall'])
        self.num_samples += 1

    def get_aggregate_metrics(self) -> Dict:
        """Get aggregate metrics."""
        results = {
            'precision': {},
            'recall': {},
            'num_samples': self.num_samples
        }
        for k in self.k_values:
            results['precision'][k] = np.mean(self.all_precisions[k]) if self.all_precisions[k] else 0.0
            results['recall'][k] = np.mean(self.all_recalls[k]) if self.all_recalls[k] else 0.0
        return results

    def print_summary(self, prefix: str = ""):
        """Print metrics summary."""
        metrics = self.get_aggregate_metrics()

        print(f"\n{'=' * 80}")
        print(f"{prefix} TCM Herb Recommendation Metrics")
        print('=' * 80)
        print(f"Total samples: {metrics['num_samples']}\n")

        print(f"{'Metric':<15}", end='')
        for k in self.k_values:
            print(f" @{k:<8}", end='')
        print()
        print('─' * 80)

        print(f"{'Precision':<15}", end='')
        for k in self.k_values:
            print(f" {metrics['precision'][k]:.4f}   ", end='')
        print()

        print(f"{'Recall':<15}", end='')
        for k in self.k_values:
            print(f" {metrics['recall'][k]:.4f}   ", end='')
        print()

        print('=' * 80)


def main():
    """Main evaluation function."""
    print("""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                                                                            ║
    ║         TCM Herb Recommendation from Symptoms (Baseline)                  ║
    ║                    PTM Dataset - Symptom → Herbs                          ║
    ║                                                                            ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """)

    # Configuration
    config = {
        'prescriptions_path': '../data/PTM/data/prescriptions.txt',
        'model': 'deepseek-chat',
        'num_samples': 20,  # Number of test samples
        'vocab_size': 200,  # Top N herbs to include in vocabulary
        'max_recommendations': 30
    }

    print("Configuration:")
    print(f"  Data path: {config['prescriptions_path']}")
    print(f"  Model: {config['model']}")
    print(f"  Test samples: {config['num_samples']}")
    print(f"  Herb vocabulary size: {config['vocab_size']}")

    # Load dataset
    print("\n" + "=" * 80)
    print("Loading Dataset")
    print("=" * 80)
    dataset = PTMHerbDataset(config['prescriptions_path'])

    # Get top herbs as vocabulary
    herb_vocabulary = dataset.get_top_herbs(config['vocab_size'])
    print(f"\nTop 20 herbs in vocabulary:")
    for i, herb in enumerate(herb_vocabulary[:20], 1):
        freq = dataset.herb_frequency[herb]
        print(f"  {i:2d}. {herb} ({freq:,} times)")

    # Initialize predictor
    print("\n" + "=" * 80)
    print("Initializing Predictor")
    print("=" * 80)
    predictor = PTMHerbPredictor(
        herb_vocabulary=herb_vocabulary,
        model=config['model']
    )

    # Initialize metrics
    metrics = HerbRecommenderMetrics()

    # Run evaluation
    print("\n" + "=" * 80)
    print("Running Evaluation")
    print("=" * 80)

    num_samples = min(config['num_samples'], len(dataset))
    total_time = 0

    for i in range(num_samples):
        print(f"\n{'=' * 80}")
        print(f"Sample {i + 1}/{num_samples}")
        print('=' * 80)

        sample = dataset.get_sample(i)
        symptoms = sample['symptoms']
        ground_truth = sample['ground_truth']

        print(f"\nSymptoms: {symptoms}")
        print(f"Ground Truth ({len(ground_truth)} herbs): {list(ground_truth)[:5]}...")

        # Predict
        start_time = time.time()
        recommendations = predictor.predict(symptoms, top_k=config['max_recommendations'])
        inference_time = time.time() - start_time
        total_time += inference_time

        print(f"\nTop {min(10, len(recommendations))} Predictions: {recommendations[:10]}")
        print(f"Inference Time: {inference_time:.2f}s")

        # Calculate metrics
        sample_metrics = metrics.evaluate_single(recommendations, ground_truth)

        print(f"\n{'─' * 80}")
        print("Sample Metrics")
        print('─' * 80)
        for k in [5, 10, 15, 20, 30]:
            print(f"@{k:2d} - P: {sample_metrics[k]['precision']:.4f}, R: {sample_metrics[k]['recall']:.4f}")

        # Update aggregate
        metrics.update(recommendations, ground_truth)

    # Print summary
    print("\n\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    print(f"Total samples evaluated: {num_samples}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per sample: {total_time / num_samples:.2f}s")

    metrics.print_summary("Baseline")

    # Analysis
    print("\n\n" + "=" * 80)
    print("Analysis")
    print("=" * 80)

    agg_metrics = metrics.get_aggregate_metrics()

    # Find optimal K using F1
    print("\nOptimal K Analysis (F1 Score):")
    f1_scores = {}
    for k in metrics.k_values:
        p = agg_metrics['precision'][k]
        r = agg_metrics['recall'][k]
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
        f1_scores[k] = f1
        print(f"K={k:2d}: P={p:.4f}, R={r:.4f}, F1={f1:.4f}")

    best_k = max(f1_scores, key=f1_scores.get)
    print(f"\n🎯 Optimal K (best F1): {best_k}")

    # Recommendations
    print("\n\n" + "=" * 80)
    print("Recommendations")
    print("=" * 80)
    print(f"""
Based on the evaluation:

1. **Optimal K value**: {best_k}
   - Best trade-off between precision and recall
   - F1 Score: {f1_scores[best_k]:.4f}

2. **Performance Summary**:
   - Vocabulary size: {config['vocab_size']} herbs
   - Coverage: Top {config['vocab_size']} herbs cover ~{dataset.herb_frequency.most_common(config['vocab_size'])[-1][1] / sum(dataset.herb_frequency.values()) * 100:.1f}% of usage
   - Constrained vocabulary prevents hallucination

3. **Dataset Characteristics**:
   - Average herbs per prescription: {np.mean([len(s['ground_truth']) for s in dataset.samples]):.2f}
   - Total unique herbs: {len(dataset.all_herbs)}
   - Using top {config['vocab_size']} for evaluation

4. **Next Steps**:
   - Expand vocabulary size if recall is low
   - Add herb category information
   - Add contraindication knowledge
   - Test with Knowledge Graph enhancement
    """)

    print("\n✅ Evaluation Complete!")


if __name__ == "__main__":
    main()
