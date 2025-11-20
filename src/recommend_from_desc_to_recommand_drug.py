"""
Medical Drug Recommendation from Disease Description.

Task: Predict recommended drugs based on disease description.
Input: Disease description (desc field)
Output: Ranked list of recommended drugs (recommand_drug field)
Evaluation: Precision@K, Recall@K for K=5,10,15,20,30
"""
import json
import os
import time
import sys
from typing import Dict, List, Set, Tuple
import numpy as np
from collections import Counter


class DescriptionDrugDataset:
    """
    Dataset for description → drug recommendation task.
    Extracts (desc, recommand_drug) pairs from medical.json.
    """

    def __init__(self, data_path: str = "../data/medical.json"):
        """
        Initialize dataset.

        Args:
            data_path: Path to medical.json
        """
        self.data_path = data_path
        self.samples = []
        self.all_drugs = set()
        self.drug_frequency = Counter()
        self.load_data()

    def load_data(self):
        """Load and process medical.json into desc→drug samples."""
        print(f"Loading dataset from {self.data_path}...")

        with open(self.data_path, 'r', encoding='utf-8') as f:
            content = f.read()

            # Parse JSON (handle both array and JSONL format)
            try:
                data = json.loads(content)
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                # Try JSONL format
                data = []
                for line in content.strip().split('\n'):
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except:
                            continue

        # Extract samples
        for entity in data:
            disease_name = entity.get('name', '')
            desc = entity.get('desc', '')
            recommand_drugs = entity.get('recommand_drug', [])

            # Skip if no description or no drugs
            if not desc or not recommand_drugs:
                continue

            # Ensure drugs is a list
            if isinstance(recommand_drugs, str):
                recommand_drugs = [recommand_drugs]
            elif not isinstance(recommand_drugs, list):
                continue

            # Filter out empty strings
            recommand_drugs = [d.strip() for d in recommand_drugs if d and d.strip()]

            if not recommand_drugs:
                continue

            # Create sample
            self.samples.append({
                'disease_name': disease_name,
                'desc': desc,
                'ground_truth': set(recommand_drugs),
                'entity': entity
            })

            # Track all drugs and frequency
            for drug in recommand_drugs:
                self.all_drugs.add(drug)
                self.drug_frequency[drug] += 1

        print(f"Loaded {len(self.samples)} samples")
        print(f"Unique drugs: {len(self.all_drugs)}")
        print(f"Average drugs per disease: {np.mean([len(s['ground_truth']) for s in self.samples]):.2f}")

        # Print drug count distribution
        drug_counts = [len(s['ground_truth']) for s in self.samples]
        print(f"\nGround truth distribution:")
        print(f"  Min: {min(drug_counts)}")
        print(f"  Max: {max(drug_counts)}")
        print(f"  Mean: {np.mean(drug_counts):.2f}")
        print(f"  Median: {np.median(drug_counts):.1f}")
        print(f"  Std: {np.std(drug_counts):.2f}")

    def get_sample(self, index: int) -> Dict:
        """Get sample by index."""
        return self.samples[index]

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.samples)

    def get_top_drugs(self, k: int = 50) -> List[str]:
        """Get top K most frequent drugs."""
        return [drug for drug, _ in self.drug_frequency.most_common(k)]


class DescriptionDrugPredictor:
    """
    Predicts recommended drugs from disease description using DeepSeek API.
    """

    def __init__(self, api_key: str = None, model: str = "deepseek-chat"):
        """
        Initialize predictor.

        Args:
            api_key: DeepSeek API key (default: from DEEPSEEK_API_KEY env)
            model: Model name
        """
        from openai import OpenAI

        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError("API key required. Set DEEPSEEK_API_KEY environment variable.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )

        print(f"Initialized predictor with model: {self.model}")

    def predict(self, description: str, top_k: int = 30) -> List[str]:
        """
        Predict recommended drugs from disease description.

        Args:
            description: Disease description
            top_k: Number of drugs to recommend

        Returns:
            Ranked list of drug names
        """
        prompt = f"""你是一个专业的医学助手。根据以下疾病描述，推荐适合的药物治疗方案。

疾病描述：
{description}

要求：
1. 请直接列出推荐的药物名称，每个药物用顿号（、）分隔
2. 按照推荐优先级从高到低排序
3. 只输出药物名称，不要输出任何解释或说明
4. 尽可能多推荐一些药物（至少15个）
5. 优先推荐常用、有效的药物

推荐药物："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的医学助手，擅长根据疾病描述推荐药物。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                top_p=0.85,
                max_tokens=800
            )

            result_text = response.choices[0].message.content.strip()

            # Extract drugs from response
            drugs = self._extract_drugs(result_text)

            # Return top K
            return drugs[:top_k]

        except Exception as e:
            print(f"Error in prediction: {e}")
            return []

    def _extract_drugs(self, text: str) -> List[str]:
        """Extract drug names from response text."""
        import re

        if not text:
            return []

        # Remove parenthetical content
        text = re.sub(r'\([^)]*\)', '', text)
        text = re.sub(r'（[^）]*）', '', text)
        text = re.sub(r'\[[^\]]*\]', '', text)

        # Common delimiters
        delimiters = ['、', '，', ',', '；', ';', '和', '或', '以及', '\n', '。']

        # Split by delimiters
        drugs = [text]
        for delimiter in delimiters:
            new_drugs = []
            for drug in drugs:
                new_drugs.extend(drug.split(delimiter))
            drugs = new_drugs

        # Clean up
        drugs = [d.strip() for d in drugs if d.strip()]

        # Remove prefixes and suffixes
        prefix_patterns = [
            r'^建议使用?', r'^推荐使用?', r'^可以使用?',
            r'^使用', r'^服用', r'^选用', r'^采用',
            r'^\d+[\.\、]',  # Remove numbering
        ]

        suffix_patterns = [
            r'进行治疗$', r'治疗$', r'为宜$', r'较好$', r'等$',
        ]

        cleaned_drugs = []
        for drug in drugs:
            for pattern in prefix_patterns:
                drug = re.sub(pattern, '', drug)
            for pattern in suffix_patterns:
                drug = re.sub(pattern, '', drug)
            drug = drug.strip()
            cleaned_drugs.append(drug)

        drugs = cleaned_drugs

        # Filter out common phrases
        exclude_phrases = {
            '基于LLM直接推荐', '基于知识图谱推荐', '推荐', '建议', '可以',
            '使用', '服用', '等', '以下', '药物', '治疗', '方法', '如下',
            '包括', '有', '为', '是', '的', '在', '及', '与', '或', '和',
            '请', '应', '需', '要', '还', '也', '', '进行'
        }

        drugs = [d for d in drugs if d and d not in exclude_phrases and len(d) >= 2]

        # Remove duplicates while preserving order
        seen = set()
        unique_drugs = []
        for drug in drugs:
            if drug not in seen:
                seen.add(drug)
                unique_drugs.append(drug)

        return unique_drugs


class RecommenderMetrics:
    """
    Evaluation metrics for drug recommendation.
    Calculates Precision@K and Recall@K.
    """

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
        print(f"{prefix} Drug Recommendation Metrics")
        print('=' * 80)
        print(f"Total samples: {metrics['num_samples']}\n")

        # Print table
        print(f"{'Metric':<15}", end='')
        for k in self.k_values:
            print(f" @{k:<8}", end='')
        print()
        print('─' * 80)

        # Precision
        print(f"{'Precision':<15}", end='')
        for k in self.k_values:
            print(f" {metrics['precision'][k]:.4f}   ", end='')
        print()

        # Recall
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
    ║         Drug Recommendation from Disease Description (Baseline)           ║
    ║                    Input: desc → Output: recommand_drug                   ║
    ║                                                                            ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """)

    # Configuration
    config = {
        'data_path': '../data/medical.json',
        'model': 'deepseek-chat',
        'num_samples': 20,  # Number of samples to test
        'max_recommendations': 30
    }

    print("Configuration:")
    print(f"  Data path: {config['data_path']}")
    print(f"  Model: {config['model']}")
    print(f"  Test samples: {config['num_samples']}")
    print(f"  Max recommendations: {config['max_recommendations']}")

    # Load dataset
    print("\n" + "=" * 80)
    print("Loading Dataset")
    print("=" * 80)
    dataset = DescriptionDrugDataset(config['data_path'])

    # Initialize predictor
    print("\n" + "=" * 80)
    print("Initializing Predictor")
    print("=" * 80)
    predictor = DescriptionDrugPredictor(model=config['model'])

    # Initialize metrics
    metrics = RecommenderMetrics()

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
        disease_name = sample['disease_name']
        description = sample['desc']
        ground_truth = sample['ground_truth']

        print(f"\nDisease: {disease_name}")
        print(f"Description: {description[:200]}...")
        print(f"Ground Truth ({len(ground_truth)} drugs): {list(ground_truth)[:5]}...")

        # Predict
        start_time = time.time()
        recommendations = predictor.predict(description, top_k=config['max_recommendations'])
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

    # Find optimal K
    print("\nOptimal K Analysis:")
    print("Looking for best trade-off between Precision and Recall...\n")

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

2. **Performance Observations**:
   - Average drugs per disease: {np.mean([len(s['ground_truth']) for s in dataset.samples]):.2f}
   - Unique drugs in dataset: {len(dataset.all_drugs)}
   - Task difficulty: Predicting from {len(dataset.all_drugs)} possible drugs

3. **Next Steps**:
   - Add Knowledge Graph enhancement
   - Experiment with different prompts
   - Consider drug frequency in predictions
   - Add drug category information

4. **Current Limitations**:
   - No KG context (baseline)
   - LLM may hallucinate drug names
   - No drug category/type filtering
    """)

    print("\n✅ Evaluation Complete!")


if __name__ == "__main__":
    main()
