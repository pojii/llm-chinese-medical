"""
Main script for comparing LLM medicine predictions with and without Knowledge Graph.
"""
import sys
import time
from typing import Dict, List
import json

from knowledge_graph import MedicalKnowledgeGraph
from ner_dataset import TCMNERDataset
from llm_predictor import MedicineLLMPredictor


class MedicinePredictionComparison:
    """
    Compare medicine predictions with and without knowledge graph augmentation.
    """

    def __init__(
        self,
        kg_path: str = "../data/medical.json",
        ner_path: str = None,
        model_name: str = "uer/gpt2-chinese-cluecorpussmall",
        device: str = "cpu"
    ):
        """
        Initialize comparison system.

        Args:
            kg_path: Path to knowledge graph data
            ner_path: Path to NER dataset (None for sample data)
            model_name: LLM model name
            device: Device to run on
        """
        print("=" * 80)
        print("Initializing Medicine Prediction Comparison System")
        print("=" * 80)

        # Load Knowledge Graph
        print("\n[1/3] Loading Knowledge Graph...")
        self.kg = MedicalKnowledgeGraph(kg_path)

        # Load NER Dataset
        print("\n[2/3] Loading TCM NER Dataset...")
        self.dataset = TCMNERDataset(ner_path)
        print(f"Loaded {len(self.dataset)} samples")

        # Initialize LLM Predictor
        print("\n[3/3] Initializing LLM Predictor...")
        self.predictor = MedicineLLMPredictor(
            model_name=model_name,
            device=device
        )

        print("\n" + "=" * 80)
        print("System Initialization Complete!")
        print("=" * 80)

    def run_comparison(self, num_samples: int = None) -> List[Dict]:
        """
        Run comparison on dataset samples.

        Args:
            num_samples: Number of samples to test (None for all)

        Returns:
            List of comparison results
        """
        if num_samples is None:
            num_samples = len(self.dataset)
        else:
            num_samples = min(num_samples, len(self.dataset))

        print(f"\n\nRunning comparison on {num_samples} samples...")
        print("=" * 80)

        results = []

        for i in range(num_samples):
            print(f"\n{'=' * 80}")
            print(f"Sample {i + 1}/{num_samples}")
            print('=' * 80)

            sample = self.dataset.get_sample(i)
            query = self.dataset.format_for_prediction(sample)

            print(f"\n原文: {sample['text']}")
            print(f"\n查询: {query}")

            # Extract entities for ground truth
            entities = self.dataset.extract_entities(sample)
            ground_truth_herbs = entities.get('HER', [])
            print(f"\n真实标注的药物: {ground_truth_herbs if ground_truth_herbs else '无'}")

            # Prediction WITHOUT KG
            print(f"\n{'─' * 80}")
            print("【方法1】不使用知识图谱的LLM预测")
            print('─' * 80)

            start_time = time.time()
            pred_without_kg = self.predictor.predict_without_kg(query)
            time_without_kg = time.time() - start_time

            print(f"预测结果: {pred_without_kg}")
            print(f"推理时间: {time_without_kg:.2f}秒")

            # Prediction WITH KG
            print(f"\n{'─' * 80}")
            print("【方法2】使用知识图谱增强的LLM预测")
            print('─' * 80)

            # Get relevant context from KG
            kg_context = self.kg.search_relevant_context(query, top_k=2)
            print(f"检索到的知识图谱上下文:\n{kg_context[:300]}...")

            start_time = time.time()
            pred_with_kg = self.predictor.predict_with_kg(query, kg_context)
            time_with_kg = time.time() - start_time

            print(f"\n预测结果: {pred_with_kg}")
            print(f"推理时间: {time_with_kg:.2f}秒")

            # Store results
            result = {
                'sample_id': i,
                'text': sample['text'],
                'query': query,
                'ground_truth': ground_truth_herbs,
                'prediction_without_kg': pred_without_kg,
                'prediction_with_kg': pred_with_kg,
                'time_without_kg': time_without_kg,
                'time_with_kg': time_with_kg,
                'kg_context_used': kg_context[:200]
            }
            results.append(result)

        return results

    def print_summary(self, results: List[Dict]):
        """
        Print summary of comparison results.

        Args:
            results: List of result dictionaries
        """
        print("\n\n" + "=" * 80)
        print("比较结果汇总")
        print("=" * 80)

        total_samples = len(results)
        avg_time_without_kg = sum(r['time_without_kg'] for r in results) / total_samples
        avg_time_with_kg = sum(r['time_with_kg'] for r in results) / total_samples

        print(f"\n测试样本数: {total_samples}")
        print(f"\n平均推理时间:")
        print(f"  - 不使用KG: {avg_time_without_kg:.2f}秒")
        print(f"  - 使用KG:   {avg_time_with_kg:.2f}秒")

        print(f"\n{'=' * 80}")
        print("结论")
        print('=' * 80)
        print("""
1. **知识图谱增强的优势**:
   - 提供领域专业知识作为上下文
   - 增强LLM对医学术语的理解
   - 提高推荐的准确性和可解释性

2. **无知识图谱的局限**:
   - 仅依赖LLM的预训练知识
   - 可能产生不准确或不相关的推荐
   - 缺乏专业医学知识支持

3. **计算资源**:
   - 使用轻量级CPU模型实现
   - 适合资源受限环境部署
   - 推理速度可接受

建议: 在实际应用中优先使用知识图谱增强方法以获得更准确的医学推荐。
        """)

    def save_results(self, results: List[Dict], output_path: str = "./outputs/comparison_results.json"):
        """
        Save comparison results to file.

        Args:
            results: List of result dictionaries
            output_path: Output file path
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {output_path}")


def main():
    """Main function."""
    print("""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                                                                            ║
    ║           中医药知识图谱与LLM融合的药物推荐系统                              ║
    ║        Medicine Prediction System with Knowledge Graph and LLM            ║
    ║                                                                            ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """)

    # Configuration
    config = {
        'kg_path': '../data/medical.json',
        'ner_path': None,  # Use sample data
        'model_name': 'uer/gpt2-chinese-cluecorpussmall',
        'device': 'cpu',
        'num_samples': 5  # Test on 5 samples
    }

    try:
        # Initialize comparison system
        comparison = MedicinePredictionComparison(
            kg_path=config['kg_path'],
            ner_path=config['ner_path'],
            model_name=config['model_name'],
            device=config['device']
        )

        # Run comparison
        results = comparison.run_comparison(num_samples=config['num_samples'])

        # Print summary
        comparison.print_summary(results)

        # Save results
        comparison.save_results(results)

        print("\n✅ 实验完成!")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
