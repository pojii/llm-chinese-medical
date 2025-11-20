"""
PTM Herb Recommender with MUSE Embeddings
Uses Google's Multilingual Universal Sentence Encoder (MUSE) for better Chinese support.

MUSE Model:
- Universal Sentence Encoder Multilingual
- Supports 16 languages including Chinese
- 512-dimensional embeddings
- Local model (~1GB)
"""
import re
import os
import csv
import numpy as np
from typing import List, Set, Dict
from collections import Counter
import time


class HerbKnowledgeRAGWithMUSE:
    """
    RAG system using MUSE (Multilingual Universal Sentence Encoder).
    Better for Chinese medical text than generic multilingual models.
    """

    def __init__(self, knowledge_path: str = "../data/herb-knowledge.csv"):
        """
        Initialize RAG system with MUSE.

        Args:
            knowledge_path: Path to herb-knowledge.csv
        """
        self.knowledge_path = knowledge_path
        self.knowledge_entries = []
        self.embeddings = None
        self.embed_model = None

        self.load_knowledge()
        self.initialize_muse()
        self.create_embeddings()

    def initialize_muse(self):
        """Load MUSE model from TensorFlow Hub."""
        print("=" * 80)
        print("Loading MUSE Model (Multilingual Universal Sentence Encoder)")
        print("=" * 80)
        print("Model: universal-sentence-encoder-multilingual/3")
        print("Size: ~1GB (will download on first run)")
        print("Languages: 16 including Chinese, English")
        print("Dimensions: 512")
        print()

        try:
            import tensorflow as tf
            import tensorflow_hub as hub

            # Suppress TensorFlow warnings
            tf.get_logger().setLevel('ERROR')
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

            print("Loading MUSE model from TensorFlow Hub...")
            start_time = time.time()

            # Load MUSE model
            self.embed_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

            load_time = time.time() - start_time
            print(f"✅ MUSE model loaded successfully! ({load_time:.2f}s)")
            print()

        except ImportError as e:
            print("❌ Error: tensorflow and tensorflow_hub required for MUSE")
            print()
            print("Install with:")
            print("  pip install tensorflow tensorflow-hub")
            print()
            raise e

    def load_knowledge(self):
        """Load herb knowledge from CSV."""
        print(f"Loading herb knowledge from: {self.knowledge_path}")

        with open(self.knowledge_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                knowledge_text = self._create_knowledge_text(row)

                self.knowledge_entries.append({
                    'pinyin': row.get('Pinyin Name', ''),
                    'english': row.get('English Name', ''),
                    'attributes': row.get('Attributes', ''),
                    'meridians': row.get('Meridians/Energy_channels', ''),
                    'effect': row.get('Effect', ''),
                    'indication': row.get('Indication', ''),
                    'knowledge_text': knowledge_text
                })

        print(f"Loaded {len(self.knowledge_entries)} herb knowledge entries")

    def _create_knowledge_text(self, row: Dict) -> str:
        """Create combined knowledge text for embedding."""
        parts = []

        if row.get('Attributes'):
            parts.append(f"Properties: {row['Attributes']}")
        if row.get('Effect'):
            parts.append(f"Effects: {row['Effect']}")
        if row.get('Indication'):
            parts.append(f"Treats: {row['Indication']}")

        return '. '.join(parts)

    def create_embeddings(self):
        """Create embeddings for all knowledge using MUSE."""
        print()
        print("=" * 80)
        print("Creating Embeddings with MUSE")
        print("=" * 80)

        knowledge_texts = [entry['knowledge_text'] for entry in self.knowledge_entries]

        print(f"Embedding {len(knowledge_texts)} herb knowledge entries...")
        start_time = time.time()

        # MUSE can handle batches efficiently
        self.embeddings = self.embed_model(knowledge_texts).numpy()

        embed_time = time.time() - start_time
        print(f"✅ Created {len(self.embeddings)} embeddings ({embed_time:.2f}s)")
        print(f"   Embedding shape: {self.embeddings.shape}")
        print()

    def retrieve_relevant_knowledge(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.3
    ) -> List[Dict]:
        """
        Retrieve top K most relevant herb knowledge using MUSE embeddings.

        Args:
            query: Symptom description
            top_k: Number of entries to retrieve
            similarity_threshold: Minimum similarity score

        Returns:
            List of most relevant knowledge entries
        """
        # Encode query with MUSE
        query_embedding = self.embed_model([query]).numpy()[0]

        # Calculate cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Filter by threshold
        valid_indices = [i for i, sim in enumerate(similarities) if sim >= similarity_threshold]

        if not valid_indices:
            return []

        # Get top K from valid indices
        valid_similarities = similarities[valid_indices]
        top_indices_in_valid = np.argsort(valid_similarities)[::-1][:top_k]
        top_indices = [valid_indices[i] for i in top_indices_in_valid]

        # Return results
        results = []
        for idx in top_indices:
            entry = self.knowledge_entries[idx].copy()
            entry['similarity'] = float(similarities[idx])
            results.append(entry)

        return results


class PTMHerbDataset:
    """Dataset for PTM symptom→herb recommendation."""

    def __init__(self, prescriptions_path: str = "../data/PTM/data/prescriptions.txt"):
        self.prescriptions_path = prescriptions_path
        self.samples = []
        self.all_herbs = set()
        self.herb_frequency = Counter()
        self.load_data()

    def load_data(self):
        """Load prescriptions and extract symptom→herb pairs."""
        print(f"Loading PTM dataset from: {self.prescriptions_path}")

        with open(self.prescriptions_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue

                symptoms = parts[0].strip()
                herbs_text = parts[1].strip()
                herbs = self._extract_herbs(herbs_text)

                if not symptoms or not herbs:
                    continue

                self.samples.append({
                    'symptoms': symptoms,
                    'ground_truth': set(herbs),
                    'herbs_text': herbs_text
                })

                for herb in herbs:
                    self.all_herbs.add(herb)
                    self.herb_frequency[herb] += 1

        print(f"Loaded {len(self.samples)} samples")
        print(f"Unique herbs: {len(self.all_herbs)}")

    def _extract_herbs(self, herbs_text: str) -> List[str]:
        """Extract herb names from prescription text."""
        herbs = []
        parts = herbs_text.strip().split()

        for part in parts:
            if not part:
                continue

            herb_name = re.sub(r'[（(].*?[）)]', '', part)
            herb_name = re.sub(r'[一二三四五六七八九十百千半]+[钱两分厘克斤枚个粒片丸分两]$', '', herb_name)
            herb_name = re.sub(r'\d+[钱两分厘克斤枚个粒片丸分两]$', '', herb_name)
            herb_name = re.sub(r'^各', '', herb_name)
            herb_name = re.sub(r'各$', '', herb_name)

            if re.match(r'^[一二三四五六七八九十百千半\d]+$', herb_name):
                continue

            herb_name = herb_name.strip()

            if herb_name and len(herb_name) >= 2 and re.search(r'[\u4e00-\u9fff]', herb_name):
                if not re.match(r'^[\d一二三四五六七八九十百千半钱两分厘克斤枚个粒片丸]+$', herb_name):
                    herbs.append(herb_name)

        return herbs

    def get_sample(self, index: int) -> Dict:
        return self.samples[index]

    def get_top_herbs(self, k: int = 100) -> List[str]:
        return [herb for herb, _ in self.herb_frequency.most_common(k)]

    def __len__(self) -> int:
        return len(self.samples)


class PTMHerbPredictorWithMUSE:
    """Predictor using MUSE embeddings for RAG."""

    def __init__(
        self,
        herb_vocabulary: List[str],
        rag_system: HerbKnowledgeRAGWithMUSE,
        api_key: str = None,
        model: str = "deepseek-chat"
    ):
        from openai import OpenAI

        self.herb_vocabulary = herb_vocabulary
        self.rag_system = rag_system
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError("DeepSeek API key required. Set DEEPSEEK_API_KEY.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )

        print(f"Initialized MUSE-RAG predictor with {len(herb_vocabulary)} herbs")

    def predict(
        self,
        symptoms: str,
        top_k: int = 30,
        rag_top_k: int = 5,
        similarity_threshold: float = 0.3
    ) -> List[str]:
        """Predict herbs with MUSE-based RAG."""
        herb_list_str = '、'.join(self.herb_vocabulary[:200])

        # Retrieve with MUSE
        relevant_knowledge = self.rag_system.retrieve_relevant_knowledge(
            symptoms,
            top_k=rag_top_k,
            similarity_threshold=similarity_threshold
        )

        knowledge_str = ""
        if relevant_knowledge:
            knowledge_str = self._format_knowledge(relevant_knowledge)

        # Create prompt
        if knowledge_str:
            prompt = f"""你是专业的中医师。根据患者症状，从给定的中药列表推荐适合的中药。

【相关中药知识】
{knowledge_str}

【患者症状】
{symptoms}

【可用中药列表】（只能从这个列表中选择）
{herb_list_str}

【要求】
1. 参考上述中药知识
2. 只推荐列表中的中药
3. 按优先级排序
4. 用顿号（、）分隔
5. 只输出中药名称

【推荐中药】："""
        else:
            prompt = f"""你是专业的中医师。根据患者症状，从给定的中药列表推荐适合的中药。

【患者症状】
{symptoms}

【可用中药列表】（只能从这个列表中选择）
{herb_list_str}

【要求】
1. 只推荐列表中的中药
2. 按优先级排序
3. 用顿号（、）分隔
4. 只输出中药名称

【推荐中药】："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是专业的中医师。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                top_p=0.85,
                max_tokens=500
            )

            result_text = response.choices[0].message.content.strip()
            herbs = self._extract_herbs(result_text)
            valid_herbs = [h for h in herbs if h in self.herb_vocabulary]

            return valid_herbs[:top_k]

        except Exception as e:
            print(f"Error: {e}")
            return []

    def _format_knowledge(self, knowledge_entries: List[Dict]) -> str:
        """Format knowledge (simplified)."""
        formatted = []
        for i, entry in enumerate(knowledge_entries, 1):
            indication = entry.get('indication', '').strip()
            if indication:
                if len(indication) > 100:
                    indication = indication[:100] + "..."
                formatted.append(f"{i}. {entry['pinyin']}: {indication}")
        return '\n'.join(formatted[:5])

    def _extract_herbs(self, text: str) -> List[str]:
        """Extract herb names."""
        if not text:
            return []

        text = re.sub(r'\([^)]*\)', '', text)
        text = re.sub(r'（[^）]*）', '', text)

        delimiters = ['、', '，', ',', '；', ';', '\n']
        herbs = [text]
        for delimiter in delimiters:
            new_herbs = []
            for herb in herbs:
                new_herbs.extend(herb.split(delimiter))
            herbs = new_herbs

        herbs = [h.strip() for h in herbs if h.strip()]
        herbs = [re.sub(r'^\d+[\.\、]', '', h).strip() for h in herbs]

        exclude = {'推荐', '建议', '使用', '等', ''}
        herbs = [h for h in herbs if h not in exclude and len(h) >= 2]

        # Remove duplicates
        seen = set()
        unique = []
        for h in herbs:
            if h not in seen:
                seen.add(h)
                unique.append(h)

        return unique


class HerbRecommenderMetrics:
    """Metrics for evaluation."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.k_values = [5, 10, 15, 20, 30]
        self.all_precisions = {k: [] for k in self.k_values}
        self.all_recalls = {k: [] for k in self.k_values}
        self.num_samples = 0

    @staticmethod
    def precision_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
        if k <= 0 or not recommended:
            return 0.0
        top_k = recommended[:k]
        return sum(1 for item in top_k if item in relevant) / k

    @staticmethod
    def recall_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
        if not relevant or k <= 0 or not recommended:
            return 0.0
        top_k = recommended[:k]
        return sum(1 for item in top_k if item in relevant) / len(relevant)

    def evaluate_single(self, recommended: List[str], relevant: Set[str]) -> Dict:
        results = {}
        for k in self.k_values:
            results[k] = {
                'precision': self.precision_at_k(recommended, relevant, k),
                'recall': self.recall_at_k(recommended, relevant, k)
            }
        return results

    def update(self, recommended: List[str], relevant: Set[str]):
        metrics = self.evaluate_single(recommended, relevant)
        for k in self.k_values:
            self.all_precisions[k].append(metrics[k]['precision'])
            self.all_recalls[k].append(metrics[k]['recall'])
        self.num_samples += 1

    def get_aggregate_metrics(self) -> Dict:
        return {
            'precision': {k: np.mean(self.all_precisions[k]) for k in self.k_values},
            'recall': {k: np.mean(self.all_recalls[k]) for k in self.k_values},
            'num_samples': self.num_samples
        }

    def print_summary(self, prefix: str = ""):
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
    """Main evaluation."""
    print("""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                                                                            ║
    ║         TCM Herb Recommendation with MUSE Embeddings                      ║
    ║              Multilingual Universal Sentence Encoder                      ║
    ║                                                                            ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """)

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("❌ Error: DEEPSEEK_API_KEY not set")
        return

    config = {
        'prescriptions_path': '../data/PTM/data/prescriptions.txt',
        'knowledge_path': '../data/herb-knowledge.csv',
        'model': 'deepseek-chat',
        'num_samples': 20,
        'vocab_size': 200,
        'max_recommendations': 30,
        'rag_top_k': 5,
        'similarity_threshold': 0.3
    }

    print("Configuration:")
    print(f"  Embeddings: MUSE (Multilingual Universal Sentence Encoder)")
    print(f"  LLM: {config['model']} (DeepSeek API)")
    print(f"  RAG top-K: {config['rag_top_k']}")
    print(f"  Similarity threshold: {config['similarity_threshold']}")
    print()

    # Initialize MUSE RAG
    rag_system = HerbKnowledgeRAGWithMUSE(config['knowledge_path'])

    # Load dataset
    dataset = PTMHerbDataset(config['prescriptions_path'])
    herb_vocabulary = dataset.get_top_herbs(config['vocab_size'])

    # Initialize predictor
    predictor = PTMHerbPredictorWithMUSE(
        herb_vocabulary=herb_vocabulary,
        rag_system=rag_system,
        model=config['model']
    )

    # Evaluate
    metrics = HerbRecommenderMetrics()
    import random
    random.seed(42)

    total_time = 0
    num_samples = min(config['num_samples'], len(dataset))

    print("\n" + "=" * 80)
    print("Running Evaluation")
    print("=" * 80)

    for i in range(num_samples):
        print(f"\nSample {i + 1}/{num_samples}")
        print('─' * 80)

        sample = dataset.get_sample(i)
        symptoms = sample['symptoms']
        ground_truth = sample['ground_truth']

        print(f"Symptoms: {symptoms[:80]}...")

        start = time.time()
        recommendations = predictor.predict(
            symptoms,
            top_k=config['max_recommendations'],
            rag_top_k=config['rag_top_k'],
            similarity_threshold=config['similarity_threshold']
        )
        total_time += time.time() - start

        print(f"Top 10: {recommendations[:10]}")
        metrics.update(recommendations, ground_truth)

    metrics.print_summary("MUSE-RAG")
    print(f"\n⏱️ Total time: {total_time:.2f}s ({total_time/num_samples:.2f}s/sample)")
    print("\n✅ Evaluation Complete!")


if __name__ == "__main__":
    main()
