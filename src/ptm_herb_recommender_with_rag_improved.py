"""
PTM Herb Recommender with IMPROVED RAG
Fixes:
1. Similarity threshold filtering
2. Reduced RAG knowledge (top-5 instead of top-10)
3. Simplified prompt format
4. Better herb name handling
"""
import re
import os
import csv
import numpy as np
from typing import List, Set, Dict
from collections import Counter

# Import base classes from RAG version
from ptm_herb_recommender_with_rag import (
    HerbKnowledgeRAG,
    PTMHerbDataset,
    HerbRecommenderMetrics
)


class ImprovedHerbKnowledgeRAG(HerbKnowledgeRAG):
    """
    Improved RAG with similarity threshold filtering.
    Only returns high-quality relevant knowledge.
    """

    def retrieve_relevant_knowledge(
        self,
        query: str,
        top_k: int = 5,  # Reduced from 10
        similarity_threshold: float = 0.3  # NEW: Filter low-quality matches
    ) -> List[Dict]:
        """
        Retrieve top K most relevant herb knowledge with similarity filtering.

        Args:
            query: Symptom description
            top_k: Number of entries to retrieve
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            List of most relevant knowledge entries
        """
        # Encode query using OpenAI API
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        )
        query_embedding = np.array(response.data[0].embedding)

        # Calculate cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Filter by threshold FIRST
        valid_indices = [i for i, sim in enumerate(similarities) if sim >= similarity_threshold]

        if not valid_indices:
            print(f"⚠️ No knowledge entries above threshold {similarity_threshold}")
            return []

        # Get top K from valid indices
        valid_similarities = similarities[valid_indices]
        top_indices_in_valid = np.argsort(valid_similarities)[::-1][:top_k]
        top_indices = [valid_indices[i] for i in top_indices_in_valid]

        # Return top K entries with similarity scores
        results = []
        for idx in top_indices:
            entry = self.knowledge_entries[idx].copy()
            entry['similarity'] = float(similarities[idx])
            results.append(entry)

        print(f"✅ Retrieved {len(results)} knowledge entries (threshold={similarity_threshold})")
        if results:
            print(f"   Similarity range: {results[-1]['similarity']:.3f} - {results[0]['similarity']:.3f}")

        return results


class ImprovedPTMHerbPredictor:
    """
    Improved predictor with:
    - Simplified prompt
    - Better knowledge formatting
    - Reduced knowledge injection
    """

    def __init__(
        self,
        herb_vocabulary: List[str],
        rag_system: ImprovedHerbKnowledgeRAG,
        api_key: str = None,
        model: str = "deepseek-chat",
        use_rag: bool = True  # NEW: Can disable RAG
    ):
        """
        Initialize improved predictor.

        Args:
            herb_vocabulary: List of valid herb names
            rag_system: Improved RAG system
            api_key: DeepSeek API key
            model: Model name
            use_rag: Whether to use RAG (for A/B testing)
        """
        from openai import OpenAI

        self.herb_vocabulary = herb_vocabulary
        self.rag_system = rag_system
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.model = model
        self.use_rag = use_rag

        if not self.api_key:
            raise ValueError("API key required. Set DEEPSEEK_API_KEY environment variable.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )

        print(f"Initialized {'RAG-enhanced' if use_rag else 'baseline'} predictor with {len(herb_vocabulary)} herbs")

    def predict(
        self,
        symptoms: str,
        top_k: int = 30,
        rag_top_k: int = 5,  # Reduced from 10
        similarity_threshold: float = 0.3
    ) -> List[str]:
        """
        Predict herbs from symptoms.

        Args:
            symptoms: Symptom description
            top_k: Number of herbs to recommend
            rag_top_k: RAG knowledge entries to retrieve
            similarity_threshold: Minimum similarity for RAG

        Returns:
            Ranked list of herb names
        """
        # Create herb vocabulary string (top 200 herbs)
        herb_list_str = '、'.join(self.herb_vocabulary[:200])

        # Retrieve RAG knowledge (if enabled)
        knowledge_str = ""
        if self.use_rag:
            relevant_knowledge = self.rag_system.retrieve_relevant_knowledge(
                symptoms,
                top_k=rag_top_k,
                similarity_threshold=similarity_threshold
            )

            if relevant_knowledge:
                # Simplified knowledge format (shorter)
                knowledge_str = self._format_knowledge_simple(relevant_knowledge)

        # Create prompt
        if self.use_rag and knowledge_str:
            # RAG-enhanced prompt (simplified)
            prompt = f"""你是一个专业的中医师。根据患者症状，从给定的中药列表推荐适合的中药。

【相关中药知识】
{knowledge_str}

【患者症状】
{symptoms}

【可用中药列表】（只能从这个列表中选择）
{herb_list_str}

【要求】
1. 参考上述中药知识（如果相关）
2. 只推荐列表中的中药
3. 按优先级排序
4. 用顿号（、）分隔
5. 只输出中药名称，不要解释

【推荐中药】："""
        else:
            # Baseline prompt (no RAG)
            prompt = f"""你是一个专业的中医师。根据患者症状，从给定的中药列表推荐适合的中药。

【患者症状】
{symptoms}

【可用中药列表】（只能从这个列表中选择）
{herb_list_str}

【要求】
1. 只推荐列表中的中药
2. 按优先级排序
3. 用顿号（、）分隔
4. 只输出中药名称，不要解释
5. 推荐10-20种中药

【推荐中药】："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是专业的中医师。严格遵守给定的中药列表。"},
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

    def _format_knowledge_simple(self, knowledge_entries: List[Dict]) -> str:
        """
        Simplified knowledge format (shorter, more focused).
        """
        if not knowledge_entries:
            return ""

        formatted = []
        for i, entry in enumerate(knowledge_entries, 1):
            # Use only indication (most relevant for symptoms)
            indication = entry.get('indication', '').strip()
            if indication:
                # Truncate long indications
                if len(indication) > 100:
                    indication = indication[:100] + "..."

                formatted.append(f"{i}. {entry['pinyin']}: {indication}")

        return '\n'.join(formatted[:5])  # Max 5 entries

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
            r'^\d+[\.\、]',
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

        # Remove duplicates
        seen = set()
        unique_herbs = []
        for herb in herbs:
            if herb not in seen:
                seen.add(herb)
                unique_herbs.append(herb)

        return unique_herbs


def main():
    """Test improved RAG version."""
    print("""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                                                                            ║
    ║         IMPROVED TCM Herb Recommendation with RAG                         ║
    ║                                                                            ║
    ║  Improvements:                                                             ║
    ║  1. Similarity threshold filtering (0.3)                                   ║
    ║  2. Reduced RAG top-K (5 instead of 10)                                    ║
    ║  3. Simplified prompt format                                               ║
    ║  4. Better knowledge formatting                                            ║
    ║                                                                            ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """)

    # Check API keys
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("❌ Error: DEEPSEEK_API_KEY not set")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not set")
        return

    # Configuration
    config = {
        'prescriptions_path': '../data/PTM/data/prescriptions.txt',
        'knowledge_path': '../data/herb-knowledge.csv',
        'model': 'deepseek-chat',
        'num_samples': 20,
        'vocab_size': 200,
        'max_recommendations': 30,
        'rag_top_k': 5,  # Reduced from 10
        'similarity_threshold': 0.3  # NEW
    }

    print("Configuration:")
    print(f"  RAG top-K: {config['rag_top_k']} (reduced from 10)")
    print(f"  Similarity threshold: {config['similarity_threshold']}")
    print(f"  Test samples: {config['num_samples']}")

    # Initialize improved RAG system
    print("\n" + "=" * 80)
    print("Initializing IMPROVED RAG System")
    print("=" * 80)
    rag_system = ImprovedHerbKnowledgeRAG(config['knowledge_path'])

    # Load dataset
    print("\n" + "=" * 80)
    print("Loading Dataset")
    print("=" * 80)
    dataset = PTMHerbDataset(config['prescriptions_path'])

    # Get herb vocabulary
    herb_vocabulary = dataset.get_top_herbs(config['vocab_size'])

    # Initialize improved predictor
    print("\n" + "=" * 80)
    print("Initializing IMPROVED Predictor")
    print("=" * 80)
    predictor = ImprovedPTMHerbPredictor(
        herb_vocabulary=herb_vocabulary,
        rag_system=rag_system,
        model=config['model'],
        use_rag=True
    )

    # Initialize metrics
    metrics = HerbRecommenderMetrics()

    # Run evaluation
    print("\n" + "=" * 80)
    print("Running Evaluation")
    print("=" * 80)

    import time
    import random
    random.seed(42)

    num_samples = min(config['num_samples'], len(dataset))
    total_time = 0

    for i in range(num_samples):
        print(f"\n{'=' * 80}")
        print(f"Sample {i + 1}/{num_samples}")
        print('=' * 80)

        sample = dataset.get_sample(i)
        symptoms = sample['symptoms']
        ground_truth = sample['ground_truth']

        print(f"\nSymptoms: {symptoms[:80]}...")
        print(f"Ground Truth: {len(ground_truth)} herbs")

        # Predict
        start_time = time.time()
        recommendations = predictor.predict(
            symptoms,
            top_k=config['max_recommendations'],
            rag_top_k=config['rag_top_k'],
            similarity_threshold=config['similarity_threshold']
        )
        inference_time = time.time() - start_time
        total_time += inference_time

        print(f"Top 10 Predictions: {recommendations[:10]}")
        print(f"Time: {inference_time:.2f}s")

        # Update metrics
        metrics.update(recommendations, ground_truth)

    # Print summary
    metrics.print_summary("IMPROVED RAG")

    print(f"\n⏱️ Total time: {total_time:.2f}s ({total_time/num_samples:.2f}s/sample)")
    print("\n✅ Evaluation Complete!")


if __name__ == "__main__":
    main()
