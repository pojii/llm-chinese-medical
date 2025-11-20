"""
Recommender Dataset for Medical Knowledge Graph.
Extracts (symptom/disease -> drugs) pairs from medical.json for recommendation evaluation.
"""
import json
from typing import Dict, List, Set, Tuple, Optional


class MedicalRecommenderDataset:
    """
    Dataset for medical recommendation task.

    Each sample consists of:
    - query: symptom description or disease name
    - ground_truth: set of recommended drugs/treatments
    """

    def __init__(self, kg_path: str = "../data/medical.json"):
        """
        Initialize dataset from medical knowledge graph.

        Args:
            kg_path: Path to medical.json knowledge graph
        """
        self.kg_path = kg_path
        self.samples = []
        self.load_data()

    def load_data(self):
        """Load and process medical knowledge graph into recommendation samples."""
        print(f"Loading recommender dataset from {self.kg_path}...")

        with open(self.kg_path, 'r', encoding='utf-8') as f:
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

        # Extract recommendation samples
        for entity in data:
            disease_name = entity.get('name', '')

            # Skip if no disease name
            if not disease_name:
                continue

            # Extract ground truth drugs
            ground_truth_drugs = set()

            # From recommand_drug field
            if 'recommand_drug' in entity and entity['recommand_drug']:
                if isinstance(entity['recommand_drug'], list):
                    ground_truth_drugs.update(entity['recommand_drug'])
                elif isinstance(entity['recommand_drug'], str):
                    ground_truth_drugs.add(entity['recommand_drug'])

            # From common_drug field
            if 'common_drug' in entity and entity['common_drug']:
                if isinstance(entity['common_drug'], list):
                    ground_truth_drugs.update(entity['common_drug'])
                elif isinstance(entity['common_drug'], str):
                    ground_truth_drugs.add(entity['common_drug'])

            # Skip if no drugs available
            if not ground_truth_drugs:
                continue

            # Create sample from disease name
            self.samples.append({
                'query_type': 'disease',
                'query': disease_name,
                'disease_name': disease_name,
                'ground_truth': ground_truth_drugs,
                'entity': entity
            })

            # Create samples from symptoms (if available)
            symptoms = entity.get('symptom', [])

            if symptoms:
                # Handle both string and list formats
                if isinstance(symptoms, str):
                    symptom_list = [s.strip() for s in symptoms.split('，') if s.strip()]
                elif isinstance(symptoms, list):
                    symptom_list = symptoms
                else:
                    symptom_list = []

                # Create a sample for combined symptoms
                if symptom_list:
                    symptom_text = '、'.join(symptom_list[:5])  # Use top 5 symptoms

                    self.samples.append({
                        'query_type': 'symptom',
                        'query': symptom_text,
                        'disease_name': disease_name,
                        'ground_truth': ground_truth_drugs,
                        'entity': entity
                    })

        print(f"Loaded {len(self.samples)} recommendation samples")

        # Print statistics
        disease_queries = sum(1 for s in self.samples if s['query_type'] == 'disease')
        symptom_queries = sum(1 for s in self.samples if s['query_type'] == 'symptom')
        print(f"  - Disease-based queries: {disease_queries}")
        print(f"  - Symptom-based queries: {symptom_queries}")

    def get_sample(self, index: int) -> Dict:
        """
        Get a sample by index.

        Args:
            index: Sample index

        Returns:
            Sample dictionary
        """
        return self.samples[index]

    def format_query(self, sample: Dict) -> str:
        """
        Format sample into query text for LLM.

        Args:
            sample: Sample dictionary

        Returns:
            Formatted query string
        """
        if sample['query_type'] == 'disease':
            return f"患者诊断为{sample['query']}，请推荐适合的药物或治疗方法。"
        else:  # symptom
            return f"患者出现以下症状：{sample['query']}，请推荐适合的药物或治疗方法。"

    def get_ground_truth(self, sample: Dict) -> Set[str]:
        """
        Get ground truth drugs for a sample.

        Args:
            sample: Sample dictionary

        Returns:
            Set of ground truth drug names
        """
        return sample['ground_truth']

    def filter_by_query_type(self, query_type: str) -> List[int]:
        """
        Get indices of samples with specific query type.

        Args:
            query_type: 'disease' or 'symptom'

        Returns:
            List of sample indices
        """
        return [i for i, s in enumerate(self.samples) if s['query_type'] == query_type]

    def filter_by_min_drugs(self, min_drugs: int = 3) -> List[int]:
        """
        Get indices of samples with minimum number of ground truth drugs.

        Args:
            min_drugs: Minimum number of drugs required

        Returns:
            List of sample indices
        """
        return [i for i, s in enumerate(self.samples) if len(s['ground_truth']) >= min_drugs]

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        """Get sample by index (supports slicing)."""
        return self.samples[index]


def extract_drugs_from_text(text: str) -> List[str]:
    """
    Extract drug names from LLM response text.

    Args:
        text: LLM response text

    Returns:
        List of extracted drug names (ordered)
    """
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

    # Remove common prefixes and suffixes
    prefix_patterns = [
        r'^建议使用?',
        r'^推荐使用?',
        r'^可以使用?',
        r'^使用',
        r'^服用',
        r'^选用',
        r'^采用',
        r'^\d+[\.\、]',  # Remove numbering like "1.", "1、"
    ]

    suffix_patterns = [
        r'进行治疗$',
        r'治疗$',
        r'为宜$',
        r'较好$',
        r'等$',
    ]

    cleaned_drugs = []
    for drug in drugs:
        # Remove prefixes
        for pattern in prefix_patterns:
            drug = re.sub(pattern, '', drug)

        # Remove suffixes
        for pattern in suffix_patterns:
            drug = re.sub(pattern, '', drug)

        drug = drug.strip()
        cleaned_drugs.append(drug)

    drugs = cleaned_drugs

    # Filter out common phrases and very short items
    exclude_phrases = {
        '基于LLM直接推荐', '基于知识图谱推荐', '推荐', '建议', '可以',
        '使用', '服用', '等', '以下', '药物', '治疗', '方法', '如下',
        '包括', '有', '为', '是', '的', '在', '及', '与', '或', '和',
        '请', '应', '需', '要', '还', '也', '', '进行'
    }

    drugs = [
        d for d in drugs
        if d and d not in exclude_phrases and len(d) >= 2
    ]

    # Remove duplicates while preserving order
    seen = set()
    unique_drugs = []
    for drug in drugs:
        if drug not in seen:
            seen.add(drug)
            unique_drugs.append(drug)

    return unique_drugs


if __name__ == "__main__":
    # Test the dataset
    print("Testing Medical Recommender Dataset")
    print("=" * 80)

    dataset = MedicalRecommenderDataset()

    # Test sample retrieval
    print("\n\nTest 1: Sample Retrieval")
    print("─" * 80)

    sample = dataset.get_sample(0)
    print(f"Query Type: {sample['query_type']}")
    print(f"Query: {sample['query']}")
    print(f"Disease: {sample['disease_name']}")
    print(f"Ground Truth Drugs ({len(sample['ground_truth'])}): {list(sample['ground_truth'])[:5]}...")

    # Test query formatting
    print("\n\nTest 2: Query Formatting")
    print("─" * 80)
    query = dataset.format_query(sample)
    print(f"Formatted Query: {query}")

    # Test filtering
    print("\n\nTest 3: Dataset Filtering")
    print("─" * 80)

    disease_indices = dataset.filter_by_query_type('disease')
    symptom_indices = dataset.filter_by_query_type('symptom')
    print(f"Disease-based samples: {len(disease_indices)}")
    print(f"Symptom-based samples: {len(symptom_indices)}")

    min_3_drugs = dataset.filter_by_min_drugs(3)
    print(f"Samples with ≥3 drugs: {len(min_3_drugs)}")

    # Test drug extraction
    print("\n\nTest 4: Drug Extraction from Text")
    print("─" * 80)

    test_text = "建议使用银翘解毒片、连花清瘟胶囊和板蓝根颗粒进行治疗。"
    extracted = extract_drugs_from_text(test_text)
    print(f"Input: {test_text}")
    print(f"Extracted: {extracted}")

    # Show some example samples
    print("\n\nTest 5: Example Samples")
    print("=" * 80)

    for i in range(min(3, len(dataset))):
        sample = dataset.get_sample(i)
        print(f"\nSample {i+1}:")
        print(f"  Type: {sample['query_type']}")
        print(f"  Query: {sample['query'][:50]}...")
        print(f"  Ground Truth: {list(sample['ground_truth'])[:3]}...")
