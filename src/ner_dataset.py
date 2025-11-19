"""
TCM NER Dataset handler for BIO-formatted text.
"""
from typing import List, Tuple, Dict
import random


class TCMNERDataset:
    """
    Handler for Traditional Chinese Medicine Named Entity Recognition Dataset.

    Format: BIO tagging
    Labels:
        - B-SYM, I-SYM: Symptoms
        - B-CAU, I-CAU: Causes
        - B-HER, I-HER: Herbs/Medicine
        - B-PRE, I-PRE: Prescriptions
        - B-EFF, I-EFF: Effects
        - O: Other
    """

    LABEL_MAP = {
        'B-SYM': 'Symptom (Beginning)',
        'I-SYM': 'Symptom (Inside)',
        'B-CAU': 'Cause (Beginning)',
        'I-CAU': 'Cause (Inside)',
        'B-HER': 'Herb/Medicine (Beginning)',
        'I-HER': 'Herb/Medicine (Inside)',
        'B-PRE': 'Prescription (Beginning)',
        'I-PRE': 'Prescription (Inside)',
        'B-EFF': 'Effect (Beginning)',
        'I-EFF': 'Effect (Inside)',
        'O': 'Other'
    }

    def __init__(self, data_path: str = None):
        """
        Initialize TCM NER dataset.

        Args:
            data_path: Path to BIO-formatted dataset file
        """
        self.data_path = data_path
        self.sentences = []

        if data_path:
            self.load_data(data_path)
        else:
            # Generate sample data for demonstration
            self.generate_sample_data()

    def load_data(self, path: str):
        """
        Load BIO-formatted data from file.

        Format:
        character label
        character label
        (blank line separates sentences)
        """
        with open(path, 'r', encoding='utf-8') as f:
            current_sentence = []
            current_labels = []

            for line in f:
                line = line.strip()
                if not line:
                    # End of sentence
                    if current_sentence:
                        self.sentences.append({
                            'text': ''.join(current_sentence),
                            'chars': current_sentence,
                            'labels': current_labels
                        })
                        current_sentence = []
                        current_labels = []
                else:
                    parts = line.split()
                    if len(parts) == 2:
                        char, label = parts
                        current_sentence.append(char)
                        current_labels.append(label)

            # Add last sentence if exists
            if current_sentence:
                self.sentences.append({
                    'text': ''.join(current_sentence),
                    'chars': current_sentence,
                    'labels': current_labels
                })

    def generate_sample_data(self):
        """Generate sample TCM NER data for demonstration."""
        samples = [
            {
                'text': '患者出现头痛发热症状，建议服用银翘解毒片进行治疗。',
                'entities': [
                    ('头痛', 'SYM'),
                    ('发热', 'SYM'),
                    ('银翘解毒片', 'HER')
                ]
            },
            {
                'text': '因风寒感冒导致咳嗽流鼻涕，可使用板蓝根颗粒清热解毒。',
                'entities': [
                    ('风寒感冒', 'CAU'),
                    ('咳嗽', 'SYM'),
                    ('流鼻涕', 'SYM'),
                    ('板蓝根颗粒', 'HER'),
                    ('清热解毒', 'EFF')
                ]
            },
            {
                'text': '腹痛腹泻明显，处方藿香正气丸以理气和中。',
                'entities': [
                    ('腹痛', 'SYM'),
                    ('腹泻', 'SYM'),
                    ('藿香正气丸', 'HER'),
                    ('理气和中', 'EFF')
                ]
            },
            {
                'text': '失眠多梦心悸，推荐服用安神补脑液养心安神。',
                'entities': [
                    ('失眠', 'SYM'),
                    ('多梦', 'SYM'),
                    ('心悸', 'SYM'),
                    ('安神补脑液', 'HER'),
                    ('养心安神', 'EFF')
                ]
            },
            {
                'text': '由于肝火旺盛引起口苦口干，使用龙胆泻肝丸清肝泻火。',
                'entities': [
                    ('肝火旺盛', 'CAU'),
                    ('口苦', 'SYM'),
                    ('口干', 'SYM'),
                    ('龙胆泻肝丸', 'HER'),
                    ('清肝泻火', 'EFF')
                ]
            }
        ]

        for sample in samples:
            text = sample['text']
            entities = sample['entities']

            # Create BIO tags
            chars = list(text)
            labels = ['O'] * len(chars)

            for entity_text, entity_type in entities:
                # Find entity position
                start = text.find(entity_text)
                if start != -1:
                    end = start + len(entity_text)
                    labels[start] = f'B-{entity_type}'
                    for i in range(start + 1, end):
                        labels[i] = f'I-{entity_type}'

            self.sentences.append({
                'text': text,
                'chars': chars,
                'labels': labels
            })

    def get_sample(self, index: int = None) -> Dict:
        """
        Get a sample from the dataset.

        Args:
            index: Index of sample, random if None

        Returns:
            Dictionary containing text, chars, and labels
        """
        if index is None:
            index = random.randint(0, len(self.sentences) - 1)
        return self.sentences[index]

    def extract_entities(self, sentence: Dict) -> Dict[str, List[str]]:
        """
        Extract named entities from a sentence.

        Args:
            sentence: Sentence dictionary with text, chars, and labels

        Returns:
            Dictionary mapping entity types to entity texts
        """
        entities = {
            'SYM': [],  # Symptoms
            'CAU': [],  # Causes
            'HER': [],  # Herbs/Medicine
            'PRE': [],  # Prescriptions
            'EFF': []   # Effects
        }

        current_entity = []
        current_type = None

        for char, label in zip(sentence['chars'], sentence['labels']):
            if label.startswith('B-'):
                # Start of new entity
                if current_entity:
                    entities[current_type].append(''.join(current_entity))
                current_type = label.split('-')[1]
                current_entity = [char]
            elif label.startswith('I-') and current_entity:
                # Continuation of entity
                current_entity.append(char)
            else:
                # End of entity
                if current_entity:
                    entities[current_type].append(''.join(current_entity))
                    current_entity = []
                    current_type = None

        # Add last entity
        if current_entity:
            entities[current_type].append(''.join(current_entity))

        return entities

    def format_for_prediction(self, sentence: Dict) -> str:
        """
        Format a sentence for medicine prediction.

        Args:
            sentence: Sentence dictionary

        Returns:
            Formatted query string
        """
        entities = self.extract_entities(sentence)

        query_parts = []
        if entities['SYM']:
            query_parts.append(f"症状: {', '.join(entities['SYM'])}")
        if entities['CAU']:
            query_parts.append(f"病因: {', '.join(entities['CAU'])}")

        if query_parts:
            return "患者" + "，".join(query_parts) + "。请推荐合适的中药。"
        else:
            return sentence['text']

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


if __name__ == "__main__":
    # Test the dataset
    dataset = TCMNERDataset()

    print(f"Dataset size: {len(dataset)}")
    print("\nSample sentences:")

    for i in range(min(3, len(dataset))):
        sample = dataset.get_sample(i)
        print(f"\nSample {i + 1}:")
        print(f"Text: {sample['text']}")

        entities = dataset.extract_entities(sample)
        print("Entities:")
        for entity_type, entity_list in entities.items():
            if entity_list:
                print(f"  {entity_type}: {entity_list}")

        print(f"Query: {dataset.format_for_prediction(sample)}")
