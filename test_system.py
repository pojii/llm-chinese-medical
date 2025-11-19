"""
Quick test script to verify the system setup.
"""
import sys
sys.path.insert(0, './src')

def test_knowledge_graph():
    """Test knowledge graph loading."""
    print("Testing Knowledge Graph...")
    try:
        from knowledge_graph import MedicalKnowledgeGraph
        kg = MedicalKnowledgeGraph("./data/medical.json")

        # Test search
        context = kg.search_relevant_context("感冒头痛", top_k=2)
        print(f"✓ KG loaded successfully with {len(kg.entities)} entities")
        print(f"✓ Context search working: {len(context)} characters retrieved")
        return True
    except Exception as e:
        print(f"✗ KG test failed: {e}")
        return False


def test_ner_dataset():
    """Test NER dataset."""
    print("\nTesting NER Dataset...")
    try:
        from ner_dataset import TCMNERDataset
        dataset = TCMNERDataset()

        sample = dataset.get_sample(0)
        entities = dataset.extract_entities(sample)

        print(f"✓ NER dataset loaded with {len(dataset)} samples")
        print(f"✓ Entity extraction working")
        print(f"  Sample text: {sample['text'][:50]}...")
        return True
    except Exception as e:
        print(f"✗ NER test failed: {e}")
        return False


def test_llm_predictor():
    """Test LLM predictor."""
    print("\nTesting LLM Predictor...")
    try:
        from llm_predictor import MedicineLLMPredictor
        predictor = MedicineLLMPredictor(device="cpu")

        query = "患者头痛发热。"
        result = predictor.predict_without_kg(query)

        print(f"✓ LLM predictor initialized")
        print(f"✓ Prediction working: {result[:50]}...")
        return True
    except Exception as e:
        print(f"✗ LLM test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("System Component Tests")
    print("=" * 60)

    results = []
    results.append(("Knowledge Graph", test_knowledge_graph()))
    results.append(("NER Dataset", test_ner_dataset()))
    results.append(("LLM Predictor", test_llm_predictor()))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n✓ All tests passed! System is ready to use.")
        print("\nRun the main comparison:")
        print("  cd src && python main_comparison.py")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
