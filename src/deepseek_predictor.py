"""
Improved LLM predictor specifically optimized for DeepSeek-R1-Distill-Llama-8B.
"""
from llm_predictor import MedicineLLMPredictor, HAS_TORCH
if HAS_TORCH:
    import torch


class DeepSeekMedicinePredictor(MedicineLLMPredictor):
    """
    Optimized predictor for DeepSeek-R1-Distill-Llama-8B model.
    Uses proper chat formatting and optimized generation parameters.
    """

    def __init__(self, device: str = "cuda", load_in_8bit: bool = False, load_in_4bit: bool = False):
        """
        Initialize DeepSeek predictor.

        Args:
            device: Device to run on ('cuda' recommended)
            load_in_8bit: Load model in 8-bit quantization (saves ~50% VRAM)
            load_in_4bit: Load model in 4-bit quantization (saves ~75% VRAM)
        """
        super().__init__(
            model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            device=device,
            max_length=512,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit
        )

        # Set pad token if not set
        if HAS_TORCH and self.tokenizer and not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def predict_without_kg(self, query: str) -> str:
        """
        Predict medicine recommendation without KG using chat format.

        Args:
            query: Patient query with symptoms

        Returns:
            Medicine recommendation
        """
        if self.model is None:
            return self._mock_prediction(query, use_kg=False)

        messages = [
            {
                "role": "system",
                "content": "你是一位专业的中医药专家。根据患者症状，推荐1-2种最合适的中药。只返回药物名称，用顿号(、)分隔，不要添加任何其他解释或标点。"
            },
            {
                "role": "user",
                "content": query
            }
        ]

        try:
            # Apply chat template
            if hasattr(self.tokenizer, 'apply_chat_template'):
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback format
                prompt = f"<|system|>\n{messages[0]['content']}\n<|user|>\n{messages[1]['content']}\n<|assistant|>\n"

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            # Move to device (handle device_map="auto" case)
            if not self.uses_device_map:
                inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=30,
                    num_return_sequences=1,
                    temperature=0.3,
                    top_p=0.85,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Decode only the new tokens
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()

            # Clean up response - take only the first line
            response = response.split('\n')[0].strip()

            # Remove common prefixes
            for prefix in ["推荐中药：", "推荐：", "建议：", "可以服用"]:
                if response.startswith(prefix):
                    response = response[len(prefix):].strip()

            return response if response else "无法生成推荐"

        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return self._mock_prediction(query, use_kg=False)

    def predict_with_kg(self, query: str, kg_context: str) -> str:
        """
        Predict medicine recommendation with KG context using chat format.

        Args:
            query: Patient query with symptoms
            kg_context: Relevant medical knowledge from KG

        Returns:
            Medicine recommendation
        """
        if self.model is None:
            return self._mock_prediction(query, use_kg=True, context=kg_context)

        # Truncate context if too long
        if len(kg_context) > 800:
            kg_context = kg_context[:800] + "..."

        messages = [
            {
                "role": "system",
                "content": "你是一位专业的中医药专家。根据提供的医学知识和患者症状，推荐1-2种最合适的中药。只返回药物名称，用顿号(、)分隔，不要添加任何其他解释或标点。"
            },
            {
                "role": "user",
                "content": f"""医学知识参考：
{kg_context}

患者情况：
{query}

请根据以上信息推荐中药："""
            }
        ]

        try:
            # Apply chat template
            if hasattr(self.tokenizer, 'apply_chat_template'):
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback format
                prompt = f"<|system|>\n{messages[0]['content']}\n<|user|>\n{messages[1]['content']}\n<|assistant|>\n"

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,  # Longer for KG context
                padding=True
            )
            # Move to device (handle device_map="auto" case)
            if not self.uses_device_map:
                inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=30,
                    num_return_sequences=1,
                    temperature=0.3,
                    top_p=0.85,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Decode only the new tokens
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()

            # Clean up response - take only the first line
            response = response.split('\n')[0].strip()

            # Remove common prefixes
            for prefix in ["推荐中药：", "推荐：", "建议：", "可以服用", "根据", "基于"]:
                if response.startswith(prefix):
                    response = response[len(prefix):].strip()

            return response if response else "无法生成推荐"

        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return self._mock_prediction(query, use_kg=True, context=kg_context)


if __name__ == "__main__":
    # Test the DeepSeek predictor
    print("Testing DeepSeek Medicine Predictor...")
    print("=" * 80)

    predictor = DeepSeekMedicinePredictor(device="cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu")

    # Test queries
    queries = [
        "患者症状: 头痛, 发热。请推荐合适的中药。",
        "患者症状: 咳嗽, 流鼻涕，病因: 风寒感冒。请推荐合适的中药。",
        "患者症状: 失眠, 多梦, 心悸。请推荐合适的中药。"
    ]

    for i, query in enumerate(queries):
        print(f"\nTest {i+1}:")
        print(f"Query: {query}")
        result = predictor.predict_without_kg(query)
        print(f"Prediction: {result}")
        print("-" * 80)
