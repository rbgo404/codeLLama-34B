from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch
import time
import pandas as pd

questions = [
        # Coding questions
        "Implement a Python function to compute the Fibonacci numbers.",
        "Write a Rust function that performs binary exponentiation.",
        "How do I allocate memory in C?",
        # Literature
        "What is the fable involving a fox and grapes?",
        "Write a story in the style of James Joyce about a trip to the Australian outback in 2083, to see robots in the beautiful desert.",
        "Who does Harry turn into a balloon?",
        # History
        "What were the major contributing factors to the fall of the Roman Empire?",
        "How did the invention of the printing press revolutionize European society?",
        "What are the effects of quantitative easing?",
        # Thoughtfulness
        "Describe the city of the future, considering advances in technology, environmental changes, and societal shifts.",
        "In a dystopian future where water is the most valuable commodity, how would society function?",
        # Math
        "What is the product of 9 and 8?",
        "If a train travels 120 kilometers in 2 hours, what is its average speed?",
        "Think through this step by step. If the sequence a_n is defined by a_1 = 3, a_2 = 5, and a_n = a_(n-1) + a_(n-2) for n > 2, find a_6.",
        # Facts
        "What are 'zombie stars' in the context of astronomy?",
        "Who were the 'Dog-Headed Saint' and the 'Lion-Faced Saint' in medieval Christian traditions?",
        "What is the story of the 'Globsters', unidentified organic masses washed up on the shores?",
    ]


class InferlessPythonModel:
    def initialize(self):
        model_id = "mistralai/Mixtral-8x7B-v0.1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)

    def infer(self, inputs):
        prompts = inputs["prompt"]
        start_time = time.perf_counter()
        inputs = self.tokenizer(prompts, return_tensors="pt").to("cuda")
        generated_ids = self.model_nf4.generate(**inputs, max_length=512)
        output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        request_time = time.perf_counter() - start_time

        return {'tok_count': generated_ids.shape[1],
            'time': request_time,
            'question': prompts,
            'answer': output,
            'note': 'mixtral-bnb'}

    def finalize(self):
        pass


if __name__ == '__main__':
    obj = InferlessPythonModel()
    obj.initialize()
    responses = []

    for question in questions:
        responses.append(obj.infer({'prompt':question})
)

    df = pd.DataFrame(responses)
    df.to_csv('bench-mixtral-bnb.csv', index=False)
