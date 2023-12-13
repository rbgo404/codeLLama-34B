from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
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
        model_name_or_path = "TheBloke/meditron-7B-GPTQ"

        # To use a different branch, change revision
        # For example: revision="gptq-4bit-32g-actorder_True"
        self.model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
                model_basename="model",
                use_safetensors=True,
                trust_remote_code=False,
                device="cuda:0",
                use_triton=False,
                disable_exllama=True,
                disable_exllamav2=True,
                quantize_config=None)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, trust_remote_code=False)

    def infer(self, inputs):
        prompts = inputs["prompt"]
        start_time = time.perf_counter()
        input_ids = self.tokenizer(prompts, return_tensors='pt').input_ids.cuda()
        output = self.model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
        text = self.tokenizer.decode(output[0])
        request_time = time.perf_counter() - start_time

        return {'tok_count': output.shape[1],
            'time': request_time,
            'question': prompts,
            'answer': text,
            'note': 'mixtral-autogptq'}


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
    df.to_csv('bench-meditron-autogptq.csv', index=False)
