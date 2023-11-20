from transformers import AutoTokenizer
import transformers
import torch


class InferlessPythonModel:
    def initialize(self):
        tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
        pipeline = transformers.pipeline(
            "text-generation",
            model="codellama/CodeLlama-7b-hf",
            torch_dtype=torch.float16,
            device_map="auto",
        )
    
    def infer(self, inputs):
        print("inputs[prompt] -->", inputs["prompt"], flush=True)
        prompts = inputs["prompt"]
        sequences = pipeline(
            prompts,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=100,
        )
        for seq in sequences:
            return {"result": seq['generated_text']}




    def finalize(self, args):
        pass