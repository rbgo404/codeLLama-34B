from vllm import LLM, SamplingParams
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

        self.sampling_params = SamplingParams(
                                                temperature=1.0,
                                                top_p=1,
                                                max_tokens=512)
        self.llm = LLM(model="TheBloke/meditron-70B-AWQ", quantization="awq", dtype="float16")

    def infer(self, inputs):
        prompts = inputs["prompt"]
        start_time = time.perf_counter()
        result = self.llm.generate(prompts, self.sampling_params)
        request_time = time.perf_counter() - start_time
        result_output = [output.outputs[0].text for output in result]

        return {'tok_count': len([output.outputs[0].token_ids for output in result][0]),
            'time': request_time,
            'question': prompts,
            'answer': result_output[0],
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
    df.to_csv('bench-meditron-vllm.csv', index=False)

