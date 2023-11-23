import time
import pandas as pd
from vllm import SamplingParams, LLM
import argparse

parser = argparse.ArgumentParser(description='Run LLM inference.')
parser.add_argument('--model', type=str, required=True, help='Model name for inference.')
parser.add_argument('--quantization', type=str, required=True, help='Quantization to be use for quant.')

# Parse the command-line arguments
args = parser.parse_args()

questions = [
    'def factorial(int n):',
    'Write a Python program that prints "Hello, World!" to the console.',
    'Write a function that takes two parameters and returns their sum.',
    'Write a function to calculate the factorial of a given number.',
    'Write a function to determine if a given string is a palindrome.',
    'Write a program that prints the numbers from 1 to 100. But for multiples of three, print "Fizz" instead of the number, and for the multiples of five, print "Buzz." For numbers that are multiples of both three and five, print "FizzBuzz."',
    'Implement a function to reverse a singly linked list.',
    'Given an unsorted array of integers, find the length of the longest increasing subsequence.',
    'Given a sequence of matrices, find the most efficient way to multiply these matrices.',
    'Given a binary search tree, write a function to find the Kth smallest element.',
    'Given a weighted graph and two vertices, find the shortest path between them using Dijkstra\'s algorithm.'
]



def generate(question, llm, note=None):
    response = {'question': question, 'note': note}
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1,
        max_tokens=512,
    )
    
    start = time.perf_counter()
    result = llm.generate(question, sampling_params)
    request_time = time.perf_counter() - start

    for output in result:
        response['tok_count'] = len(output.outputs[0].token_ids)
        response['time'] = request_time
        response['answer'] = output.outputs[0].text
    
    return response

if __name__ == '__main__':
    llm = LLM(
        model=args.model,
        #tokenizer=args.tokenizer,
        #quantization=args.quantization,
        #tensor_parallel_size=args.tensor_parallel_size,
        #max_num_seqs=args.batch_size,
        #max_num_batched_tokens=args.batch_size * args.input_len,
        #max_model_len=256,
        #trust_remote_code=args.trust_remote_code,
        #dtype='float16',
        )
    responses = []

    for q in questions:
        response = generate(question=q, llm=llm, note='vLLM')
        responses.append(response)
    
    df = pd.DataFrame(responses)
    df.to_csv('bench-vllm.csv', index=False)
