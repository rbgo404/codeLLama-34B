# Load model directly
import time
import pandas as pd
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

questions = [
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


model_name_or_path = "TheBloke/CodeLlama-34B-Python-GPTQ"

use_triton = True

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        use_safetensors=True,
        trust_remote_code=True,
        device="cuda:0",
        use_triton=use_triton,
        #inject_fused_attention=False,
        quantize_config=None)


def predict(prompt:str):
    start_time = time.perf_counter()
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    generated_ids = model.generate(**inputs, max_length=512)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    request_time = time.perf_counter() - start_time
    return {'tok_count': generated_ids.shape[1],
        'time': request_time,
        'question': prompt,
        'answer': output,
        'note': 'gptq'}


if __name__ == '__main__':
    responses = []

    for q in questions:
        responses.append(predict(q))

    df = pd.DataFrame(responses)
    df.to_csv('bench-hf-autogptq-512-w-triton.csv', index=False)

