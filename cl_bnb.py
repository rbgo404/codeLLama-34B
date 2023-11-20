# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch
import time

import pandas as pd



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

model_id = "codellama/CodeLlama-34b-Python-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_compute_dtype=torch.bfloat16
)
model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)


def predict(prompt:str):
    start_time = time.perf_counter()
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    generated_ids = model_nf4.generate(**inputs, max_length=200)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    request_time = time.perf_counter() - start_time
    return {'tok_count': generated_ids.shape[1],
        'time': request_time,
        'question': prompt,
        'answer': output,
        'note': 'nf4 4bit quantization bitsandbytes'}


if __name__ == '__main__':
    counter = 1
    responses = []

    for q in questions:
        responses.append(predict(q))
        #counter += 1

    df = pd.DataFrame(responses)
    df.to_csv('bench-hf-bb.csv', index=False)

