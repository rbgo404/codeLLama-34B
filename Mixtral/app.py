# Load model directly
import time
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer



model_id = "mistralai/Mixtral-8x7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)


def predict(prompt:str):
    start_time = time.perf_counter()
    
    inputs = tokenizer(prompt, return_tensors="pt")
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    request_time = time.perf_counter() - start_time
    
    return {'tok_count': generated_ids.shape[1],
        'time': request_time,
        'question': prompt,
        'answer': output,
        'note': 'mixtral'}


if __name__ == '__main__':
    responses = []

    for idx in range(5):
        prompt = "Write a good story."
        responses.append(predict(prompt))

    df = pd.DataFrame(responses)
    df.to_csv('bench-mixtral.csv', index=False)