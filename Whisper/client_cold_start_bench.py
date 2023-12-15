import requests, json, time
from transformers import AutoTokenizer
import pandas as pd
import argparse
import requests
from pydub import AudioSegment
from io import BytesIO

parser = argparse.ArgumentParser(description='Run LLM inference.')
parser.add_argument('--model_name', type=str, required=True, help='Model name for inference.')
#parser.add_argument('--quantization', type=str, required=True, help='Quantization to be use for quant.')
args = parser.parse_args()


urls = [
    "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/2.flac",
    "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac",
    "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/3.flac",
    "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/i-know-kung-fu.mp3",
    "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"
]


print("Starting cold start test for ",args.model_name.split('/')[-1])
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

def chat(url:str):
    payload = {
  "inputs": [
    {
      "data": [
        url
      ],
      "name": "audio_url",
      "shape": [
        1
      ],
      "datatype": "BYTES"
    }
  ]
}
    headers = {'Content-Type': 'application/json'}
    response = requests.post("http://localhost:8000/v2/repository/models/model_auto_gptq/unload")
    time.sleep(3)
    start = time.perf_counter()
    response = requests.post("http://localhost:8000/v2/repository/models/model_auto_gptq/load")
    start_inference_time = time.perf_counter()
    response = requests.post("http://localhost:8000/v2/models/model_auto_gptq/infer", headers=headers, data=json.dumps(payload))
    print(response.json())
    load_time = start_inference_time - start
    generated_text = response.json()['outputs'][0]['data'][0]
    print(generated_text)
    inference_time = time.perf_counter() - start_inference_time

    request_time = time.perf_counter() - start
    
    response = requests.get(url)
    audio = AudioSegment.from_file(BytesIO(response.content))
    length_in_seconds = len(audio) / 1000.0
    
    print(f"Inference time: {inference_time}, Load Time: {load_time},Total time: {request_time}")
    
    return {
        'tok_count': len(tokenizer.encode(generated_text)),
        'audio_length': length_in_seconds,
        'cold_time':load_time,
        'inference_time':inference_time,
        #'question': prompt,
        'answer': generated_text,
        'note': 'triton-auto-whisper'}


if __name__ == '__main__':
    responses = []
    for url in urls:
        responses.append(chat(url))

    df = pd.DataFrame(responses)
    df.to_csv(f'bench-cold-start-whisper.csv', index=False)
