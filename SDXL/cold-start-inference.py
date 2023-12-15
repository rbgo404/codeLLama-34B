import requests, json, time
from transformers import AutoTokenizer
import pandas as pd
import argparse
import requests
from pydub import AudioSegment
from io import BytesIO


prompts = [
    "A tiger running after a car.",
    "A very close look into the face of Rhino.",
    "A beautiful view of hill station",
    "Car view from inside during rain.",
    "People eating food"
]


def chat(url:str):
    payload = {
  "inputs": [
    {
      "data": [
        url
      ],
      "name": "prompt",
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
    load_time = start_inference_time - start
    generated_iamge = response.json()['outputs'][0]['data'][0]

    inference_time = time.perf_counter() - start_inference_time

    request_time = time.perf_counter() - start
    
    response = requests.get(url)
    audio = AudioSegment.from_file(BytesIO(response.content))
    length_in_seconds = len(audio) / 1000.0
    
    print(f"Inference time: {inference_time}, Load Time: {load_time},Total time: {request_time}")
    
    return {
        'cold_time':load_time,
        'inference_time':inference_time}


if __name__ == '__main__':
    responses = []
    for prompt in prompts:
        responses.append(chat(prompt))

    df = pd.DataFrame(responses)
    df.to_csv(f'bench-cold-start-sdxl.csv', index=False)
