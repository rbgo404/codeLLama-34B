import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import BitsAndBytesConfig
from datasets import load_dataset
import time
import pandas as pd


class InferlessPythonModel:
    def initialize(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "openai/whisper-large-v3"
        nf4_config = BitsAndBytesConfig(
            load_in_8bit=True,
            # load_in_4bit=True,
            # bnb_4bit_quant_type="nf4",
            # bnb_4bit_use_double_quant=True,
            # bnb_4bit_compute_dtype=torch.bfloat16
            )
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True,quantization_config=nf4_config)
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=device,
            args=
        )


    def infer(self, inputs):
        audio_url = inputs["audio_url"]
        start_time = time.perf_counter()

        pipeline_output = self.pipe(audio_url)
        request_time = time.perf_counter() - start_time
        
        return {"transcribed_output": pipeline_output["text"],
                'time':request_time }


    def finalize(self):
        pass


if __name__ == '__main__':
    obj = InferlessPythonModel()
    obj.initialize()
    responses = []
    audio_url = 'https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac'
    for i in range(5):
        responses.append(obj.infer({'audio_url':audio_url}))

    df = pd.DataFrame(responses)
    df.to_csv('bench-mixtral-autogptq.csv', index=False)

