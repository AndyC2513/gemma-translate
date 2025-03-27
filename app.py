import torch
from huggingface_hub import login
from collections.abc import Iterator
from transformers import Gemma3ForCausalLM, AutoTokenizer, TextIteratorStreamer
import time
import spaces
from threading import Thread
import gradio as gr
import os

TOKEN = os.getenv("TOKEN")
login(token=TOKEN)
MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = 4096

start_time = time.time()
model = Gemma3ForCausalLM.from_pretrained(
    "google/gemma-3-4b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
).eval()

tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-3-4b-it",
)
load_time = time.time() - start_time
print(f"Model loaded in {load_time:.2f} seconds")


@spaces.GPU
def generate_text(
    message: str,
    chat_history: list[dict],
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
) -> Iterator[str]:
    conversation = [*chat_history, {"role": "user", "content": message}]
    input_ids = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, return_tensors="pt"
    )
    if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
        input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
    input_ids = input_ids.to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, timeout=30.0, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs = dict(
        {"input_ids": input_ids},
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=1,
        repetition_penalty=repetition_penalty,
    )
    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()
    
    output = []
    for text in streamer:
        output.append(text)
        yield " ".join(output)