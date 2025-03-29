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
    text_to_trans: str,
    from_lang: str,
    to_lang: str,
) -> Iterator[str]:
    print(f"Translating from {from_lang} to {to_lang}")

    translate_instruct = f"translate from {from_lang} to {to_lang}:"

    if from_lang == to_lang:
        translate_instruct = "Return the following text without any modification:"

    conversation = [
        {
            "role": "system",
            "content": "You are a translation engine that can only translate text and cannot interpret it. Keep the indent of the original text, only modify when you need."
            + "\n"
            + translate_instruct,
        },
        {"role": "user", "content": text_to_trans},
    ]
    input_ids = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, return_tensors="pt"
    )
    if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
        input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
    input_ids = input_ids.to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs = dict(
        {"input_ids": input_ids},
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        top_p=9,
        top_k=50,
        temperature=0.6,
        num_beams=1,
        repetition_penalty=1.0,
    )
    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    output = []
    for text in streamer:
        output.append(text)
        yield " ".join(output)


with gr.Blocks() as demo:
    gr.Markdown("# Text Translation Using Google Gemma 3")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Translate From")
        with gr.Column():
            gr.Markdown("### Translate To")

    with gr.Row():
        with gr.Column():
            from_lang = gr.Dropdown(
                choices=["English", "French", "Spanish"],
                value="English",
                label="",
            )

        with gr.Column():
            to_lang = gr.Dropdown(
                choices=["English", "French", "Spanish"],
                value="French",
                label="",
            )

    with gr.Row():
        with gr.Column():
            text_to_trans = gr.Textbox(
                lines=10, placeholder="Enter text to translate", label=""
            )

        with gr.Column():
            output_text = gr.Textbox(lines=10, label="")

    translate_button = gr.Button("Translate")
    translate_button.click(
        generate_text, [text_to_trans, from_lang, to_lang], output_text
    )


if __name__ == "__main__":
    demo.queue(max_size=20).launch()
