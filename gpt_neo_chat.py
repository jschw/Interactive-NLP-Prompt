import os
os.environ["DISABLE_TELEMETRY"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys

import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

import numpy as np
import cv2
from datetime import datetime
import time


def printDirect(text):
    # Print all text at once to terminal
    print(text)

def printTw(text):
    # Print with typewriter effect
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()

        if char != "\n":
            time.sleep(0.1)
        else:
            time.sleep(1)

# model_id = "EleutherAI/gpt-neo-1.3B"
model_id = "EleutherAI/gpt-neo-2.7B"

# Load Model and Tokenizer
# device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained(model_id)
model = GPTNeoForCausalLM.from_pretrained(model_id, torch_dtype="auto", offload_state_dict=True)

printDirect(f"--> Loading model <{model_id}> complete.")

result_cache_path = "result_cache"
prompt_history = []
prompt_history.append("This is a discussion between a [human] and a [gpt].\nThe [gpt] is very nice and empathetic.\n")
max_length = 12     

# Prompt loop for GPT Neo
while True:
    prompt = input("Human >> ")
    
    if prompt == "quit": break

    if "maxLength" in prompt:
        tmp_str = prompt.split("=")
        max_length = int(tmp_str[1])
        printDirect(f"--> Max Length set to {str(max_length)}.")
        continue

    # Reset prompt history for chat mode
    if prompt == "reset":
        prompt_history.clear()
        prompt_history.append("This is a discussion between a [human] and a [gpt].\nThe [gpt] is very nice and empathetic.\n")
        continue

    if prompt == "showHistory":
        printDirect(str(prompt_history))
        continue

    # Dump entire history in text file
    if prompt == "dump":
        # Generate timestamp
        dateTimeObj = datetime.now()
        timestampStr = dateTimeObj.strftime("%d-%m-%Y-%H-%M-%S")
        with open(f"{result_cache_path}/{timestampStr}.txt", "w") as text_file:
            text_file.write('\n'.join(prompt_history))
        
        printDirect("--> History saved to text file.")
        continue

    # Preprocess input
    tokenizer_out = tokenizer(prompt, return_tensors="pt")
    input_ids = tokenizer_out.input_ids

    input_length = torch.Tensor.size(input_ids)[1]

    text_out = ""

    # Assemble all historical input
    input_str = ""
    first_cycle = True
    for line in prompt_history:
        input_str += line
        
        if first_cycle:
            first_cycle = False
            input_str += "\n"
        else: input_str += "\n###\n"

    # Append new input
    input_str += "\n[human]: " + prompt + "\n[gpt]: "

    # Preprocess input
    tokenizer_out = tokenizer(input_str, return_tensors="pt")
    input_ids = tokenizer_out.input_ids

    input_length = torch.Tensor.size(input_ids)[1]

    text_out = model.generate(input_ids,
                                max_new_tokens=max_length,
                                temperature=0.3,
                                do_sample=True,
                                top_p=0.5,
                                repetition_penalty=20.0,
                                pad_token_id=tokenizer.eos_token_id,
                                eos_token_id=21017)

    # Decode result
    text_out_decoded = tokenizer.decode(text_out[0])

    # Remove input from result
    text_out_decoded = text_out_decoded.replace(input_str, "")
    text_out_decoded = text_out_decoded.replace("###", "")
    prompt_history.append("[human]: " + prompt + "\n[gpt]: " + text_out_decoded)

    sys.stdout.write("GPT >> ")
    sys.stdout.flush()
    printTw(text_out_decoded)
    printDirect(" ")
