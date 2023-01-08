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

def changeMode():
    global prompt_mode
    global max_length
    global prompt_history

    prompt_history = []

    print("Interactive Prompt Mode:")
    print("1: Large Text Generation")
    print("2: Question answering")
    print("3: Chat Dialog")
    tmp_input = input(">> ")

    if tmp_input == "1":
        # Large text generation
        prompt_mode = 1
        max_length = 150
        printDirect("--> Mode set to 'Large text gen' ")

    elif tmp_input == "2":
        # Question answering
        prompt_mode = 2
        max_length = 50
        printDirect("--> Mode set to 'Question answering' ")

    elif tmp_input == "3":
        # Chat Dialog
        prompt_mode = 3
        max_length = 12
        prompt_history.append("This is a discussion between a [human] and a [gpt].\nThe [gpt] is very nice and empathetic.\n")

        printDirect("--> Mode set to 'Chat Dialog' ")

    else:
        print("No valid model number. Closing Application.")
        quit()

    printDirect(" ")

# Ask for task type
print("Choose type of AI model:")
print("1: Stable Diffusion")
print("2: GPT Neo")
tmp_input = input(">> ")

if tmp_input == "1":
    # Stable Diffusion
    task_type = 1
    printDirect("--> Type of AI task is 'Stable Diffusion' ")
elif tmp_input == "2":
    # GPT Neo
    task_type = 2
    printDirect("--> Type of AI task is 'GPT Neo' ")
else:
    print("No valid task number. Closing Application.")
    quit()

if task_type == 1:
    # Stable Diffusion
    print("Please choose a model:")
    print("1: Stable Diffusion 1.5")
    print("2: Stable Diffusion 2.1")
    model = input(">> ")

    if model == "1": model_id = "runwayml/stable-diffusion-v1-5"
    elif model == "2": model_id = "stabilityai/stable-diffusion-2-1-base"
    else:
        print("No valid model number. Closing Application.")
        quit()

    # Use the Euler scheduler here instead
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu")

    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

    pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None).to(device)
    # pipe.enable_attention_slicing()

    printDirect(f"--> Loading model <{model_id}> complete.")

elif task_type == 2:
    # GPT Neo
    print("Please choose a model (default is 2.7B):")
    print("1: GPT Neo 2.7B")
    print("2: GPT Neo 1.3B")
    model = input(">> ")

    if model == "1": model_id = "EleutherAI/gpt-neo-2.7B"
    elif model == "2": model_id = "EleutherAI/gpt-neo-1.3B"
    else:
        model_id = "EleutherAI/gpt-neo-2.7B"

    # Load Model and Tokenizer
    # device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    model = GPTNeoForCausalLM.from_pretrained(model_id, torch_dtype="auto", offload_state_dict=True)

    printDirect(f"--> Loading model <{model_id}> complete.")

result_cache_path = "result_cache"

if task_type == 1:
    # Prompt loop for Stable Diffusion

    print("Number of inference steps (default = 50):")
    inf_steps = input(">> ")
    if inf_steps == "":
        inf_steps = 50
        print("--> The number of inference steps was set to the default of 50 steps.")
    else:
        inf_steps = int(inf_steps)
        print(f"--> The number of inference steps was set to {str(inf_steps)} steps.")

    # Prompt loop
    while True:
        prompt = input(">> ")
        
        if prompt == "quit": break

        if "infSteps" in prompt:
            tmp_str = prompt.split("=")
            inf_steps = int(tmp_str[1])
            printDirect(f"--> Number of inference steps set to {str(inf_steps)}.")
            continue

        # Generate timestamp
        dateTimeObj = datetime.now()
        timestampStr = dateTimeObj.strftime("%d-%m-%Y-%H-%M-%S")

        # Start inference
        # First-time "warmup" pass (see explanation above)
        _ = pipe(prompt, num_inference_steps=1)
        image = pipe(prompt, height=512, width=512, num_inference_steps=inf_steps).images[0]

        img_np = np.asarray(image)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        # Write files to disk
        cv2.imwrite(f"{result_cache_path}/{timestampStr}.png", img_np)
        with open(f"{result_cache_path}/{timestampStr}.txt", "w") as text_file:
            print(f"Model: {model_id}", file=text_file)
            print(f"Input prompt: {prompt}", file=text_file)

        # Show generated image
        cv2.imshow("Result image", img_np)
        # Wait for a key press to exit
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

elif task_type == 2:
    # Prompt loop for GPT Neo

    changeMode()

    while True:
        prompt = input("Human >> ")
        
        if prompt == "quit": break
        if prompt == "changeMode":
            changeMode()
            continue
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

        if prompt_mode == 1:
            # Large text generation
            text_out = model.generate(input_ids,
                                        max_new_tokens=max_length,
                                        temperature=0.6,
                                        do_sample=True,
                                        top_p=1,
                                        repetition_penalty=10.0,
                                        pad_token_id=tokenizer.eos_token_id,
                                        eos_token_id=21017)
        
        elif prompt_mode == 2:
            # Question answering
            text_out = model.generate(input_ids,
                                        max_new_tokens=max_length,
                                        temperature=0.03,
                                        do_sample=False,
                                        num_beams=1,
                                        repetition_penalty=10.0,
                                        early_stopping=True,
                                        pad_token_id=tokenizer.eos_token_id,
                                        eos_token_id=21017)
        
        elif prompt_mode == 3:
            # Chat dialog
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
        if prompt_mode == 3:
            text_out_decoded = text_out_decoded.replace(input_str, "")
            text_out_decoded = text_out_decoded.replace("###", "")
            prompt_history.append("[human]: " + prompt + "\n[gpt]: " + text_out_decoded)
        else:
            text_out_decoded = text_out_decoded.replace(prompt, "")
            # text_out_decoded = "GPT >> " + text_out_decoded

        sys.stdout.write("GPT >> ")
        sys.stdout.flush()
        printTw(text_out_decoded)
        printDirect(" ")