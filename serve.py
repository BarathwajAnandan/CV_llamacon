import os
from vllm import LLM, SamplingParams

#Read in our example file
def read_file_to_string(file_path):
    try:
        with open(file_path, "r") as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return "File_Path_Error"

#Please remember to set `attn_temperature_tuning` to `True` for best long context performance
def load_llm():
    llm = LLM(
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        enforce_eager=False,
        tensor_parallel_size=4,
        max_model_len=200000,
        override_generation_config= {
        "attn_temperature_tuning": True,
        }
    )

    return llm
llm = load_llm()
file_content = read_file_to_string("/root/book.txt")
PROMPT = f"""Write a couple of paragraphs about Anne's house and her environs\n\n\n{file_content} """
print("Showing long content")
if len(file_content) > 100:
    print(file_content[:100])
else:
    print(file_content)

conversations = [
    [
        {
            "role": "user",
            "content": PROMPT
        }
    ],
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=1, top_p=0.95, max_tokens=4000)

# Remember to use `chat` function and not `generate` :)
outputs = llm.chat(conversations, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f" Generated text: {generated_text}")
    