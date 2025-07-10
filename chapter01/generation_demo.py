from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
print(torch.cuda.is_available())  # 如果返回 True，说明支持 CUDA
print(torch.version.cuda)
# 输出 CUDA 的版本
# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="mps",
    torch_dtype="auto",
    trust_remote_code=False,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=500,
    do_sample=False
)

messages = [
    {"role": "user", "content": "Create a funny joke about chickens."},
    {"role": "user", "content": "Create a funny joke about chickens."}
]

# Generate output
output = generator(messages)
print(output[0]["generated_text"])
#  Why did the chicken join the band? Because it had the drumsticks!

