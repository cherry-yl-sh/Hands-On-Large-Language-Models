from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 输出 CUDA 的版本
# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cpu",
    torch_dtype="auto",
    trust_remote_code=False,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened.<|assistant|>"

# Tokenize the input prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cpu")

# Generate the text
generation_output = model.generate(
    input_ids=input_ids,
    max_new_tokens=20
)

# Print the output
print("output:", tokenizer.decode(generation_output[0]))

print("input ids ", input_ids)
print("input id encode ")
for id in input_ids[0]:
    print(tokenizer.decode(id))
# output: Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened.<|assistant|> Subject: Heartfelt Apologies for the Gardening Mishap
#
#
# Dear
# input ids  tensor([[14350,   385,  4876, 27746,  5281,   304, 19235,   363,   278, 25305,
#            293, 16423,   292,   286,   728,   481, 29889, 12027,  7420,   920,
#            372,  9559, 29889, 32001]])
# input id encode
# Write
# an
# email
# apolog
# izing
# to
# Sarah
# for
# the
# trag
# ic
# garden
# ing
# m
# ish
# ap
# .
# Exp
# lain
# how
# it
# happened
# .
# <|assistant|>
