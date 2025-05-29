from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
)

model = AutoModelForCausalLM.from_pretrained(
    "georgesung/llama2_7b_chat_uncensored",
    cache_dir = '/ssdshare',
    trust_remote_code=True
)