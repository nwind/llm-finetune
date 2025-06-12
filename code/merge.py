from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 模型路径
model_name_or_path = "llama3"
# lora 文件路径
peft_model_path = "llama3_lora"
# 输出路径
output_dir = "llama3_merge"

base_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path
)

model = PeftModel.from_pretrained(base_model, peft_model_path)
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,  padding_side="right")

model.save_pretrained(output_dir, max_shard_size="10GB")
tokenizer.save_pretrained(output_dir)