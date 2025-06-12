import json

def gen_example_by_length(length):
  single_token = 'hello'

  example = {
    'instruction': '',
    'input': single_token * (length - 1),
    'output': single_token
  }

  # 生成 10 个样本
  instructions = [example for i in range(10)]

  # 写入文件
  with open(f"instructions-{length}.json", 'w') as f:
      json.dump(instructions, f)

gen_example_by_length(512)
gen_example_by_length(1024)
gen_example_by_length(2048)