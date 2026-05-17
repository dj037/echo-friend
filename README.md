# Echo Friend

## 项目简介
Echo Friend 是一个基于真实聊天记录的个性化对话模型微调项目。项目通过数据清洗、风格分析和 LoRA 微调方式，让模型在语气、用词和回复长度等方面贴近特定朋友的表达习惯。训练和评估流程集中在 Echo friend.ipynb 中完成，最终产出可用于对话的 LoRA 适配器模型。
## 运行与查看微调模型

### 训练阶段（在 Colab 中完成）
1.将整体文件下载并上传至Google Drive
2. 打开 Google Colab 并上传 Echo friend.ipynb。
3. 在 Colab 中依次运行 notebook 的所有单元格，完成数据处理、训练、评估和模型保存流程。
4. 训练完成后，模型会保存在 Google Drive 的 output/final_model 目录。

### 本地推理与对话
1. 从 Google Drive 下载 output/final_model 到本地，例如保存为 ./models/final_model。
2. 安装依赖：
   - pip install torch transformers peft
3. 运行以下示例脚本进行对话：

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "./models/final_model"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)

def chat(user_message):
    prompt = f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            temperature=0.8,
            top_p=0.9,
            max_new_tokens=50,
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("<|im_start|>assistant\n")[-1].strip()

print(chat("你好"))
```

4. 如果需要网页对话界面，请在 Echo friend.ipynb 中运行 Gradio 相关单元格。运行后会输出一个临时链接，可在浏览器网页中与模型对话。

说明：对话参数（temperature、top_p、max_new_tokens 等）可按需要调整以控制回复长度和随机性。
