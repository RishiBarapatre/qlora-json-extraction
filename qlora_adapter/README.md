---
base_model: Qwen/Qwen3-4B
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen3-4B
- lora
- qlora
- sft
- transformers
- trl
- biomedical
- clinical-nlp
license: mit
---

# QLoRA Adapter: Qwen3-4B for Structured Adverse Drug Effect Extraction

This repository contains the fine-tuned LoRA adapter weights for [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) trained on the **ADE Corpus v2** for structured entity extraction (drug name and adverse drug effect) in strict JSON format.

## Model Details

- **Base Model:** `Qwen/Qwen3-4B`
- **Adapter Type:** LoRA (Low-Rank Adaptation)
- **Task:** Clinical Information Extraction / Adverse Drug Reaction extraction
- **Output Format:** Strict JSON (`{"drug": "...", "effect": "..."}`)
- **Training Method:** QLoRA (4-bit NF4 quantized base + 16-bit LoRA adapters)
- **Trainable Parameters:** ~52M (~1.3% of total parameters)
- **Adapter Weight Size:** ~66 MB

## LoRA Hyperparameters

- **Rank ($r$):** 16
- **Scaling Factor ($\alpha$):** 32
- **Dropout:** 0.05
- **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (all attention and MLP projections)
- **Task Type:** `CAUSAL_LM`

## How to Use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model_id = "Qwen/Qwen3-4B"
adapter_path = "./qlora_adapter"  # or Hugging Face Hub repo ID

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(adapter_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
```

## Performance Highlights

Evaluated on 500 test sentences from ADE Corpus v2:
- **JSON Parse Rate:** 100.0%
- **Field Micro-F1:** 90.2% (+4.3 pp over base model)
- **Full Exact Match:** 73.4% (+11.6 pp over base model)
- **Drug Token-F1:** 94.0% (+3.5 pp over base model)

## Framework Versions

- PEFT 0.20.0
- Transformers 5.16.1
- TRL 1.13.0
- BitsAndBytes 0.50.2
- PyTorch 2.11.0