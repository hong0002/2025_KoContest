'''

python run/test.py \
  --input datasets/korean_language_rag_V1.0_test.json \
  --output runs/EXAONE-Deep-7.8B/predictions.json \
  --model_id LGAI-EXAONE/EXAONE-Deep-7.8B \
  --adapter /home/oem/checkpoints/EXAONE-Deep-7.8B/best_model \
  --device cuda:2

'''

import os
import argparse
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import PeftModel
import torch
from src.data import DataCollatorForSupervisedDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      type=str, required=True)
    parser.add_argument("--output",     type=str, required=True)
    parser.add_argument("--model_id",   type=str, required=True)  # e.g. "LGAI-EXAONE/EXAONE-Deep-7.8B"
    parser.add_argument("--adapter",    type=str, required=True)  # e.g. "~/checkpoints/…/best_model"
    parser.add_argument("--device",     type=str, required=True)  # e.g. "cuda:0"
    args = parser.parse_args()

    # 1) 베이스 모델 & 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map={0: args.device},      # bitsandbytes 쓰면 "device_map='auto'" 로
        torch_dtype=torch.bfloat16,       # 필요에 따라
        trust_remote_code=True,
    )

    # 2) PEFT adapter 로드
    model = PeftModel.from_pretrained(base_model, args.adapter, torch_dtype=torch.bfloat16)
    model.eval()

    # 3) 데이터셋 준비
    dataset = DataCollatorForSupervisedDataset(args.input, tokenizer)

    # 4) 생성
    terminators = [tokenizer.eos_token_id]
    with open(args.input, "r", encoding="utf-8") as f:
        examples = json.load(f)

    for i, inp in enumerate(dataset):
        # inp: tensor([…]) 1D
        out = model.generate(
            inp.unsqueeze(0).to(args.device),
            max_new_tokens=256,
            eos_token_id=terminators,
            pad_token_id=tokenizer.pad_token_id,
        )
        pred = tokenizer.decode(out[0, inp.shape[-1]:], skip_special_tokens=True)
        examples[i]["output"] = {"answer": pred}

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()