'''

export CUDA_VISIBLE_DEVICES=2

python test.py \
  --input  datasets/korean_language_rag_V1.0_test.json \
  --output runs/EXAONE-Deep-7.8B/predictions.json \
  --model_id LGAI-EXAONE/EXAONE-Deep-7.8B \
  --adapter /home/oem/checkpoints/EXAONE-Deep-7.8B/best_model \
  --device cuda:0

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
from src.data import CustomDataset
from transformers import BitsAndBytesConfig
from tqdm import tqdm

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

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,   # 필요시 튜닝
    )

    # bitsandbytes 8-bit 양자화 없이 단일 GPU 쓰려면:
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map={"": args.device},   # 모든 파라미터를 cuda:2 에 할당
    )

    # 2) PEFT adapter 로드
    model = PeftModel.from_pretrained(base_model, args.adapter, torch_dtype=torch.bfloat16)
    model.eval()

    # 3) 데이터셋 준비
    dataset = CustomDataset(args.input, tokenizer)

    # 4) 생성
    terminators = [tokenizer.eos_token_id]
    with open(args.input, "r", encoding="utf-8") as f:
        examples = json.load(f)

    # tqdm을 쓰려면 전체 샘플 수를 알려주면 좋습니다.
    total = len(dataset)
    for i, sample in enumerate(tqdm(dataset, total=total, desc="생성 진행")):
        inp = sample["input_ids"]       # 1D Tensor
        attention_mask = (inp != tokenizer.pad_token_id).long()

        out = model.generate(
            inp.unsqueeze(0).to(args.device),
            attention_mask=attention_mask.unsqueeze(0).to(args.device),
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