'''

export CUDA_VISIBLE_DEVICES=2

python test_kanana_rag.py \
  --input  ko_datasets/korean_language_rag_V1.0_test.json \
  --output runs/kanana-1.5-8b-rag/predictions.json \
  --model_id kakaocorp/kanana-1.5-8b-instruct-2505 \
  --adapter /home/oem/checkpoints/kanana-1.5-8b-rag/best_model \
  --device cuda:0 \
  --pdf_path "ko_datasets/국어 지식 기반 생성(RAG) 참조 문서.pdf" \
  --top_k 3

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
from transformers import BitsAndBytesConfig
from tqdm import tqdm

import re, faiss, numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2

def extract_and_chunk(pdf_path, max_len=180):
    reader = PyPDF2.PdfReader(pdf_path)
    full = "\n".join(p.extract_text() or "" for p in reader.pages)
    paras = [p.strip() for p in re.split(r"\n+", full) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) < max_len:
            buf += (" " if buf else "") + p
        else:
            if buf:
                chunks.append(buf)
            if len(p) > max_len:
                for i in range(0, len(p), max_len):
                    chunks.append(p[i:i+max_len])
                buf = ""
            else:
                buf = p
    if buf: chunks.append(buf)
    return chunks

class Retriever:
    def __init__(self, chunks, model_name):
        self.chunks = chunks
        self.embedder = SentenceTransformer(model_name)
        emb = self.embedder.encode(chunks, convert_to_numpy=True,
                                   normalize_embeddings=True).astype("float32")
        faiss.normalize_L2(emb)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

    def topk(self, query, k):
        q = self.embedder.encode([query], convert_to_numpy=True,
                                 normalize_embeddings=True).astype("float32")
        faiss.normalize_L2(q)
        _, I = self.index.search(q, k)
        return [self.chunks[i] for i in I[0] if i != -1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      type=str, required=True)
    parser.add_argument("--output",     type=str, required=True)
    parser.add_argument("--model_id",   type=str, required=True)  # e.g. "LGAI-EXAONE/EXAONE-Deep-7.8B"
    parser.add_argument("--adapter",    type=str, required=True)  # e.g. "~/checkpoints/…/best_model"
    parser.add_argument("--device",     type=str, required=True)  # e.g. "cuda:0"
    parser.add_argument("--pdf_path",   type=str, required=True, help="RAG에서 사용할 참조 PDF")
    parser.add_argument("--embed_model", type=str, default="jhgan/ko-sroberta-multitask", help="Sentence-BERT 모델")
    parser.add_argument("--top_k", type=int, default=3, help="질문마다 검색할 청크 개수")
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

    chunks = extract_and_chunk(args.pdf_path)
    retriever = Retriever(chunks, args.embed_model)

    # ────────────────── 데이터 로드
    with open(args.input, encoding="utf-8") as f:
        examples = json.load(f)

    # ────────────────── 생성
    for i, sample in enumerate(tqdm(examples, total=len(examples), desc="생성 진행")):
        q = sample["input"]["question"]
        ctxs = retriever.topk(q, args.top_k)
        prompt = "".join(f"[컨텍스트{j+1}] {c}\n" for j, c in enumerate(ctxs))
        prompt += f"[질문] {q}\n[답변]"

        inp_ids = tokenizer(prompt, return_tensors="pt",
                            truncation=True, max_length=4096).input_ids.to(args.device)
        with torch.no_grad():
            attn = (inp_ids != tokenizer.pad_token_id).long()
            out = model.generate(
                inp_ids,
                attention_mask=attn,
                max_new_tokens=256,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        pred = tokenizer.decode(out[0, inp_ids.shape[-1]:], skip_special_tokens=True)
        sample["output"] = {"answer": pred}   # ← 루프 안!

    # ────────────────── 저장
    os.makedirs(os.path.dirname(args.output), exist_ok=True)  # ← NEW!
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()