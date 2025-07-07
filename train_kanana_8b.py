'''

python train_kanana_8b.py \
  --train_file datasets/korean_language_rag_V1.0_train.json \
  --valid_file datasets/korean_language_rag_V1.0_dev.json \
  --model_id kakaocorp/kanana-1.5-8b-instruct-2505 \
  --device 2 \
  --cache_dir $TRANSFORMERS_CACHE \
  --output_dir ~/checkpoints/kanana-1.5-8b-instruct-2505

'''

#!/usr/bin/env python3
import os
import json
import argparse
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from src.data import CustomDataset, DataCollatorForSupervisedDataset
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from transformers import BitsAndBytesConfig


class SupervisedTrainer(Trainer):
    # return_outputs 뿐 아니라 추가 인자(num_items_in_batch 등)도 받도록 **kwargs 추가
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # ① inputs dict에서 labels를 꺼냅니다.
        labels = inputs.pop("labels", None)
        # ② 모델 호출 시 labels 파라미터로 넘겨주면
        #    모델 내부에서 CrossEntropyLoss가 자동 계산됩니다.
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        # ③ return_outputs 옵션에 따라 (loss, outputs) 페어를 리턴
        return (loss, outputs) if return_outputs else loss

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Kanana model with LoRA"
    )
    parser.add_argument("--train_file", type=str, required=True, help="Path to train JSON")
    parser.add_argument("--valid_file", type=str, required=True, help="Path to validation JSON")
    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--output_dir", type=str, default="runs/checkpoints", help="Directory to save checkpoints and best model")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index (e.g. '0' or '0,1')")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for HF models/tokenizers")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X steps")
    parser.add_argument("--eval_steps", type=int, default=100, help="Run evaluation every X steps")
    #parser.add_argument("--deepspeed_config", type=str, default=None, help="Path to DeepSpeed config JSON")
    parser.add_argument("--use_auth_token", type=str, default=None, help="HF auth token if needed")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # load tokenizer
    tok_kwargs = {}
    if args.use_auth_token:
        tok_kwargs["use_auth_token"] = args.use_auth_token
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
        **tok_kwargs
    )
    tokenizer.pad_token = tokenizer.eos_token

    # load model
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if args.use_auth_token:
        model_kwargs["use_auth_token"] = args.use_auth_token

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,   # 필요시 튜닝
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",             # device_map="auto"로도 OK
        cache_dir=args.cache_dir,
    )

    # apply LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)

    # datasets & collator
    train_ds = CustomDataset(args.train_file, tokenizer)
    eval_ds = CustomDataset(args.valid_file, tokenizer)
    collator = DataCollatorForSupervisedDataset(tokenizer)

    # training args
    ds_args = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.accum_steps,
        "num_train_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "logging_steps": 10,
        "save_steps": args.save_steps,
        "save_total_limit": 1,
        "fp16": True,
        "eval_strategy": "steps",
        "eval_steps": args.eval_steps,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": ["tensorboard"],
    }
    # if args.deepspeed_config:
    #     ds_args["deepspeed"] = args.deepspeed_config

    training_args = TrainingArguments(**ds_args)

    trainer = SupervisedTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSupervisedDataset(tokenizer),
    )

    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(os.path.join(training_args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # train
    try:
        trainer.train()
        best_dir = os.path.join(training_args.output_dir, "best_model")
        os.makedirs(best_dir, exist_ok=True)
        trainer.model.save_pretrained(best_dir)
        tokenizer.save_pretrained(best_dir)
        print(f"[INFO] Best model saved to {best_dir}")
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")

    print(f"[INFO] CUDA Devices: {torch.cuda.device_count()} (visible: {args.device})")

if __name__ == "__main__":
    main()