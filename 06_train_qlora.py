"""[6단계] Student(Qwen2.5-VL-3B) QLoRA 파인튜닝.

핵심 구현 포인트 (발표 '기술 내용'에서 설명할 부분)
1) QLoRA: base 가중치는 4bit NF4로 얼리고(frozen), LLM의 선형층에 저랭크 행렬 B·A만 학습.
     W' = W + (alpha/r) · B·A,   A ∈ R^{r×d}, B ∈ R^{d×r},  r ≪ d
2) 기본은 언어모델 쪽 선형층만 LoRA 적용 (vision encoder는 freeze).
   --tune-vision 을 주면 vision tower의 qkv/proj에도 LoRA → ablation 실험용.
3) Loss masking: 시스템/사용자 프롬프트와 이미지 토큰은 label=-100으로 가리고,
   assistant 답변(JSON) 토큰에만 cross-entropy loss를 건다.

사용 예 (Colab T4 16GB 기준):
    python 06_train_qlora.py --train data/splits/train.jsonl --val data/splits/val.jsonl \
        --frames data/frames --out outputs/lora-r16
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from common import USER_PROMPT, build_messages, load_processor, pick_dtype, read_jsonl

LLM_TARGETS = r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
ALL_TARGETS = r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|qkv|attn\.proj)$"


class WalkDataset(Dataset):
    def __init__(self, path, frames):
        self.rows = read_jsonl(path)
        self.frames = Path(frames)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {"image": Image.open(self.frames / r["image"]).convert("RGB"), "target": r["target"]}


class Collator:
    def __init__(self, processor):
        self.p = processor
        self.p.tokenizer.padding_side = "right"

    def __call__(self, batch):
        images = [b["image"] for b in batch]
        full = [self.p.apply_chat_template(build_messages(USER_PROMPT, b["target"]), tokenize=False)
                for b in batch]
        prompt = self.p.apply_chat_template(build_messages(USER_PROMPT), tokenize=False,
                                            add_generation_prompt=True)
        enc = self.p(text=full, images=images, padding=True, return_tensors="pt")
        labels = enc["input_ids"].clone()
        # 프롬프트 길이 = 같은 이미지로 프롬프트만 토크나이즈한 길이 (이미지 토큰 수가 이미지마다 다름)
        for i, img in enumerate(images):
            plen = self.p(text=[prompt], images=[img], return_tensors="pt")["input_ids"].shape[1]
            labels[i, :plen] = -100
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--train", default="data/splits/train.jsonl")
    ap.add_argument("--val", default="data/splits/val.jsonl")
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--out", default="outputs/lora-r16")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-pixels", type=int, default=448 * 448, help="OOM이면 줄이기")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--tune-vision", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, Trainer,
                              TrainingArguments, set_seed)

    set_seed(args.seed)
    dtype = pick_dtype()
    processor = load_processor(args.model, args.max_pixels)

    qcfg = None if args.no_4bit else BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto", quantization_config=qcfg)
    model.config.use_cache = False
    if qcfg is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lcfg = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none",
        target_modules=ALL_TARGETS if args.tune_vision else LLM_TARGETS, task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    train_ds = WalkDataset(args.train, args.frames)
    val_ds = WalkDataset(args.val, args.frames)
    print(f"train {len(train_ds)} / val {len(val_ds)}")

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_num_workers=2,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=Collator(processor))
    trainer.train()

    out = Path(args.out)
    trainer.save_model(out / "final")
    processor.save_pretrained(out / "final")
    (out / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    (out / "train_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    plot_loss(trainer.state.log_history, out / "loss_curve.png")
    print(f"저장 완료 → {out / 'final'}")


def plot_loss(history, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr = [(h["step"], h["loss"]) for h in history if "loss" in h]
    ev = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]
    plt.figure(figsize=(6, 4))
    if tr:
        plt.plot(*zip(*tr), label="train loss")
    if ev:
        plt.plot(*zip(*ev), "o-", label="val loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
