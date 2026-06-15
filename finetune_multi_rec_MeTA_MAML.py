import os
import json
import random
from typing import List

import numpy as np
import fire
import torch
import torch.optim as optim
import transformers
from datasets import load_dataset, concatenate_datasets
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer
from sklearn.metrics import roc_auc_score


# ========== プロンプト生成関数 ==========

def generate_prompt(data_point):
    """データポイントの instruction / input / output を整形してプロンプト文字列を作成する"""
    if data_point["input"]:
        return (
            "Below is an instruction that describes a task, paired with an input that "
            "provides further context. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{data_point['instruction']}\n\n"
            f"### Input:\n{data_point['input']}\n\n"
            f"### Response:\n{data_point['output']}"
        )
    else:
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{data_point['instruction']}\n\n"
            f"### Response:\n{data_point['output']}"
        )


# ========== メイン学習関数 ==========

def train(
    # --- モデル / データ パラメータ ---
    base_model: str = "",           # ベースLLMのパス (例: 'decapoda-research/llama-7b-hf')
    train_data_path: str = "",      # タスク1の学習データパス
    train_data_path2: str = "",     # タスク2の学習データパス
    val_data_path: str = "",        # タスク1の検証データパス
    val_data_path2: str = "",       # タスク2の検証データパス
    output_dir: str = "./lora-alpaca",  # 学習済みLoRAアダプタの保存先
    sample: int = -1,               # デバッグ用: 使用するサンプル数 (-1 で全件)
    seed: int = 0,                  # 乱数シード
    # --- 学習ハイパーパラメータ ---
    batch_size: int = 64,           # メタバッチサイズ
    micro_batch_size: int = 1,      # (互換性のため残存、未使用)
    num_epochs: int = 3,            # 学習エポック数
    learning_rate: float = 3e-4,    # メタ更新（外部ループ）の学習率
    cutoff_len: int = 256,          # トークン最大長
    # --- LoRA ハイパーパラメータ ---
    lora_r: int = 8,                # LoRAのランク
    lora_alpha: int = 16,           # LoRAのスケーリング係数
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["q_proj", "v_proj"],
    # --- LLM ハイパーパラメータ ---
    train_on_inputs: bool = True,   # 入力部分も損失計算に含めるか
    group_by_length: bool = False,  # (互換性のため残存、未使用)
    # --- その他 ---
    resume_from_checkpoint: str = None,  # チェックポイントから再開
    # --- MAML ハイパーパラメータ ---
    inner_adaptation_steps: int = 5,     # 内部ループの適応ステップ数
    inner_learning_rate: float = 1e-4,   # 内部ループの学習率
    early_stopping_patience: int = 3,    # (互換性のため残存、未使用)
    k_shot: int = 5,                     # サポートセットのサンプル数
):
    # ===== パラメータ表示 =====
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"train_data_path: {train_data_path}\n"
        f"val_data_path: {val_data_path}\n"
        f"sample: {sample}\n"
        f"seed: {seed}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
        f"k_shot: {k_shot}\n"
        f"inner_adaptation_steps: {inner_adaptation_steps}\n"
        f"inner_learning_rate: {inner_learning_rate}\n"
    )

    assert base_model, "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    # ===== 乱数シード固定 =====
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ===== DDP設定 =====
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    # wandb無効化
    os.environ["WANDB_DISABLED"] = "true"

    # ========================================
    # 1. モデルとトークナイザのロード
    # ========================================
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0          # パディングトークンを unk (0) に設定
    tokenizer.padding_side = "left"     # CausalLMのバッチ推論用に左パディング

    # ========================================
    # 2. トークナイズ関数
    # ========================================
    def tokenize(prompt, add_eos_token=True):
        """プロンプトをトークナイズし、必要に応じてEOSトークンを追加する"""
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        # EOSトークンを追加
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        # ラベルとしてinput_idsをコピー
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        """データポイントからプロンプトを生成してトークナイズする"""
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)

        # 入力部分の損失をマスク (train_on_inputs=False の場合)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            # -100 に設定されたラベルは損失計算から除外される
            tokenized_full_prompt["labels"] = (
                [-100] * user_prompt_len
                + tokenized_full_prompt["labels"][user_prompt_len:]
            )
        return tokenized_full_prompt

    # ========================================
    # 3. LoRAモデルの準備
    # ========================================
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    # パディング処理用のデータコレータ
    padding_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    # ========================================
    # 4. メタ学習用データコレータ (MAMLの核心部)
    # ========================================
    def meta_collate_fn(batch_of_raw_samples, k_shot_value=k_shot):
        """
        バッチをタスクIDごとに分割し、MAML用のメタバッチ辞書に変換する。
        メタバッチ = {
            'task_0': {'support': サポートセット, 'query': クエリセット},
            'task_1': {'support': サポートセット, 'query': クエリセット},
        }
        """
        # タスクIDごとにサンプルを仕分ける
        tasks_data = {}
        for s in batch_of_raw_samples:
            if 'task_id' not in s:
                print(f"Warning: 'task_id' not found in sample: {s}. Skipping.")
                continue
            task_id = s['task_id']
            tasks_data.setdefault(task_id, []).append(s)

        meta_batch = {}
        for task_id, samples in tasks_data.items():
            tokenized_samples = [generate_and_tokenize_prompt(s) for s in samples]

            # サポートセットとクエリセットに分割
            split_size = min(k_shot_value, len(tokenized_samples) // 2)
            if split_size == 0 and len(tokenized_samples) >= 2:
                split_size = 1
            elif len(tokenized_samples) < 2:
                continue

            support_samples = tokenized_samples[:split_size]
            query_samples = tokenized_samples[split_size:]

            S_i_batch = padding_collator(support_samples)
            Q_i_batch = (
                S_i_batch.copy() if not query_samples
                else padding_collator(query_samples)
            )

            meta_batch[f"task_{task_id}"] = {
                'support': S_i_batch,  # 内部ループ（適応）用
                'query': Q_i_batch,    # 外部ループ（評価）用
            }

        return meta_batch

    # ========================================
    # 5. データセットのロードと前処理
    # ========================================
    def _load_dataset(path):
        """JSONファイルまたはHugging Faceデータセット名をロードする"""
        if path.endswith(".json"):
            return load_dataset("json", data_files=path)
        return load_dataset(path)

    # タスク1: task_id=0
    train_data = _load_dataset(train_data_path).map(lambda _: {'task_id': 0})
    val_data = _load_dataset(val_data_path).map(lambda _: {'task_id': 0})

    # タスク2: task_id=1
    train_data2 = _load_dataset(train_data_path2).map(lambda _: {'task_id': 1})
    val_data2 = _load_dataset(val_data_path2).map(lambda _: {'task_id': 1})

    # ========================================
    # 6. チェックポイントからの再開
    # ========================================
    if resume_from_checkpoint:
        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(resume_from_checkpoint, "adapter_model.bin")
            resume_from_checkpoint = False

        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()

    # ========================================
    # 7. 訓練・検証データセットの結合
    # ========================================
    # デバッグ用サンプリング＆シャッフル
    train_data["train"] = (
        train_data["train"].shuffle(seed=seed).select(range(sample))
        if sample > -1
        else train_data["train"].shuffle(seed=seed)
    )
    train_data2["train"] = (
        train_data2["train"].shuffle(seed=seed).select(range(sample))
        if sample > -1
        else train_data2["train"].shuffle(seed=seed)
    )

    # タスク0とタスク1の訓練データを結合 (meta_collate_fn が task_id で分離)
    train_data["train"] = concatenate_datasets([train_data["train"], train_data2["train"]])
    print("Concatenating validation datasets...")
    val_data["train"] = val_data["train"].shuffle(seed=seed)
    val_data2["train"] = val_data2["train"].shuffle(seed=seed)
    val_data["train"] = concatenate_datasets([val_data["train"], val_data2["train"]])

    # ========================================
    # 8. 使用したデータをファイルに保存
    # ========================================
    os.makedirs(output_dir, exist_ok=True)

    def _save_dataset_to_json(dataset, filepath):
        """データセットをJSONファイルに保存する"""
        data_list = [
            {
                'instruction': item.get('instruction', ''),
                'input': item.get('input', ''),
                'output': item.get('output', ''),
                'task_id': item.get('task_id', -1),
            }
            for item in dataset
        ]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
        print(f"Data saved to {filepath} ({len(data_list)} samples)")
        return data_list

    print("Saving training and validation data to files...")
    _save_dataset_to_json(train_data["train"], os.path.join(output_dir, "used_train_data.json"))
    _save_dataset_to_json(val_data["train"], os.path.join(output_dir, "used_val_data.json"))
    print("Data saving complete!\n")

    # マルチGPU設定
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    # ========================================
    # 9. 評価用ユーティリティ
    # ========================================
    def preprocess_logits_for_metrics(logits, labels):
        """ロジットからAUC計算用の予測値と正解ラベルを抽出する"""
        # "Yes"=8241, "No"=3782 のトークンIDに対応するインデックスを取得
        labels_index = torch.argwhere(torch.bitwise_or(labels == 8241, labels == 3782))
        gold = torch.where(labels[labels_index[:, 0], labels_index[:, 1]] == 3782, 0, 1)
        labels_index[:, 1] = labels_index[:, 1] - 1
        logits = logits.softmax(dim=-1)
        logits = torch.softmax(
            logits[labels_index[:, 0], labels_index[:, 1]][:, [3782, 8241]], dim=-1
        )
        return logits[:, 1][2::3], gold[2::3]

    # 評価ステップ間隔の設定
    if sample > -1:
        eval_step = 10 if sample <= 128 else sample / 128 * 5
    else:
        eval_step = 20  # デフォルト値

    # ========================================
    # 10. 評価関数 (内部ループ適応付き)
    # ========================================
    def evaluate(model_to_eval, dataloader, inner_steps=5, inner_lr=1e-4):
        """
        検証時にもMAMLの内部ループ（Few-Shot適応）を行い、
        クエリセットでAUCと損失を計算する。
        """
        print("Running evaluation with inner loop adaptation...")
        model_to_eval.eval()
        all_preds = []
        all_labels = []
        all_losses = []

        for meta_batch in dataloader:
            if not meta_batch:
                continue

            for task_name in meta_batch.keys():
                # 現在のLoRAパラメータを保存
                saved_eval_state = get_peft_model_state_dict(model_to_eval)

                # サポートセットをデバイスに転送
                S_i_eval = {
                    k: v.to(model_to_eval.device)
                    for k, v in meta_batch[task_name]['support'].items()
                    if isinstance(v, torch.Tensor)
                }

                # 内部ループ: サポートセットで適応
                inner_trainable_params = [p for p in model_to_eval.parameters() if p.requires_grad]
                for _ in range(inner_steps):
                    outputs_support = model_to_eval(**S_i_eval)
                    grads = torch.autograd.grad(
                        outputs_support.loss, inner_trainable_params, create_graph=False
                    )
                    with torch.no_grad():
                        for param, g in zip(inner_trainable_params, grads):
                            if g is not None:
                                param.data -= inner_lr * g

                # 適応後、クエリセットで評価
                Q_i = {
                    k: v.to(model_to_eval.device)
                    for k, v in meta_batch[task_name]['query'].items()
                    if isinstance(v, torch.Tensor)
                }

                with torch.no_grad():
                    outputs = model_to_eval(
                        input_ids=Q_i['input_ids'],
                        attention_mask=Q_i['attention_mask'],
                        labels=Q_i['labels'],
                    )
                    all_losses.append(outputs.loss.item())

                    preds, labels = preprocess_logits_for_metrics(
                        outputs.logits.detach(), Q_i['labels'].detach()
                    )
                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())

                # LoRAパラメータを元の状態に復元
                set_peft_model_state_dict(model_to_eval, saved_eval_state)

        # 結果の集約
        if not all_preds or all_preds[0].nelement() == 0:
            model_to_eval.train()
            return {'eval_auc': 0.0, 'eval_loss': 0.0}

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        avg_loss = np.mean(all_losses) if all_losses else 0.0

        try:
            auc = roc_auc_score(all_labels, all_preds)
        except ValueError:
            auc = 0.0

        model_to_eval.train()
        return {'eval_auc': auc, 'eval_loss': avg_loss}

    # ========================================
    # 11. MAML カスタム学習の準備
    # ========================================
    INNER_ADAPTATION_STEPS = inner_adaptation_steps
    INNER_LEARNING_RATE = inner_learning_rate

    # メタ更新用オプティマイザ (LoRAパラメータのみ)
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
    )

    # ========================================
    # 12. データローダーの作成
    # ========================================
    train_dataloader = torch.utils.data.DataLoader(
        train_data["train"],
        batch_size=batch_size,
        collate_fn=meta_collate_fn,
        shuffle=True,
        num_workers=4,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_data["train"],
        batch_size=batch_size,
        collate_fn=meta_collate_fn,
        shuffle=False,
        num_workers=4,
    )

    # ========================================
    # 13. First-Order MAML 学習ループ
    # ========================================
    print("Starting FO-MAML LoRA Training Loop...")
    model.train()
    model.config.use_cache = False  # 学習中はキャッシュ無効化

    # 訓練可能なパラメータ（LoRA重み）とその名前
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    param_names = [name for name, p in model.named_parameters() if p.requires_grad]

    auc_history = []
    best_auc = 0.0
    global_step = 0
    save_interval = 10  # チェックポイント保存間隔

    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch + 1}/{num_epochs}")

        # メタ勾配蓄積用辞書の初期化
        accumulated_meta_grads = {
            name: torch.zeros_like(p) for name, p in zip(param_names, trainable_params)
        }
        num_tasks_processed = 0

        for i, meta_batch in enumerate(train_dataloader):
            if not meta_batch:
                continue

            # --- 各タスクについて処理 ---
            for task_name in meta_batch.keys():
                S_i = meta_batch[task_name]['support']
                Q_i = meta_batch[task_name]['query']

                # (1) 現在のLoRAパラメータを保存
                saved_state_dict = get_peft_model_state_dict(model)

                # (2) 内部ループ: サポートセットで高速適応
                for _ in range(INNER_ADAPTATION_STEPS):
                    S_i_batch = {
                        k: v.to(model.device) for k, v in S_i.items() if isinstance(v, torch.Tensor)
                    }
                    outputs = model(**S_i_batch)
                    grads = torch.autograd.grad(
                        outputs.loss, trainable_params, create_graph=False
                    )
                    with torch.no_grad():
                        for param, g in zip(trainable_params, grads):
                            if g is not None:
                                param.data -= INNER_LEARNING_RATE * g

                # (3) 外部ループ: 適応後のモデルでクエリセットのメタ損失を計算
                Q_i_batch = {
                    k: v.to(model.device) for k, v in Q_i.items() if isinstance(v, torch.Tensor)
                }
                outputs_q = model(**Q_i_batch)
                query_loss = outputs_q.loss

                # メタ勾配を計算して蓄積
                meta_grads_task = torch.autograd.grad(
                    query_loss, trainable_params, create_graph=False
                )
                with torch.no_grad():
                    for name, g in zip(param_names, meta_grads_task):
                        if g is not None:
                            accumulated_meta_grads[name] += g
                num_tasks_processed += 1

                # (4) LoRAパラメータを元の状態に復元
                set_peft_model_state_dict(model, saved_state_dict)

            # --- (5) メタ更新: 蓄積した勾配の平均でパラメータを更新 ---
            if num_tasks_processed > 0:
                optimizer.zero_grad()
                with torch.no_grad():
                    for name, param in zip(param_names, trainable_params):
                        param.grad = accumulated_meta_grads[name] / num_tasks_processed
                optimizer.step()

                # 蓄積用辞書をリセット
                accumulated_meta_grads = {
                    name: torch.zeros_like(p) for name, p in zip(param_names, trainable_params)
                }
                num_tasks_processed = 0
                global_step += 1

                # --- ログ出力 ---
                if global_step % 8 == 0:
                    print(
                        f"  Epoch {epoch}, Iter {i} (Global Step {global_step}): "
                        f"Approx Meta Loss = {query_loss.item()}"
                    )

                # --- チェックポイント保存 ---
                if global_step > 0 and global_step % save_interval == 0:
                    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    print(f"--- Saving checkpoint at step {global_step} to {checkpoint_dir} ---")
                    model.save_pretrained(checkpoint_dir)

                # --- 定期評価 ---
                if global_step > 0 and global_step % eval_step == 0:
                    print(f"--- Running evaluation at step {global_step} ---")
                    model.config.use_cache = True
                    metrics = evaluate(
                        model, val_dataloader,
                        inner_steps=INNER_ADAPTATION_STEPS,
                        inner_lr=INNER_LEARNING_RATE,
                    )
                    print(f"Step {global_step}: {metrics}")

                    eval_auc = metrics.get('eval_auc', 0.0)
                    eval_loss = metrics.get('eval_loss', 0.0)
                    auc_history.append({'step': global_step, 'auc': eval_auc, 'loss': eval_loss})

                    # AUC履歴を保存
                    history_file_path = os.path.join(output_dir, "auc_history.json")
                    with open(history_file_path, 'w') as f:
                        json.dump(auc_history, f, indent=4)
                    print(f"AUC history saved to {history_file_path}")

                    # ベストモデルの保存
                    if eval_auc > best_auc:
                        best_auc = eval_auc
                        print(f"New best model with AUC: {best_auc}. Saving to {output_dir}...")
                        model.save_pretrained(output_dir)

                    model.config.use_cache = False

    # ========================================
    # 14. 学習終了後の保存処理
    # ========================================
    print("FO-MAML LoRA training finished.")

    history_file_path = os.path.join(output_dir, "auc_history.json")
    print(f"Saving AUC history to {history_file_path}")
    with open(history_file_path, 'w') as f:
        json.dump(auc_history, f, indent=4)

    print(f"Saving final model to {output_dir}...")
    model.save_pretrained(output_dir)

    print("\nIf there's a warning about missing keys above, please disregard :)")


if __name__ == "__main__":
    fire.Fire(train)
```
</copilot-edited-file>
```
This is the complete file with the suggested change applied. The `import sys` statement has been removed as it was unused.
