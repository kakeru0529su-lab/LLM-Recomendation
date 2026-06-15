import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import json
import random
from typing import Optional

import numpy as np
import fire
import torch
torch.set_num_threads(1)
from tqdm import tqdm
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, log_loss,
    matthews_corrcoef, cohen_kappa_score,
)


# ========== デバイス判定 ==========

def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

device = _get_device()


# ========== ユーティリティ ==========

def set_seed(seed):
    """乱数シードを固定して再現性を確保する"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def generate_prompt(instruction, input_text=None):
    """評価用プロンプトを生成する（出力部分は空）"""
    if input_text:
        return (
            "Below is an instruction that describes a task, paired with an input that "
            "provides further context. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            "### Response:\n"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )


def batch_iter(lst, batch_size=32):
    """リストをバッチサイズごとに分割するジェネレータ"""
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


def compute_all_metrics(gold, pred_probs):
    """
    全評価指標を計算して辞書で返す。
    gold: 正解ラベルのリスト (0/1)
    pred_probs: Yes確率のリスト
    """
    auc_score = roc_auc_score(gold, pred_probs)
    pr_auc = average_precision_score(gold, pred_probs)

    # 予測ラベル（閾値 0.5）
    pred_labels = [1 if p > 0.5 else 0 for p in pred_probs]

    # 混同行列
    tp = sum(1 for p, g in zip(pred_labels, gold) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(pred_labels, gold) if p == 1 and g == 0)
    tn = sum(1 for p, g in zip(pred_labels, gold) if p == 0 and g == 0)
    fn = sum(1 for p, g in zip(pred_labels, gold) if p == 0 and g == 1)

    correct_count = tp + tn
    accuracy = correct_count / len(gold) if gold else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    mcc = matthews_corrcoef(gold, pred_labels)
    balanced_accuracy = (recall + specificity) / 2
    kappa = cohen_kappa_score(gold, pred_labels)
    youdens_j = recall + specificity - 1
    markedness = precision + npv - 1

    # Log Loss（log(0) 回避のためクリップ）
    pred_clipped = [max(min(p, 1 - 1e-15), 1e-15) for p in pred_probs]
    logloss = log_loss(gold, pred_clipped)

    brier_score = np.mean([(p - g) ** 2 for p, g in zip(pred_probs, gold)])

    return {
        "auc": round(auc_score, 4),
        "pr_auc": round(pr_auc, 4),
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(balanced_accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "specificity": round(specificity, 4),
        "npv": round(npv, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "mcc": round(mcc, 4),
        "cohens_kappa": round(kappa, 4),
        "youdens_j": round(youdens_j, 4),
        "markedness": round(markedness, 4),
        "log_loss": round(logloss, 4),
        "brier_score": round(brier_score, 4),
        "correct_count": correct_count,
        "total_count": len(gold),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


# ========== メイン関数 ==========

def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "tloen/alpaca-lora-7b",
    test_data_path: str = "data/test.json",
    result_json_data: str = "temp.json",
    batch_size: int = 128,
    # --- MAML パラメータ ---
    use_adaptation: bool = True,
    support_size: int = 5,
    inner_steps: int = 10,
    inner_lr: float = 1e-2,
    support_data_path: Optional[str] = None,
    seed: int = 42,
    # --- サンプル記録パラメータ ---
    num_samples: int = 100,                   # 記録するサンプル数
    samples_output_file: Optional[str] = None,  # 出力先（None で自動生成）
):
    print(f"\nSetting seed to {seed} for reproducibility...\n")
    set_seed(seed)

    assert base_model, "Please specify a --base_model"

    # ========================================
    # 1. メタ情報の解析
    # ========================================
    model_type = lora_weights.split('/')[-1]
    model_name = '_'.join(model_type.split('_')[:2])
    train_sce = 'book' if 'book' in model_type else 'movie'
    test_sce = 'book' if 'book' in test_data_path else 'movie'

    temp_list = model_type.split('_')
    if len(temp_list) >= 2:
        checkpoint_seed = str(temp_list[-2])
        sample_key = str(temp_list[-1])
    else:
        checkpoint_seed = "seed_unknown"
        sample_key = model_type

    # ========================================
    # 2. 既存結果の読み込み・重複チェック
    # ========================================
    if os.path.exists(result_json_data):
        with open(result_json_data, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    # ネストされた辞書を初期化
    data.setdefault(train_sce, {})
    data[train_sce].setdefault(test_sce, {})
    data[train_sce][test_sce].setdefault(model_name, {})
    data[train_sce][test_sce][model_name].setdefault(checkpoint_seed, {})

    if sample_key in data[train_sce][test_sce][model_name][checkpoint_seed]:
        print(f"Result already exists for {checkpoint_seed}/{sample_key}. Exiting.")
        exit(0)

    # ========================================
    # 3. モデルとトークナイザのロード
    # ========================================
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model, load_in_8bit=load_8bit,
            torch_dtype=torch.bfloat16, device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model, lora_weights,
            torch_dtype=torch.bfloat16, device_map={'': 0},
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(
            model, lora_weights, device_map={"": device},
        )

    tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.bfloat16()

    # ========================================
    # 4. MAML適応（内部ループ）
    # ========================================
    if use_adaptation:
        print(f"\n{'='*50}")
        print("MAML Adaptation Mode")
        print(f"{'='*50}")
        print(f"Support size: {support_size}")
        print(f"Inner steps: {inner_steps}")
        print(f"Inner LR: {inner_lr}\n")

        with open(test_data_path, 'r') as f:
            test_data = json.load(f)

        # サポートセットとクエリセットの分割
        if support_data_path and os.path.exists(support_data_path):
            print(f"Using external support data: {support_data_path}")
            with open(support_data_path, 'r') as f:
                support_data_all = json.load(f)
            random.shuffle(support_data_all)
            support_data = support_data_all[:support_size]
            query_data = test_data
        else:
            print("Splitting test data into support/query")
            random.shuffle(test_data)
            support_data = test_data[:support_size]
            query_data = test_data[support_size:]

        print(f"Support set: {len(support_data)} samples")
        print(f"Query set: {len(query_data)} samples\n")

        # LoRAパラメータの勾配を有効化
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True

        saved_state = get_peft_model_state_dict(model)

        model.train()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"Trainable parameters: {len(trainable_params)}\n")

        if len(trainable_params) == 0:
            print("WARNING: No trainable parameters found!")
            use_adaptation = False
        else:
            print("Starting adaptation...")

            for step in range(inner_steps):
                # サポートセットでフォワードパス
                prompts = [
                    generate_prompt(d['instruction'], d['input']) + d['output']
                    for d in support_data
                ]
                inputs = tokenizer(
                    prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=512,
                ).to(device)
                labels = inputs['input_ids'].clone()
                labels[labels == tokenizer.pad_token_id] = -100

                outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    labels=labels,
                )
                loss = outputs.loss

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  Step {step + 1}/{inner_steps}: Invalid loss!")
                    continue

                # 勾配計算とパラメータ更新
                for param in trainable_params:
                    param.grad = None
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

                with torch.no_grad():
                    for param in trainable_params:
                        if param.grad is not None:
                            param_fp32 = param.data.float()
                            grad_fp32 = param.grad.float()
                            param.data = (param_fp32 - inner_lr * grad_fp32).to(param.dtype)

                print(f"  Step {step + 1}/{inner_steps}: Loss = {loss.item():.4f}")

            if device == "cuda":
                torch.cuda.empty_cache()

            print("\nAdaptation complete!")
            test_data = query_data

    model.eval()

    # ========================================
    # 5. バッチ評価関数
    # ========================================
    def evaluate_batch(instructions, inputs_list):
        """バッチ単位で推論し、Yes/No確率を返す"""
        prompts = [
            generate_prompt(inst, inp)
            for inst, inp in zip(instructions, inputs_list)
        ]
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
        ).to(device)

        generation_config = GenerationConfig(
            temperature=0, top_p=1.0, top_k=40, num_beams=1,
        )

        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=1,
            )

        scores = generation_output.scores[0].softmax(dim=-1)
        # Token 8241 = "Yes", Token 3782 = "No"
        yes_probs = scores[:, 8241].tolist()
        no_probs = scores[:, 3782].tolist()
        return yes_probs, no_probs

    # ========================================
    # 6. テストデータの読み込みと評価実行
    # ========================================
    if not use_adaptation:
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)

    instructions = [d['instruction'] for d in test_data]
    inputs_list = [d['input'] for d in test_data]
    gold = [int(d['output'] == 'Yes.') for d in test_data]
    gold_texts = [d['output'] for d in test_data]

    print(f"\nEvaluating {len(test_data)} samples...")

    all_yes_probs = []
    samples_to_save = []
    sample_count = 0

    total_batches = (len(test_data) - 1) // batch_size + 1
    for batch_data in tqdm(
        zip(
            batch_iter(instructions, batch_size),
            batch_iter(inputs_list, batch_size),
            batch_iter(gold, batch_size),
            batch_iter(gold_texts, batch_size),
        ),
        total=total_batches,
    ):
        inst_batch, inp_batch, gold_batch, gold_text_batch = batch_data
        yes_probs, no_probs = evaluate_batch(inst_batch, inp_batch)
        all_yes_probs.extend(yes_probs)

        # サンプル記録（最初の num_samples 件）
        for i in range(len(inst_batch)):
            if sample_count >= num_samples:
                continue
            yes_prob = yes_probs[i]
            no_prob = no_probs[i]
            pred_label = "Yes" if yes_prob > no_prob else "No"
            true_label = gold_batch[i]
            is_correct = (
                (pred_label == "Yes" and true_label == 1)
                or (pred_label == "No" and true_label == 0)
            )

            samples_to_save.append({
                "index": sample_count,
                "instruction": inst_batch[i],
                "input": inp_batch[i],
                "ground_truth": gold_text_batch[i],
                "prediction": pred_label,
                "yes_probability": round(yes_prob, 4),
                "no_probability": round(no_prob, 4),
                "correct": is_correct,
            })

            print(f"\n{'='*60}")
            print(f"Sample {sample_count + 1}/{num_samples}")
            print(f"{'='*60}")
            print(f"【Instruction】:\n{inst_batch[i]}")
            print(f"\n【Input】:\n{inp_batch[i]}")
            print(f"\n【Ground Truth】: {gold_text_batch[i]}")
            print(f"【Prediction】: {pred_label} (Yes: {yes_prob:.4f}, No: {no_prob:.4f})")
            print(f"【Correct】: {'✓' if is_correct else '✗'}")
            sample_count += 1

    # ========================================
    # 7. 評価指標の計算と表示
    # ========================================
    metrics = compute_all_metrics(gold, all_yes_probs)
    cm = metrics["confusion_matrix"]

    print(f"\n{'='*60}")
    print("=== Overall Results ===")
    print(f"{'='*60}")
    print(f"AUC Score:          {metrics['auc']:.4f}")
    print(f"PR-AUC:             {metrics['pr_auc']:.4f}")
    print(f"Accuracy:           {metrics['accuracy']:.4f} ({metrics['correct_count']}/{metrics['total_count']})")
    print(f"Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
    print(f"Precision:          {metrics['precision']:.4f}")
    print(f"Recall (TPR):       {metrics['recall']:.4f}")
    print(f"F1 Score:           {metrics['f1_score']:.4f}")
    print(f"Specificity (TNR):  {metrics['specificity']:.4f}")
    print(f"NPV:                {metrics['npv']:.4f}")
    print(f"FPR (Fall-out):     {metrics['fpr']:.4f}")
    print(f"FNR (Miss Rate):    {metrics['fnr']:.4f}")
    print(f"MCC:                {metrics['mcc']:.4f}")
    print(f"Cohen's Kappa:      {metrics['cohens_kappa']:.4f}")
    print(f"Youden's J:         {metrics['youdens_j']:.4f}")
    print(f"Markedness:         {metrics['markedness']:.4f}")
    print(f"Log Loss:           {metrics['log_loss']:.4f}")
    print(f"Brier Score:        {metrics['brier_score']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP={cm['tp']}, FP={cm['fp']}")
    print(f"  FN={cm['fn']}, TN={cm['tn']}")
    print(f"  Total Positive (Gold): {cm['tp'] + cm['fn']}")
    print(f"  Total Negative (Gold): {cm['tn'] + cm['fp']}")
    print(f"{'='*60}\n")

    # MAML状態を復元
    if use_adaptation:
        print("Restoring original model state...")
        set_peft_model_state_dict(model, saved_state)

    # ========================================
    # 8. 結果をファイルに保存
    # ========================================
    data[train_sce][test_sce][model_name][checkpoint_seed][sample_key] = metrics
    with open(result_json_data, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Results saved to: {result_json_data}")

    # サンプルを別ファイルに保存
    if samples_output_file is None:
        samples_output_file = result_json_data.replace('.json', f'_samples_{model_type}.json')

    samples_data = {
        "settings": {
            "base_model": base_model,
            "lora_weights": lora_weights,
            "test_data_path": test_data_path,
            "use_adaptation": use_adaptation,
            "support_size": support_size if use_adaptation else 0,
            "inner_steps": inner_steps if use_adaptation else 0,
            "inner_lr": inner_lr if use_adaptation else 0,
            "seed": seed,
        },
        "metrics": metrics,
        "num_samples": len(samples_to_save),
        "samples": samples_to_save,
    }

    with open(samples_output_file, 'w', encoding='utf-8') as f:
        json.dump(samples_data, f, indent=2, ensure_ascii=False)
    print(f"Samples ({len(samples_to_save)} items) saved to: {samples_output_file}")


if __name__ == "__main__":
    fire.Fire(main)
