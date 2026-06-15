import sys

import fire
import gradio as gr
import torch
torch.set_num_threads(1)
import transformers
import json
import os
import time
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from sklearn.metrics import roc_auc_score
import numpy as np
import random
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def set_seed(seed):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "tloen/alpaca-lora-7b",
    test_data_path: str = "data/test.json",
    result_json_data: str = "temp.json",
    batch_size: int = 128,
    share_gradio: bool = False,
    # MAML Parameters
    use_adaptation: bool = True,
    support_size: int = 5,  # k_shotと統一（訓練時と同じ値）
    inner_steps: int = 10,
    inner_lr: float = 1e-2,  # 訓練時と同じ学習率
    support_data_path: str = None,
    seed: int = 42,  # Seed値を追加
):
    # Seed値を固定
    print(f"\n{'='*50}")
    print(f"Setting seed to {seed} for reproducibility...")
    print(f"{'='*50}\n")
    set_seed(seed)
    
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    model_type = lora_weights.split('/')[-1]
    model_name = '_'.join(model_type.split('_')[:2])

    if model_type.find('book') > -1:
        train_sce = 'book'
    else:
        train_sce = 'movie'
    
    if test_data_path.find('book') > -1:
        test_sce = 'book'
    else:
        test_sce = 'movie'
    
    temp_list = model_type.split('_')
    if len(temp_list) >= 2:
        checkpoint_seed = temp_list[-2]
        sample = temp_list[-1]
    else:
        # checkpoint-800などの場合、命名規則が異なるためダミー値を入れるか、
        # 親ディレクトリ名から推測する等の処理が必要ですが、
        # ここではとりあえずエラーで止まらないように現在のフォルダ名を使用します。
        checkpoint_seed = "seed_unknown"
        sample = model_type
    
    # データ構造の初期化（文字列キーであることを保証）
    checkpoint_seed = str(checkpoint_seed)
    sample = str(sample)
    
    if os.path.exists(result_json_data):
        f = open(result_json_data, 'r')
        data = json.load(f)
        f.close()
    else:
        data = dict()
    
    if train_sce not in data:
        data[train_sce] = {}
    if test_sce not in data[train_sce]:
        data[train_sce][test_sce] = {}
    if model_name not in data[train_sce][test_sce]:
        data[train_sce][test_sce][model_name] = {}
    if checkpoint_seed not in data[train_sce][test_sce][model_name]:
        data[train_sce][test_sce][model_name][checkpoint_seed] = {}
    if sample in data[train_sce][test_sce][model_name][checkpoint_seed]:
        print(f"Result already exists for {checkpoint_seed}/{sample}. Exiting.")
        exit(0) 


    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.bfloat16,
            device_map={'': 0}
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )


    tokenizer.padding_side = "left"
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        # model.half()  # seems to fix bugs for some users.
        model.bfloat16()  # seems to fix bugs for some users.

    # === MAML Adaptation ===
    if use_adaptation:
        print(f"\n{'='*50}")
        print(f"MAML Adaptation Mode")
        print(f"{'='*50}")
        print(f"Support size: {support_size}")
        print(f"Inner steps: {inner_steps}")
        print(f"Inner LR: {inner_lr}\n")
        
        # Load test data
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        # Prepare support data
        if support_data_path and os.path.exists(support_data_path):
            print(f"Using external support data: {support_data_path}")
            with open(support_data_path, 'r') as f:
                support_data_all = json.load(f)
            random.shuffle(support_data_all)
            support_data = support_data_all[:support_size]
            query_data = test_data
        else:
            print(f"Splitting test data into support/query")
            random.shuffle(test_data)
            support_data = test_data[:support_size]
            query_data = test_data[support_size:]
        
        print(f"Support set: {len(support_data)} samples")
        print(f"Query set: {len(query_data)} samples\n")
        
        # Save support set to file
        support_set_file = result_json_data.replace('.json', '_support_set.json')
        with open(support_set_file, 'w') as f:
            json.dump(support_data, f, indent=2, ensure_ascii=False)
        print(f"Support set saved to: {support_set_file}")
        
        # Display support set samples
        print(f"\n{'='*60}")
        print("SUPPORT SET SAMPLES")
        print(f"{'='*60}")
        for i, support_sample in enumerate(support_data):
            gold_text = support_sample['output']
            print(f"\nSupport Sample {i+1}/{len(support_data)}:")
            print(f"  Instruction: {support_sample['instruction'][:80]}...")
            print(f"  Input: {support_sample['input'][:80]}...")
            print(f"  Output (Gold): {gold_text}")
        print(f"{'='*60}\n")
        
        # Enable gradients for LoRA parameters
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
        
        # Save original state
        saved_state = get_peft_model_state_dict(model)
        
        # Adaptation on support set
        model.train()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"Trainable parameters: {len(trainable_params)}\n")
        
        # Check if we actually have trainable parameters
        if len(trainable_params) == 0:
            print("WARNING: No trainable parameters found!")
            print("Skipping adaptation and using original model.")
            test_data = test_data  # Use original test data
            use_adaptation = False
        else:
            print("Starting adaptation...")
            
            # デバッグ: 最初のLoRAパラメータの値を記録
            first_lora_param = None
            first_lora_name = None
            for name, param in model.named_parameters():
                if 'lora' in name.lower() and param.requires_grad:
                    first_lora_param = param.data.clone()
                    first_lora_name = name
                    print(f"  Tracking parameter: {name}")
                    print(f"  Initial value sample: {param.data.flatten()[:5]}")
                    break
            
            # Calculate initial loss before adaptation
            prompts_init = [generate_prompt(d['instruction'], d['input']) + d['output'] for d in support_data]
            inputs_init = tokenizer(prompts_init, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            labels_init = inputs_init['input_ids'].clone()
            labels_init[labels_init == tokenizer.pad_token_id] = -100
            
            with torch.no_grad():
                outputs_init = model(
                    input_ids=inputs_init['input_ids'],
                    attention_mask=inputs_init['attention_mask'],
                    labels=labels_init
                )
                initial_loss = outputs_init.loss.item()
            print(f"  Initial Loss: {initial_loss:.4f}\n")
            
            for step in range(inner_steps):
                # Prepare prompts
                prompts = [generate_prompt(d['instruction'], d['input']) + d['output'] for d in support_data]
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                
                # Prepare labels
                labels = inputs['input_ids'].clone()
                labels[labels == tokenizer.pad_token_id] = -100
                
                # Forward pass
                outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    labels=labels
                )
                loss = outputs.loss
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  Step {step+1}/{inner_steps}: Invalid loss!")
                    continue
                
                # Backward pass
                for param in trainable_params:
                    param.grad = None
                loss.backward()
                
                # Check gradient norms
                grad_norms = []
                for param in trainable_params:
                    if param.grad is not None:
                        grad_norms.append(param.grad.norm().item())
                avg_grad = sum(grad_norms) / len(grad_norms) if grad_norms else 0
                
                # Gradient clipping
                total_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                
                # Update parameters (in float32 for precision)
                with torch.no_grad():
                    for param in trainable_params:
                        if param.grad is not None:
                            # Convert to float32 for precise update
                            param_fp32 = param.data.float()
                            grad_fp32 = param.grad.float()
                            param_fp32 = param_fp32 - inner_lr * grad_fp32
                            # Convert back to original dtype
                            param.data = param_fp32.to(param.dtype)
                
                # デバッグ: 最初のステップで更新を確認
                if step == 0 and first_lora_param is not None:
                    for name, param in model.named_parameters():
                        if name == first_lora_name:
                            # float32で比較（精度の問題を避ける）
                            param_change_fp32 = (param.data.float() - first_lora_param.float()).abs().max().item()
                            param_change_bf16 = (param.data - first_lora_param).abs().max().item()
                            print(f"  [DEBUG] Parameter change (float32): {param_change_fp32:.10f}")
                            print(f"  [DEBUG] Parameter change (bfloat16): {param_change_bf16:.10f}")
                            print(f"  [DEBUG] Updated value sample: {param.data.flatten()[:5]}")
                            print(f"  [DEBUG] Original value sample: {first_lora_param.flatten()[:5]}")
                            print(f"  [DEBUG] Learning rate: {inner_lr}")
                            if param.grad is not None:
                                print(f"  [DEBUG] Gradient sample: {param.grad.flatten()[:5]}")
                                print(f"  [DEBUG] Expected change: ~{(inner_lr * param.grad.abs().max()).item():.10f}")
                            break
                
                print(f"  Step {step+1}/{inner_steps}: Loss = {loss.item():.4f}, Grad Norm = {avg_grad:.6f}")
            
            # GPUメモリをクリア
            if device == "cuda":
                torch.cuda.empty_cache()
            
            print(f"\nAdaptation complete!")
            # inner_steps > 0 の場合のみ損失の変化を表示
            if inner_steps > 0:
                print(f"Loss change: {initial_loss:.4f} -> {loss.item():.4f} (Delta: {loss.item() - initial_loss:.4f})")
            else:
                print(f"No adaptation performed (inner_steps=0). Initial Loss: {initial_loss:.4f}")
            print(f"{'='*50}\n")
            
            # Use query data for testing
            test_data = query_data

    model.eval()
    # torch.compileは大量のメモリを使用するため無効化
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    def evaluate(
        instructions,
        inputs=None,
        gold_labels=None,  # 正解ラベルを追加
        temperature=0,
        top_p=1.0,
        top_k=40,
        num_beams=1,
        max_new_tokens=1,  # メモリ削減: 128トークン生成
        batch_size=4,
        show_top4=False,  # Top4表示を制御
        profile_timing=False,  # タイミング計測を制御
        **kwargs,
    ):
        timings = {}
        
        # Tokenization
        t0 = time.time()
        prompt = [generate_prompt(instruction, input) for instruction, input in zip(instructions, inputs)]
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        timings['tokenization'] = time.time() - t0
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        
        # Model Inference
        t0 = time.time()
        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                # batch_size=batch_size,
            )
        timings['inference'] = time.time() - t0
        
        s = generation_output.sequences
        scores = generation_output.scores[0].softmax(dim=-1)

        # Top4表示は最初のバッチのみ（メモリ削減）
        t0 = time.time()
        if show_top4:
            top4_probs, top4_indices = torch.topk(scores, k=4, dim=-1)
            print("\n=== Top 4 Token Probabilities ===")
            for batch_idx in range(min(3, scores.shape[0])):  # 最初の3サンプルを表示
                print(f"\nSample {batch_idx}:")
                # 正解ラベルを表示
                if gold_labels is not None and batch_idx < len(gold_labels):
                    gold_text = "Yes." if gold_labels[batch_idx] == 1 else "No."
                    print(f"  Ground Truth: {gold_text} (Label: {gold_labels[batch_idx]})")
                for rank in range(4):
                    token_id = top4_indices[batch_idx, rank].item()
                    prob = top4_probs[batch_idx, rank].item()
                    token_text = tokenizer.decode([token_id])
                    print(f"  Rank {rank+1}: '{token_text}' (ID: {token_id}, Prob: {prob:.4f})")
            print("="*40 + "\n")
        timings['top4'] = time.time() - t0
        
        # Decoding
        t0 = time.time()
        # scores is already softmax'd, so just extract the Yes/No token probabilities
        # Token 8241 = "Yes", Token 3782 = "No"
        batch_logits = torch.tensor(scores[:, [8241, 3782]])  # [batch_size, 2]
        logits = batch_logits.tolist()  # No additional softmax needed!
        
        input_ids = inputs["input_ids"].to(device)
        L = input_ids.shape[1]
        s = generation_output.sequences
        output = tokenizer.batch_decode(s, skip_special_tokens=True)
        timings['decoding'] = time.time() - t0
        
        # Postprocessing
        t0 = time.time()
        output = [_.split('Response:\n')[-1] for _ in output]
        timings['postprocessing'] = time.time() - t0
        
        if profile_timing:
            return output, logits, timings
        return output, logits
        
    # testing code for readme
    logit_list = []
    gold_list= []
    outputs = []
    logits = []
    from tqdm import tqdm
    gold = []
    pred = []

    # Load test data if not already loaded by MAML
    if not use_adaptation:
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
    
    instructions = [_['instruction'] for _ in test_data]
    inputs = [_['input'] for _ in test_data]
    gold = [int(_['output'] == 'Yes.') for _ in test_data]
    def batch(list, batch_size=32):
        chunk_size = (len(list) - 1) // batch_size + 1
        for i in range(chunk_size):
            yield list[batch_size * i: batch_size * (i + 1)]
    
    print(f"\nEvaluating {len(test_data)} samples...")
    
    # Timing statistics
    total_batches = (len(test_data) - 1) // batch_size + 1
    timing_stats = {
        'tokenization': [],
        'inference': [],
        'decoding': [],
        'postprocessing': [],
        'top4': []
    }
    
    test_start_time = time.time()
    
    for i, batch_data in tqdm(enumerate(zip(batch(instructions), batch(inputs), batch(gold)))):
        instructions_batch, inputs_batch, gold_batch = batch_data
        # 最初のバッチのみTop4を表示
        show_top4 = (i == 0)
        # 最初の5バッチでタイミングを計測
        profile_timing = (i < 5)
        
        if profile_timing:
            output, logit, timings = evaluate(instructions_batch, inputs_batch, gold_labels=gold_batch, show_top4=show_top4, profile_timing=True)
            for key in timing_stats:
                timing_stats[key].append(timings.get(key, 0))
        else:
            output, logit = evaluate(instructions_batch, inputs_batch, gold_labels=gold_batch, show_top4=show_top4)
        
        outputs = outputs + output
        logits = logits + logit
    
    # メモリ削減: logitsをtest_dataに保存しない（必要な場合のみコメント解除）
    # for i, test in tqdm(enumerate(test_data)):
    #     test_data[i]['predict'] = outputs[i]
    #     test_data[i]['logits'] = logits[i]
    
    pred = [logit[0] for logit in logits]

    from sklearn.metrics import roc_auc_score

    if np.isnan(pred).any() or np.isnan(gold).any():
        print("NaN detected in pred or gold!")
        print("pred:", pred)
        print("gold:", gold)

    auc_score = roc_auc_score(gold, pred)
    
    # Display timing statistics
    test_total_time = time.time() - test_start_time
    if timing_stats['inference']:
        print(f"\n{'='*60}")
        print("TIMING PROFILING RESULTS (Average per batch)")
        print(f"{'='*60}")
        
        avg_tokenization = np.mean(timing_stats['tokenization'])
        avg_inference = np.mean(timing_stats['inference'])
        avg_decoding = np.mean(timing_stats['decoding'])
        avg_postprocessing = np.mean(timing_stats['postprocessing'])
        avg_top4 = np.mean(timing_stats['top4']) if timing_stats['top4'] else 0
        
        print(f"Tokenization:       {avg_tokenization:.3f}s")
        print(f"Model Inference:    {avg_inference:.3f}s  <-- MAIN BOTTLENECK")
        print(f"Decoding:           {avg_decoding:.3f}s")
        print(f"Postprocessing:     {avg_postprocessing:.3f}s")
        if avg_top4 > 0:
            print(f"Top4 Calculation:   {avg_top4:.3f}s (first batch only)")
        
        per_batch = avg_tokenization + avg_inference + avg_decoding + avg_postprocessing
        per_sample = per_batch / batch_size
        
        print(f"\n{'-'*60}")
        print(f"Per-batch time:     {per_batch:.3f}s")
        print(f"Per-sample time:    {per_sample:.3f}s")
        print(f"Batch size:         {batch_size}")
        print(f"Total batches:      {total_batches}")
        print(f"Actual total time:  {test_total_time:.1f}s = {test_total_time/60:.1f} minutes")
        
        print(f"\n{'='*60}")
        print("BREAKDOWN")
        print(f"{'='*60}")
        total_processing = per_batch
        print(f"Tokenization:    {avg_tokenization/total_processing*100:.1f}%")
        print(f"Inference:       {avg_inference/total_processing*100:.1f}%")
        print(f"Decoding:        {avg_decoding/total_processing*100:.1f}%")
        print(f"Postprocessing:  {avg_postprocessing/total_processing*100:.1f}%")
        
        inference_pct = avg_inference / total_processing * 100
        print(f"\n{'='*60}")
        print("OPTIMIZATION SUGGESTIONS")
        print(f"{'='*60}")
        
        if inference_pct > 70:
            print(f"🔴 Inference is the main bottleneck ({inference_pct:.1f}%)")
            print("   Suggestions:")
            print(f"   - Increase batch_size (currently {batch_size})")
            print("   - Use torch.compile() (PyTorch 2.0+)")
            print("   - Use Flash Attention")
            print("   - Consider model quantization (8-bit)")
        elif inference_pct > 50:
            print(f"🟡 Inference is a significant bottleneck ({inference_pct:.1f}%)")
            print("   Consider increasing batch_size")
        
        if avg_tokenization / total_processing > 0.1:
            print(f"🟡 Tokenization takes {avg_tokenization/total_processing*100:.1f}% of time")
            print("   Consider pre-tokenizing data")
        
        if avg_decoding / total_processing > 0.1:
            print(f"🟡 Decoding takes {avg_decoding/total_processing*100:.1f}% of time")
            print("   This is normal, but can be reduced by avoiding decode if not needed")
        
        print(f"\n{'='*60}\n")
    
    print(f"\n{'='*50}")
    print(f"AUC Score: {auc_score:.4f}")
    print(f"{'='*50}\n")
    
    # Restore original model state if MAML was used
    if use_adaptation:
        print("Restoring original model state...")
        set_peft_model_state_dict(model, saved_state)
    
    # デバッグ: 保存前に型を確認
    print(f"\n{'='*50}")
    print("DEBUG: Before saving results")
    print(f"{'='*50}")
    print(f"train_sce: {train_sce} (type: {type(train_sce).__name__})")
    print(f"test_sce: {test_sce} (type: {type(test_sce).__name__})")
    print(f"model_name: {model_name} (type: {type(model_name).__name__})")
    print(f"checkpoint_seed: {checkpoint_seed} (type: {type(checkpoint_seed).__name__})")
    print(f"sample: {sample} (type: {type(sample).__name__})")
    print(f"auc_score: {auc_score} (type: {type(auc_score).__name__})")
    
    # すべてのキーを文字列に変換（辞書型の場合はエラーメッセージを表示）
    def safe_str_convert(value, name):
        if isinstance(value, dict):
            print(f"ERROR: {name} is a dict: {value}")
            raise TypeError(f"{name} cannot be a dict!")
        return str(value)
    
    train_sce = safe_str_convert(train_sce, "train_sce")
    test_sce = safe_str_convert(test_sce, "test_sce")
    model_name = safe_str_convert(model_name, "model_name")
    checkpoint_seed = safe_str_convert(checkpoint_seed, "checkpoint_seed")
    sample = safe_str_convert(sample, "sample")
    
    print(f"\nAfter conversion:")
    print(f"checkpoint_seed: {checkpoint_seed} (type: {type(checkpoint_seed).__name__})")
    print(f"sample: {sample} (type: {type(sample).__name__})")
    print(f"{'='*50}\n")
    
    data[train_sce][test_sce][model_name][checkpoint_seed][sample] = auc_score
    f = open(result_json_data, 'w')
    json.dump(data, f, indent=4)
    f.close()
    
    print(f"Results saved to: {result_json_data}")

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{instruction}

### Response:
"""


if __name__ == "__main__":
    fire.Fire(main)
