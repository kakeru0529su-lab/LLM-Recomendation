# TALLRec+MeTA-LoRA
本研究はLLMベース推薦手法であるTALLRecにMeTA-LoRAを統合し、推論時適応を用いたクロスドメイン推薦手法の実装です

初めて実行する際は1.初期設定から、次からは2.コンテナの立ち上げから実行してください。
## 1. 初期設定
### 1.1. ディレクトリ構造
```
Programs/
├ docker/
│  │
│  ├ Dockerfile
│  ├ docker-compose.yml
│  ├ requirements.txt
│  └ times.ttf
│
└ TALLRec-main/
   │
   ├ finetune.py
   ├ finetune_rec_my.py
   ├ finetune_multi_rec_my.py
   ├ finetune_multi_rec_MeTA_MAML.py
   │
   ├ evaluate_maml_my.py
   ├ evaluate_maml_with_samples.py
   │
   ├ export_hf_checkpoint.py
   ├ export_state_dict_checkpoint.py
   │
   ├ data/
   │  ├ book/
   │  │  ├ train.json
   │  │  ├ valid.json
   │  │  └ test.json
   │  │
   │  ├ movie/
   │  │  ├ train.json
   │  │  ├ valid.json
   │  │  └ test.json
   │  │
   │  └ music/
   │     ├ train.json
   │     ├ valid.json
   │     └ test.json
   │
   ├ alpaca-lora-7B/
   │
   ├ lora-alpaca/
   │
   ├ models--baffo32--decapoda-research-llama-7B-hf/
   │
   └ shell/
      ├ instruct_7B.sh
      ├ instruct_multi_7B.sh
      ├ instruct_multi_MeTA_7B.sh
      │
      ├ evaluate.sh
      ├ evaluate_book.sh
      ├ evaluate_movie.sh
      └ evaluate_music.sh
      
```
    
### 1.2. Docker環境構築
ここでは~/Programs/直下にコピーする例を示します。適宜変更してください。
~/Programs/直下にusbからdocker・TALLRec-mainをコピーして
```
Programs/
├ docker/
│
└ TALLRec-main/
```
となるようにし,以下を実行してください。

~~~
$ cd ~/Programs/docker
$ docker build -t TALLRec .
~~~

### 1.3. コンテナの立ち上げ

~~~bash
$ docker-compose up -d
$ docker exec -it tallrec bash
~~~

## 2. モデルの学習
### 2.1 従来手法
#### 2.1.1 TALLRec-Single
`~/shell/instruct_7B`内の`train_data`,`val_data`を任意のドメインに変更して実行してください
(例:train_data = './data/movie/train.json')
`output_dir`を識別可能に変更してください

~~~bash
$ bash ./shell/instruct_7B.sh < GPU_ID >
~~~
(例:bash ./shell/instruct_7B.sh 0,1)

出力例:
```
./experiments/experiments_15_16/
      ├ adapter_config.json (モデルのパラメータ記載)
      └ adapter_model.safetensors (モデルの重み)
```

#### 2.1.2 TALLRec-Multi
`~/shell/instruct_multi_7B`内の`train_data1`,`val_data1`,`train_data2`,`val_data2`を任意のドメインに変更して実行してください
(例:train_data1 = './data/movie/train.json')
`output_dir`を識別可能に変更してください

~~~bash
$ bash ./shell/instruct_multi_7B.sh < GPU_ID >
~~~

出力例:
```
./experiments/multi/experiments_multi_15_16/
      ├ adapter_config.json (モデルのハイパーパラメータ)
      ├ adapter_model.safetensors (モデルの重み)
      ├ auc_log.txt (パラメータ更新10回毎のval_auc)
      ├ used_train_data.json (使用した学習データ)
      └ used_val_data.json (使用した検証データ)

```

#### 2.2 提案手法
`~/shell/instruct_multi_MeTA_7B`内の`train_data1`,`val_data1``train_data2`,`val_data2`を任意のドメインに変更して実行してください
(例:train_data1 = './data/movie/train.json')
`output_dir`を識別可能に変更してください

~~~bash
$ bash ./shell/instruct_multi_MeTA_7B.sh < GPU_ID >
~~~

出力例:
```
./experiments/multi_MeTA/experiments_multi_15_16/
   ├ checkpoint-10/
   │  ├ adapter_config.json (モデルのハイパーパラメータ)
   │  └ adapter_model.safetensors (モデルの重み)
   ├ checkpoint-20/
   ├ ・
   ├ ・
   ├ ・
   ├ checkpoint-200/
   │
   ├ auc_history.json (パラメータ更新10回毎のval_auc)
   ├ used_train_data.json (使用した学習データ)
   └ used_val_data.json (使用した検証データ)
```

## 3. モデルの推論
推論時適応を行う場合は`use_adaptation=True`に,推論時適応を行わない場合は`use_adaptation=False`にしてください
`result_dir`を識別可能に変更してください
提案手法では`val_auc`が最高のcheckpointにおいて推論を行ってください

~~~bash
$ bash ./shell/evaluate_[domain].sh < GPU_ID > < output_dir >
~~~
(例:bash ./shell/evaluate_music.sh 0,1 ./experiments/multi_MeTA/experiments_multi_MeTA_15_16)

出力例:
```
./experiments/multi_MeTA/experiments_multi_15_16/
   ├ checkpoint-10/
   │  ├ adapter_config.json (モデルのハイパーパラメータ)
   │  └ adapter_model.safetensors (モデルの重み)
   ├ checkpoint-20/
   ├ ・
   ├ ・
   ├ ・
   ├ checkpoint-200/
   │
   ├ result_music/
   │  ├ experiments_multi_MeTA_15_16_support_set.json (サポートセットの具体例)
   │  └ experiments_multi_MeTA15_16.json (結果)
   │
   ├ auc_history.json (パラメータ更新10回毎のval_auc)
   ├ used_train_data.json (使用した学習データ)
   └ used_val_data.json (使用した検証データ)
```

## 4. 結果
### 4.1. 出力ファイルの中身
`./experiments/multi_MeTA/experiments_multi_MeTA_12_16/result_music/checkpoint-170.json`を例に挙げます

```
{
    "movie": {
        "movie": {
            "checkpoint-170": {
                "seed_unknown": {
                    "checkpoint-170": 0.698612571081194
                }
            }
        }
    }
}
```

(`evaluate_maml_with_samples.py`を用いて推論を行うと詳細な実験結果が得られます

例:
```
{
    "movie": {
        "movie": {
            "checkpoint-170": {
                "seed_unknown": {
                    "checkpoint-170": {
                        "auc": 0.6988,
                        "pr_auc": 0.9303,
                        "accuracy": 0.6667,
                        "balanced_accuracy": 0.6521,
                        "precision": 0.9254,
                        "recall": 0.6717,
                        "f1_score": 0.7784,
                        "specificity": 0.6324,
                        "npv": 0.2211,
                        "fpr": 0.3676,
                        "fnr": 0.3283,
                        "mcc": 0.2111,
                        "cohens_kappa": 0.1696,
                        "youdens_j": 0.3041,
                        "markedness": 0.1465,
                        "log_loss": 0.8936,
                        "brier_score": 0.255,
                        "correct_count": 2626,
                        "total_count": 3939,
                        "confusion_matrix": {
                            "tp": 2306,
                            "fp": 186,
                            "fn": 1127,
                            "tn": 320
                        }
                    }
                }
            }
        }
    }
}
```
)

### 4.2. 推論結果例(music)
| seed | book | book_no | movie | movie_no | both | both_no | MeTA | MeTA_no |
|------|------|--------|------|---------|------|--------|------|--------|
| 12 |  | 0.50891 |  | 0.5899 |  | 0.59085 | 0.69861 |  |
| 13 |  | 0.38342 |  | 0.5418 |  | 0.51892 | 0.5769 |  |
| 14 |  | 0.50944 |  | 0.53478 |  | 0.6043 | 0.67085 |  |
| 15 | 0.54576 | 0.56913 | 0.61982 | 0.57535 | 0.62578 | 0.63716 | 0.62719 | 0.56892 |
| 16 | 0.47428 | 0.47502 | 0.5588 | 0.54867 | 0.50864 | 0.50977 | 0.55426 | 0.56883 |
| 17 | 0.47403 | 0.47935 | 0.47429 | 0.49864 | 0.51844 | 0.48184 | 0.64959 | 0.61061 |
| 平均 | 0.4980 | 0.4875 | 0.5510 | 0.5482 | 0.5510 | 0.5571 | 0.6296 | 0.5828 |
| 標準偏差 | 0.0413 | 0.0611 | 0.0731 | 0.0321 | 0.0650 | 0.0619 | 0.0554 | 0.0241 |

卒業論文におけるProposedの値を再現するには

学習(seed = 15, 16, 17):
`bash ./shell/instruct_multi_MeTA_7B.sh 0,1`

推論:
`bash ./shell/evaluate_music.sh 0,1 ./experiments/multi_MeTA/experiments_multi_MeTA_12_16/checkpoint-170`

の3seedの平均を取ると0.6103が得られます


## 5. アブレーション実験
### 5.1 w/o メタ学習
TALLRec-Bothに対して推論時適応を行わずに実行してください

### 5.2 w/o Phase I
`inner_adaptation_step`を0にして実行してください

### 5.3 w/o Phase II
TALLRec-Bothに対して推論時適応を行ってください

### 5.4 w/o Phase III
提案手法に対して`use_adaptation=False`にして推論を行ってください

## 6. 提案手法の実装
本研究ではTALLRecにMeTA-LoRAを統合し、推論時適応を導入しました
提案手法の実装は主に`finetune_multi_rec_MeTA_MAML.py`と`evaluate_maml_with_samples.py`に記述しています

TALLRecではTransformersのTrainerを用いたFine-Tuningを行っていた(`TALLRec-Both`)のに対して提案手法ではメタ学習を導入するためにTrainerを使用せずに実装を行いました

### 6.1. データ分割
メタ学習におけるタスクをドメインとして扱い、Support set, Query setを作成します(finetune_multi_rec_MeTA_MAML[Line:206-251])
これにより以下のように各ドメインに対してデータが分割されます
```
'task_0': {'support': (サポートセット), 'query': (クエリセット)}
'task_1': {'support': (サポートセット), 'query': (クエリセット)}
```

### 6.2. タスク固有適応
Support setを用いてLoRAパラメータを`Inner Adaptation Steps`の数だけ更新します(finetune_multi_rec_MeTA_MAML[Line:511-523)]
これにより各ドメインに適応したパラメータが得られます

その後タスク固有適応で適応したモデルを用いてQuery setの損失を計算します(finetune_multi_rec_MeTA_MAML[Line;525-543])
これにより各ドメインにおける勾配が得られます

### 6.3. メタ知識更新
複数タスクのメタ勾配を平均しパラメータ更新を行います(finetune_multi_rec_MeTA_MAML[Line:546-551])
これにより少数データで適応可能な初期値が学習されます

### 6.4. 推論時適応
推論時にも Support setを用いた適応を行います(finetune_multi_rec_MeTA_MAML[Line:227-320])
これによりターゲットドメインに対するFew-shot適応が可能になる。

### 6.5. 量子化
メタ学習では勾配計算を複数回行うため，量子化を用いると計算が不安定になる場合があります

そのため
```
load_in_8bit=False,
torch_dtype=torch.bfloat16
```
として8bit量子化を使用せずbfloat16で学習を行います(finetune_multi_rec_MeTA_MAML[Line:128-129])


## 7. 実験設定
### 7.1 ハイパーパラメータ
#### 7.1.1 モデル設定
| Parameter | Value | Description |
|------|------|--------|
| Base Model | baffo32/decapoda-research-llama-7B-hf | 事前学習済みLLM |
| Instruction Model | alpaca-lora-7B | Alpaca-Tuning |
| LoRA Rank (r) | 8 | LoRA低ランク行列の次元 |
| LoRA Alpha | 16 | LoRA更新スケーリング係数 |
| LoRA Dropout | 0.05 | LoRA層のドロップアウト |
| LoRA Target Modules | q_proj, v_proj | LoRAを挿入するAttention層 |

#### 7.1.2 学習時設定

| Parameter | Value | Description |
|------|------|--------|
| Batch Size | 64 | バッチサイズ |
| Micro Batch Size | 32 | GPU1回のミニバッチ |
| Epochs | 200 | エポック数 |
| Learning Rate | 1e-4 | メタ知識更新の学習率 |
| Cutoff Length | 512 | 入力トークン最大長 |
| Seed | (12, 13, 14) 15, 16, 17 | 乱数シード |
| Inner Adaptation Steps | 5 | タスク固有適応の更新回数 |
| Inner Learning Rate | 1e-2 | タスク固有適応の学習率 |
| Sample Size | 16 | 学習時のサンプル数 |
| k-shot | 5 | few-shot 学習時のサポートセット数 |


#### 7.1.3 推論時設定
| Parameter | Value | Description |
|------|------|--------|
| support_size | 5 | 推論時のサポートセット数 |
| inner_steps | 10 | 推論時の更新回数 |
| inner_lr | 1e-2 | 推論時の学習率 |

### 7.2 使用データ
book   : Book-Crossing 
(https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset?resource=download-directory)

movie  : MovieLens
(https://grouplens.org/datasets/movielens/)

music  : Amazon Review Data (2018)(CD&Vinyl)
(https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)

### 7.3 使用ベースLLM
LLaMA 7B (baffo32/decapoda-research-llama-7B-hf)
(https://huggingface.co/baffo32/decapoda-research-llama-7B-hf)

## 8. 注意点
MeTA-LoRAのコードが配布されていなかったため、手動で実装しました
LoRArankを変更して実行する場合は指示追従能力向上のためのAlpaca-Tuningのrankを`finetune.py`を用いて変更してからRec-Tuningを行う必要があります

## 9. 経験的知見
### 9.1. 量子化
既存のTALLRec実装では8bit量子化（bitsandbytes）を使用していたが，本研究ではメタ学習を導入したため量子化を使用すると勾配計算が不安定になる問題が確認されました
そのため本研究では量子化を使用せず，bfloat16 を用いて学習を行いました

### 9.2. メタ学習の導入
単純なマルチドメイン学習（TALLRec-Both）ではドメイン間の干渉が発生し，ターゲットドメインの性能が低下する場合がありました
そこで提案手法ではメタ学習により少数データで適応可能な初期パラメータ を学習することで，クロスドメイン推薦において性能改善が確認されました


