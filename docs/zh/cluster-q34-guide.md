# Q3.4 在 NEU Explorer 叢集上跑超參搜尋

## 0. 前提：帳號

NEU 的 HPC 叢集現在叫 **Explorer**（取代舊的 Discovery），用 Slurm 排程。

- 如果你還沒有帳號，到 [rc.northeastern.edu](https://rc.northeastern.edu/) 送 **Access Request** ticket，
  或寄信到 `rchelp@northeastern.edu`。通常 1 個工作日內開通。
- 確認能登入：`ssh <你的NEU帳號>@login.explorer.northeastern.edu`（需要 Duo 二階段驗證）
- 也可以用網頁版 [ood.explorer.northeastern.edu](http://ood.explorer.northeastern.edu)（Open OnDemand），
  有檔案瀏覽器和終端機，第一次用會比較直覺。

### 問課程助教一件事
INFO7375 如果有向 Research Computing 登記，你會有 `courses-gpu` partition 的權限
（預設 4 小時、上限 8 小時），這比一般的 `gpu-short`（上限 2 小時）寬鬆很多，
**Q3.5 的最終訓練會需要它**。先問清楚，可以省下之後拆 job 的麻煩。

## 1. 把程式碼傳上去

`.venv/`（1GB+）和 `data/tokens.npz`（544MB）都不要傳，到叢集上重建比較快：

```bash
cd "<repo root>"
rsync -av --progress \
  --exclude '.venv' --exclude 'data/tokens.npz' --exclude '__pycache__' \
  --exclude '.pytest_cache' --exclude 'wandb' --exclude 'outputs' \
  A2/ <你的帳號>@login.explorer.northeastern.edu:~/hw2/
```

## 2. 建環境（在 login node 跑一次）

```bash
ssh <你的帳號>@login.explorer.northeastern.edu
cd ~/hw2
bash slurm/setup-explorer.sh
```

⚠️ **這個腳本裡的 module 名稱我沒辦法替你驗證**。先跑 `module avail anaconda` 和
`module avail cuda` 看實際有哪些版本，再對應改 `slurm/setup-explorer.sh` 和
`slurm/q34-sweep.sbatch` 開頭的 `module load` 那行。腳本已經寫了 `||` fallback，但不保證命中。

## 3. 登入 wandb（順便解決 Q3.3 的待辦）

```bash
wandb login     # 貼上 https://wandb.ai/authorize 的 API key，會寫進 ~/.netrc
```

Batch job 沒辦法互動輸入，所以**一定要先在 login node 做這步**，否則 job 會卡住或改用 offline 模式。

> 這同時解決了你 Q3.3 缺訓練曲線截圖的問題——之後所有 run 都會同步到 wandb 雲端，
> 你直接上網站截圖就好。

## 4. 送出 Q3.4 的 sweep

```bash
cd ~/hw2
sbatch slurm/q34-sweep.sbatch
```

監看：

```bash
squeue -u $USER                      # 排隊/執行狀態
tail -f slurm/logs/q34-<jobid>.out   # 即時輸出
scancel <jobid>                      # 要取消的話
```

job 結尾會自動印出五組的 `eval.json` 彙總。

## 5. 五組配置的設計理由

**最重要的隱形約束：Gradescope 限制模型 100MB**，而 `train.py` 在 98MB 就會警告。
因為有 weight tying，參數量主要是 token embedding `50257 × n_embd`，所以：

| n_embd | 模型大小 | 可用? |
|---|---|---|
| 256 | 71 MB | ✅ |
| 320 | 94 MB | ✅ 上限 |
| 384 | 120 MB | ❌ 超過 |

**所以 `n_embd` 最大只能到 320。** 這點在 PDF 裡沒明講，但會直接讓你的 Q3.5 交不出去。

FLOP 公式（`train.py:275`）：
```
FLOPs = flops_per_token × num_training_steps × grad_accumulation_steps × batch_size × seq_len
flops_per_token = 6 × params + 12 × n_layer × n_embd × n_positions
```

五組都固定 `batch_size=64`、`seq_len=256`，只掃模型寬度，最後一組測學習率：

| config | n_embd/head/pos/layer | 參數 | 大小 | steps | FLOPs |
|---|---|---|---|---|---|
| `s1-e128` | 128/4/256/4 | 7.3M | 29MB | 1352 | 9.996e14 |
| `s2-e192` | 192/6/256/6 | 12.4M | 50MB | 785 | 9.999e14 |
| `s3-e256` | 256/8/256/6 | 17.7M | 71MB | 551 | 9.997e14 |
| `s4-e320` | 320/8/256/6 | 23.6M | 94MB | 414 | 9.990e14 |
| `s5-e256lr` | 256/8/256/6（max_lr 3e-3） | 17.7M | 71MB | 551 | 9.997e14 |

全部都貼在預算的 99.9%，不浪費也不超支。這是 Q3.4 想讓你體會的 compute-optimal 取捨：
模型越大每步越貴、步數越少，存在一個最佳點。s1→s4 是「一次只動一個軸」的乾淨掃描，
適合寫進 3.4.A 的實驗流程；s5 則是在最佳寬度上再調學習率。

## ⚠️ 未解決:PDF 與助教 note 對 model hyperparameters 的說法衝突

**狀態:已採用 PDF 的讀法跑 sweep,同時向助教確認中。**

逐字對照:

| 來源 | 說法 |
|---|---|
| PDF Q3.4 | "you are free to modify any of the following hyperparameters... **model hyperparameters such as n_embd, n_head, n_positions and n_layer**; and training hyperparameters such as batch size, seq len, grad accumulation steps, and num training steps" |
| `note.md` 第三段 | "**Please do not change any model hyperparameters in the assignment.** You should only adjust the training parameters in configs/GPT-tiny.yaml" |

### 為什麼採用 PDF 的讀法

若完全凍結 GPT-tiny 的 `model_config`（n_embd=32 / n_head=2 / n_positions=128 / n_layer=2）:

| | 數值 |
|---|---|
| 總參數 | 1.64M |
| 其中 token embedding | 1.61M（**98%**） |
| **真正的 transformer 參數** | **0.025M（25K）** |
| 1e17 FLOPs 下可訓練 | 10.1B tokens = 19.4 個 epoch |

瓶頸不是算力而是架構:輸出層 `h @ E.T` 的 `h` 只有 32 維,logits 的 rank 上限即 32,
卻要區分 50257 個 token —— 這是硬性的 softmax bottleneck。
實測 GPT-tiny 跑 0.016 epoch 得 PPL 1146;Q3.5 的計分門檻是 PPL < 50 滿分、< 75 半分。
**字面讀法會讓這道 20 分題在資訊論上無法得分,故判斷非原意**,
note 第三段應是指「Q3.5 不要改動 Q3.4 定下的架構」。

### 寫信問助教時可以用的要點

1. PDF Q3.4 明文列出 `n_embd, n_head, n_positions, n_layer` 為可調項,與 note 第三段衝突,請問哪個優先?
2. 若 model hyperparameters 完全不可改,`n_embd=32` 只有 25K 個非 embedding 參數,
   即使用滿 1e17 FLOPs（19 個 epoch）也無法達到 Q3.5 要求的 PPL < 50,請問是否理解有誤?
3. 確認 Q3.5 的「do not change your model hyperparameters」是指「沿用 Q3.4 的最佳架構」。

### 如果助教回覆是「嚴格不可改」

那 Q3.4 的 sweep 要改成只掃 `batch_size` / `seq_len` / `grad_accumulation_steps` /
`num_training_steps` / `min_lr` / `max_lr`,`model_config` 五組全部維持 GPT-tiny 原值。
注意 `train.py:262` 有 `assert config.seq_len <= config.model_config.n_positions`,
所以 `n_positions` 凍結在 128 時 `seq_len` 也不能超過 128。

## 6. 跑完之後

1. 看哪組 val PPL 最低 → 那組的 YAML 貼進報告的 3.4.C
2. 3.4.A 寫實驗流程（固定 FLOP 預算、固定 batch/seq、掃寬度、再調 lr）
3. 3.4.B 寫結果與解釋
4. **Q3.5**：沿用最佳組的**模型**超參（`model_config` 整塊不准動），
   只把 `num_training_steps` 乘 100，預算放寬到 1e17

以 `s3-e256` 為例，Q3.5 就是 `num_training_steps: 55115`
（= 903M tokens，約 1.7 個 epoch），其他 model_config 完全不動。
這會跑比較久，估計 1-3 小時（視分到的 GPU），所以前面才要問 `courses-gpu` 的權限。

## 7. 給 Q3.5 的提醒

Q3.5 要求 val PPL < 50 才拿滿分（< 75 拿半分）。從 1e15 的結果外推時注意：
太小的模型（如 s1-e128）在 1e15 可能贏，但到了 1e17 會因為容量不足而先飽和。
如果 s1 和 s3 在 1e15 的 PPL 很接近，**選大的那個**，它在 100 倍預算下的上限更高。
