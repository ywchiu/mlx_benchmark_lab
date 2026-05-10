# 跨框架交叉比較與總結

## 一、各指標冠軍速覽

| 指標 | 冠軍 | 次強 | 備註 |
|---|---|---|---|
| 短 context decode（64–512） | **dflash-mlx**（167 tps） | rapid-mlx（125） | speculative 命中率高時最快 |
| 中 context decode（2K–4K） | **omlx**（120 tps） | dflash @2K（160） | omlx 最穩，dflash 看內容 |
| 長 context decode（8K–32K） | **omlx**（82 tps @32K） | rapid-mlx（72） | omlx 衰退最緩 |
| Prefill 峰值 | **omlx**（3989 tps @4K） | mlx-vlm（3818 @8K） | dflash 無法測量 |
| 短 TTFT 中位數 | **mlx-vlm / omlx**（130–250ms） | rapid-mlx | dflash 永遠慢 200ms+ |
| TTFT 變異最小 | **omlx**（512: stddev 3ms） | mlx-vlm | rapid 在 64 token 抖 136ms |
| Decode 變異最小 | **mlx-vlm**（短 ctx 0.2 tps） | omlx | 全範圍都穩 |
| Decode 衰退率 | **mlx-vlm**（-29%） | omlx（-34%） | dflash 災難 -92% |
| 32K context | **omlx**（82 tps） | rapid-mlx（72） | dflash 不可用（12 tps） |
| 多模態支援 | **mlx-vlm**（唯一） | — | — |

---

## 二、關鍵交叉比較

### 2.1 「快」與「穩」的取捨

dflash 用 167 tps 拿下短 context 王座，但代價是：
- TTFT 永遠比競爭者慢 200–300ms（draft 模型啟動成本）
- 32K 時崩盤
- 即使在中段 context（512、4K）也未必加速

omlx 全程 117–124 tps，看似平凡，但：
- 任何 context 都不會比第二名差太多
- 變異最低（穩定可預測）
- **長 context 突然變王者**（其他都掉得比它快）

### 2.2 為什麼 rapid-mlx 在 prefill 表現比預期差？

第一次測試（含 prefix cache）rapid-mlx prefill 高達 5086 tps；本次測試（`--disable-prefix-cache`）僅 1987 tps。
差距是因為連續用同 prompt，prefix cache 命中讓多數 run 跳過實際計算。

→ **若你的場景重複 prompt 多**（system prompt 固定、user 變化少），rapid-mlx 啟用 prefix cache 後實際吞吐會遠超本表數據。

### 2.3 dflash 32K 崩盤的數字解析

| 階段 | 64 tokens | 32K tokens | 變化 |
|---|---:|---:|---:|
| TTFT (ms) | 334 | 31,205 | **93×** |
| Decode tps | 167.3 | 12.6 | **0.075×** |
| 1 個 token 的攤銷成本 | 6 ms | 79 ms | 13× 變慢 |

主要瓶頸：draft 模型 prefill 也要 32K，加上 verify 階段的多次主模型 forward 變得不再值得。

### 2.4 TTFT 對 RAG 應用的意義

長 context（16K+）TTFT 已經 4–13 秒，使用者必須等很久才看到第一個字。
- omlx：32K → 12.7s（最快）
- mlx-vlm：32K → 12.8s
- rapid-mlx：32K → 13.3s
- dflash：32K → 31.2s（**不可接受**）

對 RAG 應用，建議搭配 streaming + 思考過程提示用戶系統正在處理，避免 12 秒空白。

---

## 三、決策樹（如何挑框架）

```
你的 context 多長？
├── ≤ 512 tokens
│   ├── 重視最快 decode → dflash-mlx
│   └── 重視 TTFT → omlx 或 mlx-vlm
├── 512 – 4,096 tokens
│   ├── 內容可預測（程式碼/JSON） → dflash-mlx (@2K 有 160 tps 大幅加速)
│   └── 內容多元 → omlx
├── 4,096 – 16,384 tokens
│   └── 一律 omlx（領先 12–15%）
└── > 16,384 tokens
    └── omlx 唯一選擇（dflash 崩盤、其他都慢 10%+）

你需要視覺輸入嗎？
└── 是 → mlx-vlm（唯一）

你需要同機跑多個模型？
└── 是 → omlx（內建 LRU 多模型管理）

你的 prompt 重複度高（system prompt 固定）？
└── 啟用 rapid-mlx prefix cache，prefill 可達 5000+ tps
```

---

## 四、實驗的方法學發現

### 4.1 Prefix cache 是雙面刃

第一次測試出現 prefill tps 高達 100K+（35B 模型物理上不可能達到），原因是 **bench 的 warm-up 與重複 run 用同一 prompt**，啟用 prefix cache 後第二個 run 起完全略過 prefill。

→ 正確做法：
1. 測試框架若有 prefix cache 旗標，**啟動時就關掉**（rapid-mlx `--disable-prefix-cache`、omlx `--no-cache`）
2. 或在每個 run 的 prompt 加唯一前綴打亂 cache
3. **不要只清 cache（`/v1/cache/clear`）**——本測試發現多次清除後仍有殘留

### 4.2 變異測試的價值

短 context（≤512 tokens）所有框架的 decode stddev 都 < 3 tps。**單次測試就足夠**。
但長 context（16K+）開始出現明顯抖動，例如：
- omlx @16K：stddev 2.7 tps
- dflash @32K：TTFT stddev 2,663 ms

→ 長 context 場景**務必至少跑 3–5 次取中位數**，否則單次數據可能誤導 ±20%。

### 4.3 `/no_think` 在 Qwen3 上的可靠性

僅 mlx-vlm 與 dflash-mlx **完全不輸出 thinking tokens**（output 整齊維持 256）。
rapid-mlx 與 omlx 仍會大量輸出 reasoning_content（rapid 甚至完全不受 max_tokens 限制，可達 2155 tokens）。

→ 若評測需要嚴格控制 thinking 行為，建議改用 chat template 的 `enable_thinking=false` 參數，或在 client 端過濾 reasoning_content。

---

## 五、未測試但值得做的後續

1. **mlx-vlm + dflash 組合**——vlm 內建支援，可能同時拿 prefill 強項 + decode 加速
2. **batched / continuous batching** 下的吞吐量比較——rapid 與 omlx 都支援
3. **KV cache 量化**對 32K decode 的影響（rapid 與 mlx-vlm 都支援 4-bit/8-bit KV）
4. **64K / 128K context** 的記憶體與速度極限
5. **Dense 模型**（如 Qwen3-32B-Dense）下的相對排名
6. **omlx prefix cache 實際命中**情況下的吞吐量（生產實境會受惠）
