# rapid-mlx 個別測試報告

## 啟動指令

```bash
rapid-mlx serve mlx-community/Qwen3.6-35B-A3B-4bit \
  --port 8765 --disable-prefix-cache
```

> **重要**：`--disable-prefix-cache` 是必要的，否則 prefix cache 會嚴重污染重複 prompt 的 prefill 數據（部分 run 達到 100K+ tps 是 cache hit，非真實計算）。

---

## 完整統計（n=5）

### Decode tps

| size | median | mean | stddev | min | max |
|---:|---:|---:|---:|---:|---:|
| 64 | 124.9 | 124.7 | 0.6 | 123.8 | 125.3 |
| 512 | 119.5 | 120.6 | 2.6 | 118.0 | 124.4 |
| 2,048 | 102.5 | 105.3 | 5.2 | 100.7 | 112.9 |
| 4,096 | 97.6 | 97.7 | 0.5 | 97.2 | 98.4 |
| 8,192 | 90.3 | 90.1 | 1.6 | 87.8 | 91.7 |
| 16,384 | 83.2 | 82.9 | 0.7 | 82.0 | 83.7 |
| 32,768 | 72.3 | 72.3 | 0.2 | 72.1 | 72.7 |

### Prefill tps

| size | median | mean | stddev | min | max |
|---:|---:|---:|---:|---:|---:|
| 64 | 456 | 359 | 150 | 167 | 473 |
| 512 | 1,026 | 1,227 | 373 | 899 | 1,634 |
| 2,048 | 3,221 | 2,819 | 596 | 2,130 | 3,279 |
| 4,096 | 3,070 | 2,943 | 250 | 2,644 | 3,207 |
| 8,192 | 2,696 | 2,705 | 94 | 2,572 | 2,814 |
| 16,384 | 2,323 | 2,285 | 93 | 2,120 | 2,343 |
| 32,768 | 1,987 | 1,982 | 21 | 1,957 | 2,005 |

### TTFT (ms)

| size | median | mean | stddev | min | max |
|---:|---:|---:|---:|---:|---:|
| 64 | 169 | 259 | 136 | 163 | 462 |
| 512 | 426 | 382 | 107 | 268 | 486 |
| 2,048 | 521 | 619 | 142 | 511 | 787 |
| 4,096 | 1,086 | 1,139 | 99 | 1,039 | 1,260 |
| 8,192 | 2,461 | 2,455 | 86 | 2,359 | 2,580 |
| 16,384 | 5,703 | 5,807 | 250 | 5,656 | 6,251 |
| 32,768 | 13,326 | 13,361 | 140 | 13,207 | 13,531 |

---

## 觀察與分析

### 強項
- **短 context decode 快**：64–512 區間中位數 119–125 tps，接近領先群
- **長 context 變異極低**：4K 起所有 stddev 都 < 1.6，32K 時 stddev 0.2 tps（最穩）
- **Prefill 在 2K–4K 達峰值**（3000+ tps）

### 弱項
- **TTFT 在小 context 抖動明顯**：64 token 時 stddev 達 136ms（中位數 169ms 的 80%），偶有單發 462ms 的 outlier
- **Prefill 在 64–512 短 prompt 時偏低**（456 / 1026 tps），可能是 chunk-prefill 對小 prompt 不友善
- **長 context decode 衰退較快**：32K 時 72.3 tps，比 omlx（82.1）慢約 12%

### 衰退率
從 64 → 32K：**124.9 → 72.3 tps（-42%）**

---

## 適用情境

✅ 短到中 context 的 chat、coding agent
✅ 對 decode 穩定性有要求（長 context stddev 極低）
✅ 不在意小 context TTFT 偶有抖動

❌ 32K+ 超長 context（omlx 較佳）
❌ 對 TTFT 變異敏感的 SLA 場景

---

## 注意事項

1. **必須加 `--disable-prefix-cache`**（否則 prefill 數據不可信）
2. `/no_think` 不會生效——模型仍會輸出 reasoning_content（thinking）；max_tokens 也不會限制 thinking 區段，可能產出超出預期的 token 數（本測試最高 2155 tokens）
3. Health endpoint：`GET /health` 回傳 `{"ready":true}` 後再開始打 request
