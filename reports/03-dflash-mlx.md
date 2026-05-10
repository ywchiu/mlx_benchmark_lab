# dflash-mlx 個別測試報告

## 啟動指令

```bash
dflash-serve \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --draft z-lab/Qwen3.6-35B-A3B-DFlash \
  --port 8765
```

> **重要**：`z-lab/Qwen3.6-35B-A3B-DFlash` 是 **draft 模型**，不能單獨當 target。
> 必須以 `--model <target>` + `--draft <dflash>` 配對啟動，否則會因為缺少 91 個權重 (`fc.weight`、`hidden_norm.weight`、layers.0..7) 而崩潰。

---

## 完整統計（n=5）

### Decode tps

| size | median | mean | stddev | min | max |
|---:|---:|---:|---:|---:|---:|
| 64 | **167.3** | 167.3 | 2.9 | 163.1 | 170.9 |
| 512 | 122.9 | 122.1 | 2.2 | 118.3 | 123.5 |
| 2,048 | **160.1** | 158.7 | 2.8 | 153.8 | 160.5 |
| 4,096 | 104.5 | 104.4 | 0.8 | 103.5 | 105.6 |
| 8,192 | 96.3 | 95.9 | 1.6 | 93.6 | 97.9 |
| 16,384 | 84.1 | 84.4 | 2.4 | 81.7 | 88.0 |
| 32,768 | **12.6** ⚠️ | 12.8 | 1.3 | 11.3 | 14.5 |

### Prefill tps

dflash-serve 在 streaming 的 `usage` 欄位**沒有回傳 `prompt_tokens`**（永遠為 0），因此本表無法填入。
若需要估算，可從 TTFT 反推（已知實際輸入 tokens 為 77/437/1677/3333/6636/13250/26475）。

### TTFT (ms)

| size | median | mean | stddev | min | max |
|---:|---:|---:|---:|---:|---:|
| 64 | 334 | 340 | 17 | 328 | 371 |
| 512 | 360 | 364 | 8 | 357 | 378 |
| 2,048 | 587 | 572 | 36 | 507 | 591 |
| 4,096 | 1,072 | 1,063 | 27 | 1,019 | 1,090 |
| 8,192 | 2,205 | 2,217 | 33 | 2,178 | 2,261 |
| 16,384 | 6,023 | 5,957 | 185 | 5,651 | 6,119 |
| 32,768 | **31,205** ⚠️ | 31,796 | 2,663 | 28,469 | 35,830 |

---

## 觀察與分析

### 強項：speculative decoding 在小至中 context 加速顯著
- **64 tokens：167 tps**（領先第二名 33%）
- **2,048 tokens：160 tps**（領先第二名 30%）
- 命中率高時 decode 速度遠超其他框架
- 執行時非常穩定（stddev 0.8–2.9 tps）

### 致命弱項：32K 災難性崩盤
- **decode 12.6 tps**——比 rapid-mlx/omlx/mlx-vlm 慢 6× 以上
- **TTFT 31 秒**——其他框架約 13 秒，dflash 是 2.4×
- TTFT stddev 達 2,663ms（最大 35.8s 最小 28.5s，極不穩定）

### 為什麼長 context 會崩盤？
1. **draft 模型也要處理完整 context**——35B 主模型 prefill 已經慢，draft 又再做一次，等於雙重開銷
2. **長 context 下 draft 命中率下降**——預測錯就 verify 失敗，回滾代價高
3. **每生成 1 token 要：draft N tokens → main 平行 verify → 接受/拒絕回退**——驗證次數隨 context 變長而變慢
4. 32K 時 verify 階段成本 >> 純 decode 成本，speculative 變成負效益

### 中間 context（512、4K）為什麼也沒加速？
- 512 token：decode 122.9 tps，與 baseline 約略相同（無加速）
- 4K token：decode 104.5 tps，**比 omlx（120.4）慢**
- 推測原因：draft 預測率隨內容語料不同而異，本測試 prompt 為自然語言文本，512 區段恰好命中率低

### 衰退率
從 64 → 32K：**167.3 → 12.6 tps（-92%）**——4 個框架最差，且差距懸殊

---

## 適用情境

✅ 短至中 context（< 4K）的快速生成
✅ 確定性高、可預測的內容（程式碼、JSON、結構化輸出）
✅ 對 TTFT 不敏感的 batch 任務

❌ **絕對不要用在 16K+ context**
❌ TTFT 敏感的互動應用（永遠比其他框架慢 200–300ms）
❌ 自然語言 / 多元內容的生成（命中率低時無加速）

---

## 注意事項

1. **不能單獨用 DFlash 模型**——必須配對基礎 4-bit 模型
2. **無 `/health` endpoint**——readiness 改用 `GET /v1/models`
3. **`usage` 不回 `prompt_tokens`**——若工具鏈需要 token 計數要另外算
4. **32K context 是嚴格上限**——超過會更糟
5. CLI 還有 `dflash`（單次生成）和 `dflash-benchmark`（內建 baseline vs DFlash 比較）兩個工具
