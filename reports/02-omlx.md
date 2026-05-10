# omlx 個別測試報告

## 啟動指令

```bash
# 模型必須先放在 ~/.omlx/models/<model_id>/，可用 symlink 連結 HuggingFace cache
ln -snf \
  ~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/<HASH>/ \
  ~/.omlx/models/Qwen3.6-35B-A3B-4bit

omlx serve --port 8765 --log-level warning --no-cache
```

> **API 注意**：omlx 用目錄名為 model id，不是 HuggingFace repo 路徑。
> Request 時 `model` 欄位填 `Qwen3.6-35B-A3B-4bit`（去掉 `mlx-community/` 前綴）。

---

## 完整統計（n=5）

### Decode tps

| size | median | mean | stddev | min | max |
|---:|---:|---:|---:|---:|---:|
| 64 | 123.7 | 123.8 | 0.4 | 123.5 | 124.5 |
| 512 | 119.4 | 120.3 | 1.9 | 118.4 | 122.9 |
| 2,048 | 121.1 | 121.0 | 0.4 | 120.4 | 121.2 |
| 4,096 | 120.4 | 119.5 | 2.3 | 115.5 | 120.9 |
| 8,192 | 118.0 | 117.9 | 0.3 | 117.4 | 118.3 |
| 16,384 | 105.3 | 105.0 | 2.7 | 101.3 | 107.9 |
| 32,768 | 82.1 | 82.6 | 1.5 | 81.0 | 84.7 |

### Prefill tps

| size | median | mean | stddev | min | max |
|---:|---:|---:|---:|---:|---:|
| 64 | 520 | 536 | 39 | 510 | 604 |
| 512 | 1,735 | 1,741 | 23 | 1,720 | 1,774 |
| 2,048 | 3,569 | 3,563 | 33 | 3,530 | 3,608 |
| 4,096 | 3,989 | 3,966 | 116 | 3,843 | 4,097 |
| 8,192 | 3,467 | 3,406 | 259 | 3,122 | 3,707 |
| 16,384 | 2,826 | 2,842 | 84 | 2,742 | 2,955 |
| 32,768 | 2,083 | 2,084 | 68 | 2,004 | 2,186 |

### TTFT (ms)

| size | median | mean | stddev | min | max |
|---:|---:|---:|---:|---:|---:|
| 64 | 148 | 144 | 10 | 128 | 151 |
| 512 | 252 | 251 | 3 | 246 | 254 |
| 2,048 | 470 | 471 | 4 | 465 | 475 |
| 4,096 | 836 | 841 | 25 | 813 | 867 |
| 8,192 | 1,914 | 1,957 | 150 | 1,790 | 2,125 |
| 16,384 | 4,689 | 4,666 | 138 | 4,483 | 4,833 |
| 32,768 | 12,709 | 12,713 | 412 | 12,113 | 13,209 |

---

## 觀察與分析

### 強項
- **長 context decode 王者**：4K 起全段領先，32K 時仍有 82 tps（比 rapid-mlx 快 14%、比 dflash 快 6.5×）
- **TTFT 最穩定**：512–4K 區間 stddev 僅 3–25ms（其他框架是 100ms+）
- **Decode 衰退最緩**：64 → 32K 衰退 -34%（其他都更陡）
- **Prefill 在 4K 達峰值 3989 tps**（所有框架最高）
- **多模型管理**：支援 LRU 自動卸載，可同機切換多個 model

### 弱項
- **設定門檻高**：模型必須放 `~/.omlx/models/`，不能直接吃 HuggingFace repo
- **短 context decode 略低**：64 區間 123.7 tps，比 rapid-mlx 與 dflash 慢
- **16K/32K TTFT 變異略大**：stddev 138–412ms

### 衰退率
從 64 → 32K：**123.7 → 82.1 tps（-34%）**——4 個框架中最緩

---

## 適用情境

✅ **長 context RAG / 長文摘要 / 程式碼分析**（最強）
✅ 對 TTFT 變異敏感的 SLA 場景
✅ 同機多模型熱切換的 dev 環境
✅ 32K+ 超長 context（其他都不行）

❌ 短 prompt chat（dflash 更快）
❌ 不想處理模型目錄管理

---

## 注意事項

1. `~/.omlx/settings.json` 預設 `max_context_window=32768`；要測 64K+ 需先調此值
2. 預設啟用 SSD cache（`~/.omlx/cache/`），測效能必須加 `--no-cache`
3. 多模型部署時要監控 `max_model_memory`，否則模型會被 LRU 卸載
4. Admin API（`/admin/api/...`）需要 cookie auth；reload 模型只能透過重啟 server 或先登入 admin UI
