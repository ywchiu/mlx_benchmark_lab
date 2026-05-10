# MLX 推論框架效能比較實驗室

針對 Apple Silicon 上的四個 MLX 推論框架（**rapid-mlx**、**omlx**、**dflash-mlx**、**mlx-vlm**），在 4-bit MoE 模型上進行 7 種 context 長度、5 次重複的完整速度評測。

> **TL;DR**：長 context 用 **omlx**，短 context（<2K）用 **dflash-mlx**，多模態場景用 **mlx-vlm**，dflash 在 32K 會災難性崩盤（decode 跌至 12.6 tps，慢 6×）。

---

## 一、測試環境

| 項目 | 規格 |
|---|---|
| 硬體 | Apple Silicon (Mac, 64 GB 統一記憶體) |
| 模型 | `mlx-community/Qwen3.6-35B-A3B-4bit`（4-bit 量化、MoE，總參數 35B / 啟用 3B） |
| Draft 模型（dflash 專用） | `z-lab/Qwen3.6-35B-A3B-DFlash` |
| 受測框架 | rapid-mlx、omlx、dflash-mlx、mlx-vlm |
| 通訊協定 | OpenAI 相容 `/v1/chat/completions`（streaming） |
| 測試日期 | 2026-05-09 |

---

## 二、測試方法

| 設定 | 值 |
|---|---|
| Prompt 長度 | **64 / 512 / 2048 / 4096 / 8192 / 16384 / 32768** tokens |
| `max_tokens` | 256 |
| 每情境執行次數 | **5 次**（取中位數、平均、stddev） |
| Warm-up | 每情境跑一次同尺寸 prompt（捨棄不計） |
| Thinking 控制 | prompt 開頭加上 `/no_think` |
| Prefix cache | rapid-mlx 啟動時加 `--disable-prefix-cache`；omlx 啟動時加 `--no-cache` |
| 並發 | 單一 request（不測 batching） |
| 取樣指標 | TTFT、Prefill tps、Decode tps |

每個框架皆獨立啟動於 `port 8765`，測完即關閉，避免記憶體互相干擾。

腳本：[`scripts/bench_inline.py`](scripts/bench_inline.py)　
產圖：[`scripts/plot_results.py`](scripts/plot_results.py)　
原始日誌：[`logs/`](logs/)　
原始 JSONL：[`data/`](data/)

---

## 三、視覺化結果

### 3.1 Decode 速度隨 context 變化

![Decode tps](charts/decode_tps.png)

dflash 在 64 與 2048 tokens 出現明顯尖峰，但在 4K 之後優勢消失，**32K 時崩盤至 12.6 tps**。omlx 在 4K 之後的長 context 表現最穩。

### 3.2 Decode 速度衰退曲線（相對 64-token 基準）

![Degradation](charts/degradation.png)

- **omlx 衰退最緩（-34%）**
- mlx-vlm 因為 64 token 時起點低，相對曲線最平
- dflash 從 167 tps 跌到 12.6 tps，**衰退 -92%**

### 3.3 Decode 速度穩定性（stddev）

![Decode stddev](charts/decode_stddev.png)

短 context 所有框架都很穩；長 context 開始出現抖動。dflash 在 64 tokens 雖快但變異也最大。

### 3.4 Decode 變異盒鬚圖（每 context 5 runs）

![Decode boxplot](charts/decode_box.png)

### 3.5 Prefill 速度

![Prefill tps](charts/prefill_tps.png)

dflash 因 `usage` 不回傳 `prompt_tokens`，無法量測。三個能測的框架都在 4K–8K 達峰值。

### 3.6 TTFT（首 token 延遲，對數刻度）

![TTFT](charts/ttft.png)

dflash 在 32K 的 TTFT 31 秒，是其他三個框架的 2.4×。

---

## 四、Decode tps 中位數總覽

| Prompt size | rapid-mlx | omlx | dflash-mlx | mlx-vlm |
|---:|---:|---:|---:|---:|
| 64 | 124.9 | 123.7 | **167.3** | 95.5 |
| 512 | 119.5 | 119.4 | **122.9** | 94.8 |
| 2,048 | 102.5 | 121.1 | **160.1** | 88.5 |
| 4,096 | 97.6 | **120.4** | 104.5 | 91.4 |
| 8,192 | 90.3 | **118.0** | 96.3 | 87.2 |
| 16,384 | 83.2 | **105.3** | 84.1 | 83.1 |
| 32,768 | 72.3 | **82.1** | 12.6 ⚠️ | 67.7 |

完整統計（mean、stddev、min、max）見 [reports/](reports/) 內各框架報告。

---

## 五、選擇建議

| 使用情境 | 首選框架 | 原因 |
|---|---|---|
| 短 prompt 互動 chat（< 2K） | **dflash-mlx** | 167 tps 最快；可接受 300ms 額外 TTFT |
| TTFT 優先的互動式應用 | **omlx** | TTFT 中位數最低、變異最小 |
| 長文摘要、RAG（4K–32K） | **omlx** | 長 context decode 最快、衰退最緩 |
| 程式碼生成（中 context、確定性高） | **dflash-mlx** | 2K 區間 160 tps，speculative 命中率高 |
| 32K+ 超長 context | **omlx** | dflash 崩盤、其他都比 omlx 慢 |
| 生產 SLA、要求低變異 | **omlx 或 mlx-vlm** | TTFT/decode stddev 都最低 |
| 圖文/影音多模態 | **mlx-vlm** | 唯一支援視覺輸入 |
| 同機多模型熱切換 | **omlx** | 多模型管理 + LRU 卸載 |

---

## 六、結論

1. **omlx 是綜合最強的長 context 框架**——4K–32K 全段 decode 最快，TTFT 最穩定，是目前最值得部署的生產級選項。
2. **dflash-mlx 是「短 context 神器，長 context 災難」**——≤2K 時最快，但 32K 比所有框架慢 6×；使用前必須嚴格限定 context 上限。
3. **rapid-mlx 是穩健的中間派**——短 context decode 強，但 TTFT 抖動大；長 context 表現遜於 omlx。
4. **mlx-vlm 純文字效能墊底，但變異最低**——除非需要多模態，否則純文字場景不建議使用。
5. **變異測試的關鍵發現**：所有框架在小 context 時 stddev 都很低（<3 tps），但長 context（16K+）開始出現明顯抖動，此時建議實機測試而非依賴單次數據。

---

## 七、目錄結構

```
mlx_benchmark_lab/
├── README.md                      # 本檔（總覽 + 結論）
├── data/                          # 原始 JSONL 數據（每行一次 run）
│   ├── rapid_v5.jsonl
│   ├── omlx_v5.jsonl
│   ├── dflash_v5.jsonl
│   └── vlm_v5.jsonl
├── logs/                          # 完整測試 log
│   ├── rapid_v5.log
│   ├── omlx_v5.log
│   ├── dflash_v5.log
│   └── vlm_v5.log
├── scripts/
│   ├── bench_inline.py            # 主測試腳本
│   └── plot_results.py            # 圖表生成腳本
├── reports/                       # 個別框架深入分析
│   ├── 01-rapid-mlx.md
│   ├── 02-omlx.md
│   ├── 03-dflash-mlx.md
│   ├── 04-mlx-vlm.md
│   └── 99-summary.md              # 總結（含交叉比較）
└── charts/                        # 視覺化圖表（PNG）
    ├── decode_tps.png
    ├── prefill_tps.png
    ├── ttft.png
    ├── decode_box.png
    ├── decode_stddev.png
    └── degradation.png
```

---

## 八、如何重現

```bash
# 1. 安裝對應框架（任一即可，視要測哪個）
pip install rapid-mlx omlx dflash-mlx mlx-vlm

# 2. 下載模型（HuggingFace cache 或 omlx 模型目錄）
huggingface-cli download mlx-community/Qwen3.6-35B-A3B-4bit
huggingface-cli download z-lab/Qwen3.6-35B-A3B-DFlash

# 3. 啟動 server（範例：rapid-mlx）
rapid-mlx serve mlx-community/Qwen3.6-35B-A3B-4bit \
  --port 8765 --disable-prefix-cache &

# 4. 跑測試
python3 scripts/bench_inline.py \
  --url http://localhost:8765 \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --sizes 64,512,2048,4096,8192,16384,32768 \
  --runs 5 \
  --max-tokens 256 \
  --json-out data/rapid_v5.jsonl > logs/rapid_v5.log

# 5. 產生圖表
python3 scripts/plot_results.py
```

---

## 九、限制與後續方向

- 單一 sequence，未測 batching；rapid-mlx 與 omlx 都支援 continuous batching，多人併發排名可能改變
- 4-bit MoE 受記憶體頻寬限制較重；換成 dense 模型結果可能不同
- 未測試 KV cache 量化對長 context 效能的影響
- 未測試 mlx-vlm 啟用 `--draft-kind dflash` 後是否能結合兩者優勢
- `/no_think` 在所有框架皆未完全生效，decode tps 包含 reasoning_content tokens
- dflash-mlx 不回傳 `prompt_tokens`，prefill tps 數據缺失
- 32K 之後（如 64K、128K）未測試
