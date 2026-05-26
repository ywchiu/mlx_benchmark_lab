# MLX 推論框架效能比較實驗室

[English Version](README.md)

## 這份專案在做什麼

這個 repo 收錄了一份在 Apple Silicon 上對四個 MLX 推論框架做正面對決的速度測試所有原始資料、腳本與分析。受測的四個框架是 **rapid-mlx**、**omlx**、**dflash-mlx** 與 **mlx-vlm**，全部跑同一個模型（`mlx-community/Qwen3.6-35B-A3B-4bit`，這是一個 4-bit 量化的 MoE 模型，總參數 35B、每個 token 啟用 3B），覆蓋從 64 token 到 32,768 token 共七種 prompt 脈絡長度，每一格跑五次取中位數，這樣才能同時看出各框架的代表速度與穩定程度。

之後我們又加測了第五個框架 **mtplx**，但跑的是另一個模型（`Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed`，27B dense、專為 MTPLX 的 speculative decoding 調校過）。因為模型不同，mtplx 在所有圖表中以虛線呈現、表格內以 `*` 標註，它的數字本身有獨立意義，但不能直接拿來跟另外四個比較。詳見 [四之一、mtplx（不同模型）](#41-mtplx不同模型)的獨立結果與測試方法。

結論的精簡版：**dflash-mlx** 在每個測試脈絡長度的 decode 速度都領先——從 4K 到 32K 都是冠軍（128、125、119、121 tps 中位數），短脈絡也是最快（512 到 2K 落在 139–155 tps）。**omlx** 是「無聊但穩定」的選擇，當穩定度比峰值更重要時請選它——它的 decode tps 衰退曲線最平緩、TTFT 變異最低，最適合要訂 SLA 的場景。**rapid-mlx** 是穩當的中堅，從不領先但也從不落後太多。**mlx-vlm** 是這次唯一支援圖像、影片與音訊輸入的框架，但純文字的速度比另外三個慢約 25–30%，所以只在你真的需要多模態時才用它。**mtplx** 跑它自己那個速度優化過的 27B 模型，decode 整體落在 31–60 tps 之間，記憶體則非常穩（512 token 約 15.6 GB，32K 時也才 22.1 GB），所以它的賣點是「記憶體可預期」而不是「絕對最快」。

> **關於 dflash-mlx 數字的說明。** dflash-mlx 那一列是 2026-05-25 用 dflash-mlx **v0.1.7** 測的，run 之間 `cooldown=60`（給 M5 Max 足夠散熱時間，量到的是峰值表現）。其他四個框架是 2026-05-09 用 lab 預設的 `cooldown=2` 測的；如果用較長的 cooldown 重測，它們的長脈絡數字也會比較高。dflash 完整方法論請見 [`reports/03-dflash-mlx_zh.md`](reports/03-dflash-mlx_zh.md)。

---

## 一、測試環境

硬體是 Apple M5 Max，配 64 GB 統一記憶體。前四個主要框架都載入同一個 target 模型 `mlx-community/Qwen3.6-35B-A3B-4bit`；只有 dflash-mlx 額外載入搭配的 draft 模型 `z-lab/Qwen3.6-35B-A3B-DFlash`，用來驅動 speculative decoding。所有 server 都暴露 OpenAI 相容的 streaming 介面 `/v1/chat/completions`，測試 client 也透過這個介面說話，所以雖然各框架內部實作不同，但在 API 層面這是真正公平的對比。這幾組測試的日期是 2026-05-09。

後來加測的 **mtplx** 於 2026-05-16 進行，使用的模型是 `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed`，跑的是 MTPLX 內建的 `prefill-ladder` 測試工具而不是 HTTP client。設定為：`sustained` 模式、MTP（`depth=3`）、`disable-thinking`、每個 context 生成 128 tokens。每個脈絡長度只跑一次，並記錄了每一格的尖峰記憶體使用量。

---

## 二、測試方法

我們測了七種 prompt 脈絡長度——64、512、2,048、4,096、8,192、16,384、32,768 個 token——做法是產生對應長度的填充文字，請模型把它摘要起來。每一格重複跑五次，記錄中位數、平均值與標準差，這樣才能區分「這個框架真的比較快」和「這個框架剛好這一次比較快」。在計時開始之前，每個情境會先跑一次同尺寸但捨棄不計的 warm-up，確保 Metal kernel 與權重已經載到 GPU 上，避免冷啟動成本污染資料。

有一個容易忽略但非常關鍵的細節：我們在每個有 prefix cache 機制的框架啟動時都明確關閉它（rapid-mlx 用 `--disable-prefix-cache`、omlx 用 `--no-cache`）。第一次測試時 prefill 的數字異常離譜——35B 模型不可能達到的每秒十萬以上 token——後來才發現是 warm-up 把 prefix cache 灌滿，後續的計時 run 直接用 cache 裡的 KV 跳過實際計算。把快取關掉之後，每次 run 才是誠實的冷啟動 prefill 表現。

prompt 開頭加了 `/no_think`（Qwen3 用來抑制 reasoning 輸出的慣例），`max_tokens` 設為 256。每個框架都獨立啟在 8765 port，測完關閉再換下一個，避免記憶體相互干擾。我們沒有測 batching 或 concurrent 請求——這次的數字只代表單一 sequence 的延遲。

測試 client 在 [`scripts/bench_inline.py`](scripts/bench_inline.py)；繪圖腳本在 [`scripts/plot_results.py`](scripts/plot_results.py)。完整 log 在 [`logs/`](logs/)、每次 run 的原始資料在 [`data/`](data/) 以 JSONL 格式存放。

---

## 三、視覺化結果

### 3.1 Decode 速度隨脈絡長度變化

![Decode tps](charts/zh/decode_tps.png)

這張是頭條圖。x 軸是 prompt 脈絡長度，y 軸是 decode 速度的中位數。dflash-mlx（紅線）在整個區間都領先——它在 2K 的尖峰來自 speculative decoding 在自然語言摘要任務上的高命中率，而那條線一路維持在其他人之上直到 32K。omlx（綠線）是「無聊但有效」的直線——長脈絡第二名、變異最低。rapid-mlx（藍線）在小脈絡下表現很強，但隨著脈絡變長衰退得比 omlx 快。mlx-vlm（橘線）整段都最慢，但也是最平緩的曲線。紫色虛線是 mtplx 跑它自己的 27B 速度優化模型——因為 27B dense 每個 token 要走比 35B-MoE 多得多的參數，整條線明顯偏低，但從 4K 之後曲線異常平坦（4K→32K 是 43→31 tps）。

### 3.2 Decode 衰退曲線

![衰退曲線](charts/zh/degradation.png)

把同一份資料正規化，讓每個框架自己最小脈絡的速度當作 100% 起點（前四個框架是 64 token，mtplx 沒測 64 所以用 512 token）。這樣可以把「KV cache 變大時，decode 慢多少」跟「哪個框架絕對最快」拆開來看。omlx 衰退最少，從 100% 降到 32K 時的 66%；mlx-vlm 相對曲線甚至更平，但這只是因為它的 64 token 起點本來就低。rapid-mlx 到 32K 時掉了大約 42% 的速度。dflash-mlx 到 32K 時大約是 64 token 基準的 82%——是這張圖裡衰退最緩的曲線，因為 speculative decoding 用「每次 main 模型 verify 多接受幾個 draft token」攤分了長脈絡的成本。mtplx 到 32K 時大約是 512 token 基準的 52%——斜率比 omlx 陡，但全程沒有懸崖，跟它平穩成長的記憶體曲線是一致的。

### 3.3 Decode 穩定性（標準差）

![穩定性](charts/zh/decode_stddev.png)

這張顯示同一格五次 run 之間的變異程度。柱越短代表這個框架 run 與 run 之間越可預期。短脈絡下所有框架都很穩（標準差都低於 3 tps），但脈絡一拉長，抖動就跑出來了。這裡要學到的教訓是：到了 16K 以上，單次測量已經不可信了——你在實際環境裡看到的數字，和單次量到的可能差個 ±5 tps。所以當你要根據長脈絡表現挑框架時，請務必自己重複多測幾次，不要相信單一資料點。*（mtplx 沒出現在這張圖和下一張盒鬚圖裡，因為每個 context 只跑了一次，無變異可畫。）*

### 3.4 各脈絡長度下的 decode 分佈

![盒鬚圖](charts/zh/decode_box.png)

每個（框架、脈絡長度）組合的 decode tps 盒鬚圖。盒子是四分位距，鬚是 5 次 run 的最大最小值。這張圖能看到中位數隱藏的細節，例如 rapid-mlx 在 2,048 token 處的盒子明顯比較大，因為各次 run 的 thinking token 輸出長度不一致導致下游時間有波動。

### 3.5 Prefill 速度

![Prefill tps](charts/zh/prefill_tps.png)

Prefill 速度衡量模型消化 prompt 有多快，也就是「打完字到開始吐第一個字之前，模型內部在忙什麼」的速度。35B-MoE 那四個框架都在 4K–16K 區間達到峰值，這是這款硬體上 attention 計算和記憶體頻寬的甜蜜點。mtplx 的 prefill 在絕對值上明顯較低（在 1K 峰值也只有 879 tps），因為 27B dense 模型 prefill 計算量本來就比 MoE 重得多，這也讓它在長 context 時 TTFT 的成本遠大於 decode 成本。

### 3.6 TTFT（首字元延遲，對數刻度）

![TTFT](charts/zh/ttft.png)

TTFT 是使用者實際感受到的延遲，從按下送出到第一個 token 出現之間的等待。y 軸用對數刻度，因為 TTFT 在不同脈絡長度下會跨過將近三個數量級。dflash-mlx 在 32K 處落在 8.1 秒——35B-MoE 框架裡最低。mtplx 在每個 context 都明顯坐在 35B-MoE 四人組之上，到 32K 更是吃到 62 秒——這個配置最大的成本就是弱化的 prefill，所以 mtplx 比較適合短 prompt 而非長 prompt 的應用。

---

## 四、Decode tps 中位數彙整表

| Prompt size | rapid-mlx | omlx | dflash-mlx† | mlx-vlm | mtplx\* |
|---:|---:|---:|---:|---:|---:|
| 64 | 124.9 | 123.7 | **148.7** | 95.5 | — |
| 512 | 119.5 | 119.4 | **138.9** | 94.8 | 59.8 |
| 1,024 | — | — | — | — | 49.6 |
| 2,048 | 102.5 | 121.1 | **154.5** | 88.5 | 55.7 |
| 4,096 | 97.6 | 120.4 | **127.7** | 91.4 | 43.3 |
| 8,192 | 90.3 | 118.0 | **124.6** | 87.2 | 43.1 |
| 16,384 | 83.2 | 105.3 | **119.0** | 83.1 | 41.4 |
| 32,768 | 72.3 | 82.1 | **121.2** | 67.7 | 31.3 |

† dflash-mlx 那一欄：v0.1.7，2026-05-25 測，`cooldown=60`（量峰值表現）。其他四欄是 2026-05-09 那輪在 `cooldown=2` 下的結果；如果用較長的 cooldown 重測，它們的長脈絡數字也會稍高。

\* mtplx 跑的是**不同的模型**（`Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed`，27B dense）也用了不同的方法（n=1、輸出 128 tokens、MTP depth=3、`disable-thinking`），所以這一欄不能直接和其他欄比較。

完整的統計資料——平均值、標準差、最小值、最大值——請見 [`reports/`](reports/) 下的個別框架深入分析。

### 4.1 mtplx（不同模型）

這一節把 mtplx 當作「MTPLX 框架跑它自己調校過的模型會長什麼樣子」的參考點。因為這裡的 runtime 跟模型都跟前四格不同，請當作獨立檔案而不是第五個比較欄看待。

**設定。** 模型：`Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed`。測試工具：MTPLX 內建的 `prefill-ladder`，模式為 `sustained`，啟用 `--mtp --depth 3 --disable-thinking --max-tokens 128`。每個 context 只跑一次。

| Context | Decode tps | Prefill tps | TTFT | 尖峰記憶體 |
|---:|---:|---:|---:|---:|
| 512 | 59.76 | 800.21 | 0.65 秒 | 15.58 GB |
| 1,024 | 49.56 | 879.06 | 1.17 秒 | 16.18 GB |
| 2,048 | 55.69 | 720.81 | 2.84 秒 | 17.29 GB |
| 4,096 | 43.28 | 693.90 | 5.90 秒 | 17.73 GB |
| 8,192 | 43.09 | 664.82 | 12.32 秒 | 18.37 GB |
| 16,384 | 41.40 | 646.68 | 25.35 秒 | 19.62 GB |
| 32,768 | 31.34 | 530.79 | 61.74 秒 | 22.12 GB |

值得注意的是曲線的形狀。短脈絡段（512–2K）decode 大概落在 50–60 tps，從 4K 到 16K 進入 41–43 tps 的平台期，要到 32K 才真的滑下來變成 31 tps。decode 速率明顯低於 35B-MoE 那組——因為 35B-MoE 模型每個 token 只啟用 3B 參數，而這個 27B dense 每個 token 全部 27B 都要走一遍，單 token 的記憶體頻寬成本大很多。換來的好處是記憶體：尖峰 GPU 記憶體從 512 tokens 的 15.6 GB 只長到 32K 的 22.1 GB——一台 36 GB 的 Mac 跑完整條 ladder 還有餘裕。

弱點則在 TTFT。Prefill 上限低於 900 tps，32K 那一格要等超過一分鐘才出第一個 token——所以這個組合適合「短 prompt + 長輸出」的場景，比較不適合 RAG 那種長 prompt 工作負載。

---

## 五、實務選擇建議

如果你的應用是互動式 chat，使用者打一段短訊息然後等回應，**dflash-mlx** 是首選——前提是你能接受 draft model 暖機帶來的首字元延遲成本。它在短脈絡下每秒 139–155 token 的 decode 速度比其他框架快，而那點 TTFT 的代價通常使用者不會察覺，因為整體回應時間還是被生成階段主導。但如果 TTFT 比吞吐量更重要——例如你希望回應「立刻」開始 stream——那就用 **omlx**：它在小脈絡區間既有最低的中位數 TTFT，也有最低的 TTFT 變異。

對於檢索增強生成、長文件摘要、或是任何把脈絡推到 4K–32K 區間的工作負載，**dflash-mlx** 是最強選項——它在這個區段每一格都拿下 decode tps 冠軍（32K 121 tps），32K TTFT 也是最低的（8.1 秒）。如果穩定度比峰值速度重要，**omlx** 仍然是個強選項，它有這次測試裡最低的 TTFT 變異。

如果你做的是程式碼生成這種特定場景，**dflash-mlx** 是當然之選——程式碼高度可預測，speculative decoding 在結構化輸出的命中率比自然語言高得多，自然語言測試在 512–2K 區間已經是 139–155 tps，程式碼工作負載可能會更快。

如果你需要把圖像、聲音、影片送進模型，**mlx-vlm** 是這次測試中唯一的選擇。純文字 decode 慢個 25–30% 是你為了多模態 stack 付出的稅金，但如果你需要視覺能力，這裡沒有別的框架做得到。mlx-vlm 還有一個我們沒測但很有意思的功能：它支援 `--draft-kind dflash`，理論上可以把它的強 prefill 跟 dflash 的 decode 加速結合起來。

**mtplx** 跑那個速度優化過的 27B 模型，比較適合「瓶頸在記憶體而不是 token/秒」或「prompt 短」的情境。即便到 32K 脈絡，尖峰記憶體仍維持在 23 GB 以下，因此 36 GB 統一記憶體的 M 系列 Mac 仍有充足空間；代價是 decode 速度落在 30–60 tps 之間（取決於 context）。這個組合適合「短 prompt + 長輸出」的生成情境（草稿、程式碼補完、結構化輸出）在記憶體受限的機器上跑，反過來不適合長脈絡 RAG，因為那種場景 TTFT 會把總延遲拉滿。

如果是上線到 production、穩定性與可預測性比峰值速度更重要，答案還是 **omlx**。它在 TTFT 和 decode 的標準差都是這次測試最低的，常常比 rapid-mlx 低十倍。如果你要對外承諾 SLA、要保證 p99 延遲，omlx 會讓你的承諾比較有把握。

---

## 六、結論

這次 benchmark 跑出了兩個互補的贏家。**dflash-mlx 是全程最快的**——它的 speculative decoding 架構在每個測試的脈絡長度都拿到最高的 decode tps，包括 32K（121 tps），靠著 draft token 攤分長脈絡成本，衰退曲線是這次 benchmark 裡最緩的。**omlx 是穩定優先框架的代表**——它讓出了大約 10–30% 的峰值 decode 給 dflash，但換來明顯更低的 run-to-run 變異跟這次測試最低的 TTFT 變異。對 p99 比中位數更重要的 production 部署來說，omlx 仍然是最值得信賴的預設選項。

**rapid-mlx 是穩當的中堅**。它在任何尺寸都不是絕對最快，但也從來不會離冠軍太遠。它的主要弱點是小脈絡下的 TTFT 抖動：在 64 token 我們量到的中位數是 169 ms，但標準差高達 136 ms——意思是某些 request 不為什麼就慢了將近一倍。如果你的應用能容忍這種變異，rapid-mlx 仍然是個好選擇，而且它的功能組合最豐富（paged KV cache、MTP、prefix cache、KV 量化都有）。

**mlx-vlm 是多模態特例**。純文字場景每一格都最慢，但它是這份測試裡唯一支援視覺與音訊的框架，所以這個比較其實不太公平：你選 mlx-vlm 不是為了文字速度，而是為了把圖片餵給模型。

**mtplx 是「記憶體受限」的選項**，但因為配的是不同模型，這個比較需要加註腳。它跑自家那個 27B dense 速度優化模型時 decode 不會超過 60 tps，長 context 時 TTFT 還會吃掉異常多時間；但它從 512 到 32K 尖峰記憶體都壓在 23 GB 以下——比任何一個 35B-MoE 框架在同樣 context 下都還少很多。如果你的硬體限制是「長 context 下要塞進 24 GB」，mtplx + 這個模型是少數做得到的組合；如果你的限制是「把吞吐量壓到最大」，那它打不贏 omlx。

這一輪測試的另一個重要發現是關於 benchmark 方法本身。長脈絡下的「單次」速度數字是會誤導人的——變異是真實的，兩個框架在 32K 的差距，可能比同一個框架兩次 run 的差距還小。本次採用的「五次測量加 warm-up」方法每個框架要多花約 10 分鐘，但這多出來的成本就是「這個框架真的比較快」和「這個框架看起來比較快」的差別。

---

## 七、目錄結構

```
mlx_benchmark_lab/
├── README.md                     # 英文（主檔）
├── README_zh.md                  # 本檔（繁體中文）
├── data/                         # 原始 JSONL（每行一次 run）
│   ├── rapid_v5.jsonl
│   ├── omlx_v5.jsonl
│   ├── dflash_v6_c60.jsonl       # dflash-mlx v0.1.7，cooldown=60（目前資料）
│   ├── vlm_v5.jsonl
│   └── mtplx_v5.jsonl            # mtplx + Qwen3.6-27B-MTPLX-Optimized-Speed（n=1）
├── logs/                         # 完整測試 log
├── scripts/
│   ├── bench_inline.py           # streaming 速度測試 client
│   └── plot_results.py           # 圖表生成（--lang en|zh）
├── reports/                      # 個別框架深入分析（中英對照）
│   ├── 01-rapid-mlx.md
│   ├── 01-rapid-mlx_zh.md
│   ├── 02-omlx.md
│   ├── 02-omlx_zh.md
│   ├── 03-dflash-mlx.md
│   ├── 03-dflash-mlx_zh.md
│   ├── 04-mlx-vlm.md
│   ├── 04-mlx-vlm_zh.md
│   ├── 99-summary.md
│   └── 99-summary_zh.md
└── charts/
    ├── *.png                     # 英文標籤圖表
    └── zh/                       # 中文標籤圖表
        └── *.png
```

---

## 八、如何重現

要在你自己的 Mac 上重現這份數字，完整步驟如下。每一步都假設你已經裝好 Python 3.11+ 和對應的框架。

```bash
# 1. 安裝想測的框架
pip install rapid-mlx       # 或 omlx、dflash-mlx、mlx-vlm

# 2. 下載模型（HuggingFace cache）
huggingface-cli download mlx-community/Qwen3.6-35B-A3B-4bit
huggingface-cli download z-lab/Qwen3.6-35B-A3B-DFlash   # 只有 dflash 需要

# 3. 啟動 server（rapid-mlx 為例）
rapid-mlx serve mlx-community/Qwen3.6-35B-A3B-4bit \
  --port 8765 --disable-prefix-cache &

# 4. 跑 benchmark
python3 scripts/bench_inline.py \
  --url http://localhost:8765 \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --sizes 64,512,2048,4096,8192,16384,32768 \
  --runs 5 \
  --max-tokens 256 \
  --json-out data/rapid_v5.jsonl > logs/rapid_v5.log

# 5. 產生圖表
python3 scripts/plot_results.py --lang en
python3 scripts/plot_results.py --lang zh    # 中文版（選用）
```

`data/mtplx_v5.jsonl` 那一份是用 MTPLX 內建的 `prefill-ladder` 工具產生的（不是 `bench_inline.py`），跑的是 `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed`：

```bash
mtplx prefill-ladder \
  --model /Users/david/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed \
  --mode sustained --mtp --depth 3 --disable-thinking \
  --sizes 512,1024,2048,4096,8192,16384,32768 --max-tokens 128
```

bench 腳本本身會處理 streaming、解析 Server-Sent Events、把 thinking token（`reasoning_content`）和可見內容 token 分開計算、產出每次 run 的統計值。它也會自動處理 server 不支援 `/v1/cache/clear` 的情況——把回傳的 404 當成沒這回事。

---

## 九、限制與後續可做的方向

這份 benchmark 只測了單一 sequence 的情境。rapid-mlx 和 omlx 都支援 continuous batching，在多人併發下這份比較會變得很不一樣——高負載時 omlx 跟 rapid-mlx 可能會把優勢拉得更開，因為它們可以把 prefill 攤分到同時進來的多個請求上。dflash-mlx 的 speculative decoding 本質上是 single-stream，無法從 batching 受惠。

我們只測了 4-bit MoE 模型。dense 模型（例如 Qwen3-32B-Dense）和啟用更多參數的 MoE 模型瓶頸不一樣；dense 32B 的 prefill 會從記憶體頻寬瓶頸變成計算瓶頸，這時框架的相對排名就可能改變。KV cache 量化是 rapid-mlx 與 mlx-vlm 都支援的功能，但本次沒試——它應該會縮小 omlx 在 32K 的領先幅度，因為 rapid-mlx 特別會從量化 KV 受益。

mlx-vlm + dflash 的組合我們有發現但沒測到。這可能是最值得做的後續實驗：vlm 在中等脈絡的 prefill 最強、dflash 在短脈絡的 decode 最快，把兩者疊加可以驗證效果是相加還是相互干擾。

`/no_think` 在這次只被部分尊重——mlx-vlm 和 dflash-mlx 完全不輸出 thinking，但 rapid-mlx 和 omlx 仍然會吐 reasoning token。這兩個框架的 decode tps 數字其實混合了 thinking token 的速率和可見內容的速率，雖然兩者速度接近但並不完全相同。更嚴謹的後續做法是透過 chat template 的 `enable_thinking=false` 參數而不是 prompt 內的慣例。（dflash-mlx v0.1.7 在 CLI 接受 `--chat-template-args '{"enable_thinking":false}'`，我們重測時用的就是這個。）

最後，我們沒有測超過 32K 的脈絡。在 64K 與 128K 下，KV cache 會變成主要的記憶體佔用（這個模型 fp16 KV 大約是 16 GB 與 32 GB），這時測出來的差異會包含很多記憶體壓力的成分而不只是計算速度。對某些應用來說這個區段才是真正關鍵，是個值得回頭再做的後續題目。
