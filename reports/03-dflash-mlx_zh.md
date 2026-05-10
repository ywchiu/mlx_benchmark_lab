# dflash-mlx 詳細報告

[English Version](03-dflash-mlx.md)

## 怎麼跑這個測試

```bash
dflash-serve \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --draft z-lab/Qwen3.6-35B-A3B-DFlash \
  --port 8765
```

關於 dflash-mlx 最重要要先知道的一件事是：`z-lab/Qwen3.6-35B-A3B-DFlash` 是個 *draft 模型*，不是 target。它只有 8 層 transformer 加上一個 output head，是設計成跟完整 35B 模型搭配以驅動 speculative decoding。如果你想把它當主 `--model` 載入，啟動時就會因為缺少 91 個權重（它沒有的那些 layer）而當掉。正確的呼叫方式永遠是「`--model <完整 target>` 配 `--draft <DFlash 變體>`」。

---

## 完整統計（每格 5 次）

### Decode tps

| size | 中位數 | 平均 | 標準差 | 最小 | 最大 |
|---:|---:|---:|---:|---:|---:|
| 64 | **167.3** | 167.3 | 2.9 | 163.1 | 170.9 |
| 512 | 122.9 | 122.1 | 2.2 | 118.3 | 123.5 |
| 2,048 | **160.1** | 158.7 | 2.8 | 153.8 | 160.5 |
| 4,096 | 104.5 | 104.4 | 0.8 | 103.5 | 105.6 |
| 8,192 | 96.3 | 95.9 | 1.6 | 93.6 | 97.9 |
| 16,384 | 84.1 | 84.4 | 2.4 | 81.7 | 88.0 |
| 32,768 | **12.6** ⚠️ | 12.8 | 1.3 | 11.3 | 14.5 |

### Prefill tps

dflash-serve 的 streaming `usage` 物件裡不會回傳 `prompt_tokens`——每次都是零。所以無法直接從 API 回應算 prefill tps。如果你需要估值，可以從 TTFT 加上自己做 tokenization 反推（這次測試的實際輸入 token 數，依七種 size 分別是 77/437/1677/3333/6636/13250/26475，這是 Qwen tokenizer 處理後的結果）。

### TTFT（毫秒）

| size | 中位數 | 平均 | 標準差 | 最小 | 最大 |
|---:|---:|---:|---:|---:|---:|
| 64 | 334 | 340 | 17 | 328 | 371 |
| 512 | 360 | 364 | 8 | 357 | 378 |
| 2,048 | 587 | 572 | 36 | 507 | 591 |
| 4,096 | 1,072 | 1,063 | 27 | 1,019 | 1,090 |
| 8,192 | 2,205 | 2,217 | 33 | 2,178 | 2,261 |
| 16,384 | 6,023 | 5,957 | 185 | 5,651 | 6,119 |
| 32,768 | **31,205** ⚠️ | 31,796 | 2,663 | 28,469 | 35,830 |

---

## 數據說明

dflash-mlx 是兩個極端的故事。短脈絡下它是無爭議的贏家：64 token 的 167 tps 比第二名快 35%，2,048 token 的 160 tps 也是壓倒性領先。run 之間的標準差都低於 3 tps，所以這個加速是穩定的——speculative decoding 一旦命中，就會穩定地命中。

但到了 32K，dflash-mlx 整個崩了。decode tps 跌到 12.6，比第二慢的框架還要再慢約 6 倍。TTFT 飆到 31 秒，是其他框架的兩倍多。run 之間的 TTFT 變異更是膨脹到 2.7 秒——這次測試裡最高的變異。

根本原因是結構性的。speculative decoding 的運作方式是讓一個小 draft 模型生成 token 候選，然後讓主模型平行 verify。短脈絡下這是明顯的勝利，因為 draft pass 相對於 main pass 便宜得多，而 verify 批次往往會接受大部分 draft 提議。但長脈絡下兩件事壞掉。第一，draft 模型也得跟主模型一樣吃完整個 prompt——當 prompt 是 32K token 時，draft 在做的工作幾乎跟主模型一樣多，draft 比較便宜的優勢就消失了。第二，draft 的預測準確度會隨脈絡變長而下降，因為長脈絡 conditioning 對小模型來說本來就比較難。預測失敗會觸發昂貴的 verify-then-rollback 循環。到了 32K，verify 成本主導，每接受一個 token 你就在付 draft pass 加 main pass 加 rollback 工作的代價——所以 decode 速率不只是回到 baseline，而是掉到 baseline 之下。

512 token 的結果值得多一句說明。那一格 decode tps 是 122.9，跟非 speculative 的 baseline 幾乎一樣——意思是該脈絡長度下 speculative decoding 沒有提供任何加速。2K 跟 64 token 都受惠很多，512 卻沒有。我們認為這是因為 draft 命中率本來就跟內容有關，而 512 token 的自然語言摘要 prompt 剛好不太符合 draft 模型的預測模式。你的程式碼補全或結構化輸出工作負載可能會比這數字好。

---

## 什麼時候該用 dflash-mlx

當你的脈絡長度受控在 4K 以下、且輸出是高度可預測的內容（程式碼、JSON、結構化模板、重複內容）時，這是對的選擇。在這個情境下你能拿到比競爭對手快 35% 的 decode 加速，而稍微高一點的 TTFT（比 omlx 多 200–300 ms）對「生成主導型」的工作負載來說鮮少是瓶頸。

絕對不要在 16K 之後用它，尤其 32K。12 tps 的 decode 速率對任何互動式應用來說根本不能用。也別在 TTFT 敏感的工作負載上用它——就算在 64 token，dflash 的 334 ms TTFT 也是 omlx 的 148 ms 的兩倍多。

---

## 注意事項

沒有 `/health` endpoint——用 `GET /v1/models` 檢查 server 是否就緒。rapid-mlx 暴露的 `/v1/cache/clear` 在 dflash-serve 上也不存在，但 bench 腳本會自己吞掉 404。

附帶的 CLI 工具 `dflash`（單次生成）和 `dflash-benchmark`（內建 baseline 對 DFlash 的比較）都很實用，如果你想快速測試 speculative 加速效果但不想啟一個完整 server，可以用它們。`dflash-benchmark` 特別方便，因為它會直接在同一個 prompt 上比較 baseline MLX decoding 跟 DFlash decoding，加速差距不用自己算。

32K 脈絡看起來是個硬上限。我們沒測 64K，但預期失敗模式只會更糟——draft 模型的 context window 應該跟主模型的一樣重要，而這個 draft 是在特定 context 預算下訓練的。
