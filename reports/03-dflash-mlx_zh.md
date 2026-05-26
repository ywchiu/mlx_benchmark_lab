# dflash-mlx 詳細報告

[English Version](03-dflash-mlx.md)

## 怎麼跑這個測試

```bash
dflash serve \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --draft z-lab/Qwen3.6-35B-A3B-DFlash \
  --port 8765 \
  --no-prefix-cache \
  --no-prefix-cache-l2 \
  --chat-template-args '{"enable_thinking":false}'
```

使用 dflash-mlx **v0.1.7**，跑 `bench_inline.py`，每格 5 次、`max_tokens=256`、
run 之間 `cooldown=60`（給 M5 Max 足夠時間散熱，量到的是這個框架的峰值表現
而不是背靠背負載降頻的數字）。兩層 prefix cache 都關掉，以量到每次請求的
冷計算成本。原始資料：[`data/dflash_v6_c60.jsonl`](../data/dflash_v6_c60.jsonl)。

關於 dflash-mlx 最重要要先知道的一件事是：`z-lab/Qwen3.6-35B-A3B-DFlash`
是個 *draft 模型*，不是 target。它只有 8 層 transformer 加上一個 output
head，是設計成跟完整 35B 模型搭配以驅動 speculative decoding。如果你想把它
當主 `--model` 載入，啟動時就會因為缺少 91 個權重（它沒有的那些 layer）
而當掉。正確的呼叫方式永遠是「`--model <完整 target>` 配
`--draft <DFlash 變體>`」。

---

## 完整統計（每格 5 次）

### Decode tps

| size | 中位數 | 平均 | 標準差 | 最小 | 最大 |
|---:|---:|---:|---:|---:|---:|
| 64 | 148.7 | 147.8 | 1.8 | 145.0 | 149.5 |
| 512 | 138.9 | 138.4 | 2.1 | 134.7 | 140.1 |
| 2,048 | **154.5** | 154.4 | 0.4 | 153.6 | 154.7 |
| 4,096 | 127.7 | 127.7 | 0.2 | 127.5 | 128.0 |
| 8,192 | 124.6 | 124.6 | 0.2 | 124.4 | 125.0 |
| 16,384 | 119.0 | 118.5 | 0.9 | 117.3 | 119.2 |
| 32,768 | **121.2** | 121.2 | 0.2 | 121.0 | 121.4 |

### Prefill tps

| size | 中位數 | 平均 | 標準差 | 最小 | 最大 |
|---:|---:|---:|---:|---:|---:|
| 64 | 182 | 178 | 8 | 165 | 186 |
| 512 | 801 | 797 | 27 | 762 | 833 |
| 2,048 | 2,229 | 2,219 | 34 | 2,176 | 2,256 |
| 4,096 | 2,971 | 3,005 | 96 | 2,946 | 3,175 |
| 8,192 | **3,425** | 3,427 | 5 | 3,423 | 3,435 |
| 16,384 | 3,633 | 3,631 | 10 | 3,612 | 3,639 |
| 32,768 | 3,271 | 3,277 | 17 | 3,258 | 3,299 |

### TTFT（毫秒）

| size | 中位數 | 平均 | 標準差 | 最小 | 最大 |
|---:|---:|---:|---:|---:|---:|
| 64 | 434 | 444 | 21 | 424 | 478 |
| 512 | 548 | 551 | 19 | 527 | 576 |
| 2,048 | 753 | 757 | 12 | 744 | 771 |
| 4,096 | 1,123 | 1,111 | 34 | 1,051 | 1,132 |
| 8,192 | 1,938 | 1,937 | 3 | 1,933 | 1,939 |
| 16,384 | 3,647 | 3,650 | 10 | 3,642 | 3,668 |
| 32,768 | **8,094** | 8,079 | 42 | 8,027 | 8,127 |

---

## 數據說明

dflash-mlx 在我們測過的每個脈絡長度都拿下 decode tps 冠軍。峰值在 2K
（154 tps），因為自然語言摘要剛好跟 draft 模型的預測 pattern 很契合。
從 4K 一路到 32K decode 線都維持在 120 tps 以上，長脈絡的標準差都穩定低於
1 tps——只要給晶片足夠的散熱裕度，speculative decoding 同時把速度跟穩定度
都交出來。

Prefill 在 16K 達峰（3,633 tps），到 32K 也維持在 3,200 tps 以上。
32K TTFT 是 8.1 秒，是本次測試框架裡最低的。

512 token 那一格值得加註：那裡的 decode（138.9 tps）跟非 speculative 的
baseline 幾乎相同——意思是在那個特定尺寸的這個自然語言工作負載下，
speculative decoding 沒有提供可量到的加速。64、2K 跟更長脈絡的格子都受惠
明顯。draft 命中率跟內容相關，所以你的程式碼補完或結構化輸出工作負載
可能會比這裡的自然語言數字更好。

---

## 什麼時候該用 dflash-mlx

如果 decode 吞吐量是優先項、TTFT 可以另外控制（warm-up 請求、已知前綴
開 prefix cache、或單純可以接受短脈絡比 omlx 多 ~500 ms 的 TTFT），
dflash-mlx 就是對的選擇。它在每個測試脈絡長度都領先，只要熱壓力被控制好，
穩定度也很好（多數格子 stddev < 1 tps）。

如果你的工作負載有已知的 prefix pattern——多輪對話、agent loop 裡重複的
system prompt、有穩定 header 的 RAG——請把 `--no-prefix-cache` 跟
`--no-prefix-cache-l2` 拿掉。v0.1.7 的 prefix cache 預設就開著，
就是為這種場景設計的。我們在這裡把它關掉只是為了量到每次請求的冷計算，
跟其他框架的測試方法一致。

如果穩定度比峰值吞吐量重要，omlx 仍然是個強選項——它讓出大約 10% 的峰值
decode，但 TTFT 變異是這次測試裡最低的。

---

## 注意事項

沒有 `/health` endpoint——用 `GET /v1/models` 檢查 server 是否就緒。
bench 腳本對 `/v1/cache/clear` 的呼叫會收到 404（自動處理掉），
因為 dflash 是用 CLI flag 而不是 HTTP endpoint 來管 prefix cache。

附帶的 CLI 工具 `dflash generate`（單次生成）跟 `dflash benchmark`
（內建 baseline 對 DFlash 的比較）都很實用，如果你想快速測試 speculative
加速效果但不想啟一個完整 server，可以用它們。

Cooldown 在 Apple Silicon 上很重要。上面的數字是 run 之間 `cooldown=60`；
cooldown 較短時 M5 Max 在長脈絡會降頻，32K 的 decode 速率會明顯下降
（背靠背、無 cooldown 的 benchmark 會報出比這份報告低的數字）。
如果你在做 production sizing，請用符合你流量型態的 cooldown 自己量。

---

## 致謝

感謝這位回報的社群網友，指出原本 dflash-mlx v0.1.0 的數字已經不能反映
目前行為，並提供了完整可重現的指令讓我用 v0.1.7 重新測試。
