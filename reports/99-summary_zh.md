# 跨框架交叉比較與經驗總結

[English Version](99-summary.md)

> **關於這份 summary 裡的 dflash-mlx 數字。** dflash-mlx 數字來自 2026-05-25
> 的 v0.1.7 測試，`cooldown=60`。其他四個框架是 2026-05-09 用 lab 預設的
> `cooldown=2` 測的。所以這份跨框架比較在長脈絡略低估了其他框架——如果
> 用較長的 cooldown 重測它們，差距會縮小。dflash 完整方法論請見
> [`reports/03-dflash-mlx_zh.md`](03-dflash-mlx_zh.md)。

## 誰贏了哪一項

各個指標的冠軍分布得相當乾淨。dflash-mlx 在我們測過的每個脈絡長度都拿下
decode 冠軍——短脈絡段（64 到 2K 是 139–155 tps）領先 20–30%，
長脈絡段（4K 到 16K 是 119–128 tps、32K 是 121 tps）領先 25–45%。
omlx 拿下穩定度——它的 decode tps 標準差在幾乎每一格都是四個框架裡最低的，
TTFT 標準差在中等脈絡只有 3–25 ms，比其他框架穩定 10 倍。omlx 在長脈絡段的
prefill 也最強（4K 峰值 3,989 tps），decode 則在每個脈絡長度都是亞軍。
mlx-vlm 拿下短脈絡 decode 穩定度（64 跟 512 token 的標準差都是 0.2 tps），
8K 的 prefill 也是並列最高。rapid-mlx 沒有在任何單一指標拿到絕對冠軍，
但每個地方都有競爭力，而且它的功能組合最豐富。

至於圖像、音訊、影片這類多模態輸入——mlx-vlm 是這組裡唯一支援的，所以那部分沒得比。

## 速度與穩定的取捨

這份資料裡最有意思的型態，是框架排名隨「你想優化什麼」劇烈變化。
dflash-mlx 在 decode tps 整段領先；omlx 的峰值 decode 比較低（短脈絡 124 tps、
32K 82 tps）但每次跑都完美穩定，要綁 SLA 容易得多。在這兩者之間做選擇，
問的是「你的流量型態是什麼」，不是「哪個框架比較好」。

從這個型態可以歸納出兩個原則。第一，峰值速度跟穩定性往往會交換——
speculative decoding 的加速效果取決於 draft 預測命中，而命中率隨內容變化，
所以速度也會跟著變化。第二，沒有萬用冠軍。這裡每個框架都有自己會贏的區段、
也有自己會輸的區段，挑對框架的第一步是先搞清楚自己的工作負載落在哪個區段。

## 為什麼 rapid-mlx 在第一輪測試看起來更強

我們最初的 benchmark pass 裡，rapid-mlx 的 prefill 數字非常驚人（4K 超過 5,000 tps、8K 超過 100,000 tps）。後來才發現那些數字是 prefix cache 命中的副作用——warm-up run 把 prompt 前綴快取起來，第二到第五次 run 直接重用快取的 KV，根本沒跑 prefill。修正方法是啟動時加上 `--disable-prefix-cache`，加完之後 rapid-mlx 的 prefill 在 4K 掉到比較合理的 3,070 tps，落後 omlx 跟 mlx-vlm。

這件事值得提，是因為在 production 裡，prompt 共用 system 前綴或 RAG header 時，你的流量*確實*會從 prefix cache 受惠。所以如果你真實的工作負載 prefix 重複率高（例如固定的 system message），rapid-mlx 的實效 prefill 會比這份冷快取 benchmark 顯示的高很多。冷數字是下界，cache 命中是上界。

## TTFT 對 RAG 很要命

當脈絡爬到 16K 以上，TTFT——也就是第一個 token 出現給使用者看之前的等待——
變成真實的 UX 問題。32K 下這份 benchmark 測到的最快 TTFT 是 8.1 秒
（dflash-mlx），omlx 是 12.7 秒，其他更高。對 RAG 應用來說，那是一段
很長的「螢幕空白讓使用者乾瞪眼」的時間。修補方式一部分是 UX
（顯示思考中的動畫、把 prompt 串回來、或秀出部分檢索結果），另一部分是
架構——如果你的 RAG pipeline 每查一次就拉 32K 的脈絡進來，你應該預期
模型開始回應之前會有大約 8–13 秒的開銷，並設計時把這個納進去考量。

## 從方法論學到的事

關於怎麼公平 benchmark 這些框架，我們在這次過程裡學到一些值得記下來的事。

**Prefix cache 是雙面刃。** 我們第一輪跑出物理上不可能的 prefill 數字，正是因為快取的 KV 重用被一起量到。修法是在 server 啟動時關掉 prefix cache（用各框架特定的 flag），或者在每次 run 加唯一前綴打亂快取。光是 run 之間呼叫 `/v1/cache/clear` 並*不夠*——我們試過了，殘留的快取效果還是會出現。

**長脈絡下變異測試是必要的。** 短脈絡（512 token 以下）每個框架五次 run 的標準差都在 3 tps 以下。那個區間單次 benchmark 沒問題。但到了 16K 以上，真實的 run-to-run 變異就出現了，單一資料點可能誤差 10–20%。多跑五次（每個框架多花約 10 分鐘）的成本，比根據雜訊資料做出錯誤決定的成本小得多。

**Apple Silicon 上 run 之間的 cooldown 是真的方法論變數。**
長脈絡下，decode 速率明顯取決於晶片在 request 之間有多放鬆熱壓力。
短 cooldown（2 秒）量到的是背靠背持續吞吐量；長 cooldown（60 秒）
量到的是單請求的峰值表現。這份 summary 裡的 dflash-mlx 數字用的是
`cooldown=60`，其他框架是 `cooldown=2`。如果你在做 production sizing，
用符合你真實流量型態的 cooldown 自己量一次。

**Qwen3 的 `/no_think` 各框架尊重程度不一。** 四個框架裡只有 mlx-vlm 跟 dflash-mlx 完全壓制了 reasoning token。rapid-mlx 跟 omlx 仍然會吐 `reasoning_content`，rapid-mlx 甚至連 `max_tokens` 都不對 thinking 階段生效——我們在請求 256 token 時看到過長達 2,155 token 的輸出。如果你需要嚴格的 thinking 控制，請透過 chat template 參數設 `enable_thinking=false`，不要靠 prompt 內的慣例。

## 接下來想測的東西

最有意思的後續實驗是 mlx-vlm 配 `--draft-kind dflash`。mlx-vlm 在中等脈絡的 prefill 最強、dflash 在短脈絡的 decode 最快；組合起來可能兩全其美，也可能爆發出某種交互效應讓兩個加速都失效。這次來不及做。

第二優先是「持續批次（continuous batching）」在多人併發下的吞吐量。rapid-mlx 跟 omlx 都支援，但這次只測了單一 sequence 的延遲。在現實的多人負載下，批次化跟非批次化框架的差距會放大，差多少值得知道。dflash-mlx 由於 speculative decoding 本質上是 single-stream，從 batching 受惠不了——這意味著它的短脈絡優勢在 production 可能會消失。

第三是 KV cache 量化。rapid-mlx 跟 mlx-vlm 都支援 4-bit 跟 8-bit KV cache；這應該會大幅幫助長脈絡效能，特別是目前在 32K 落後 omlx 的 rapid-mlx。如果量化 KV 能補回那個差距，production 的框架選擇就要重新洗牌。

第四是超過 32K——到了 64K 跟 128K，KV cache 記憶體壓力（這個模型 fp16 KV 大概分別是 16 GB 跟 32 GB）開始主導，這時候測到的差異就會包含很多記憶體管理的成分而不只是計算吞吐量。那已經是另一個 benchmark 了，但對某些真實應用來說那才是關鍵區段，重要性正在快速上升。
