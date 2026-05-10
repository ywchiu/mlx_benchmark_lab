# mlx-vlm 個別測試報告

## 啟動指令

```bash
python3 -m mlx_vlm server \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --port 8765
```

> mlx-vlm 是視覺語言模型（VLM）框架，本測試僅以純文字輸入測速。
> 它支援 `--draft-model` + `--draft-kind dflash`，可以結合 dflash 加速（本次未測）。

---

## 完整統計（n=5）

### Decode tps

| size | median | mean | stddev | min | max |
|---:|---:|---:|---:|---:|---:|
| 64 | 95.5 | 95.4 | 0.2 | 95.1 | 95.6 |
| 512 | 94.8 | 94.8 | 0.2 | 94.5 | 95.1 |
| 2,048 | 88.5 | 89.2 | 4.1 | 83.3 | 93.1 |
| 4,096 | 91.4 | 91.5 | 0.3 | 91.1 | 92.0 |
| 8,192 | 87.2 | 85.8 | 2.4 | 82.4 | 87.9 |
| 16,384 | 83.1 | 82.7 | 1.1 | 81.3 | 84.0 |
| 32,768 | 67.7 | 67.3 | 2.7 | 63.1 | 70.6 |

### Prefill tps

| size | median | mean | stddev | min | max |
|---:|---:|---:|---:|---:|---:|
| 64 | 592 | 605 | 41 | 570 | 677 |
| 512 | 1,826 | 1,817 | 30 | 1,779 | 1,846 |
| 2,048 | 3,370 | 3,287 | 126 | 3,143 | 3,395 |
| 4,096 | 3,672 | 3,679 | 21 | 3,655 | 3,712 |
| 8,192 | 3,818 | 3,762 | 150 | 3,498 | 3,853 |
| 16,384 | 2,850 | 2,855 | 80 | 2,741 | 2,964 |
| 32,768 | 2,075 | 2,064 | 70 | 1,987 | 2,162 |

### TTFT (ms)

| size | median | mean | stddev | min | max |
|---:|---:|---:|---:|---:|---:|
| 64 | 133 | 131 | 8 | 117 | 139 |
| 512 | 240 | 242 | 4 | 238 | 247 |
| 2,048 | 498 | 511 | 20 | 494 | 534 |
| 4,096 | 908 | 907 | 5 | 899 | 913 |
| 8,192 | 1,739 | 1,767 | 74 | 1,723 | 1,898 |
| 16,384 | 4,650 | 4,645 | 131 | 4,471 | 4,834 |
| 32,768 | 12,761 | 12,840 | 432 | 12,247 | 13,326 |

---

## 觀察與分析

### 強項
- **變異最低**：64–512 區間 decode stddev 僅 0.2 tps（極度穩定）
- **Prefill 在 8K 達全測試最高**：3818 tps（贏過 omlx 的 3467 與 rapid 的 2696）
- **Decode 衰退率最低**：64 → 32K 只衰退 -29%（曲線最平）
- **TTFT 中位數低且穩定**：64 token 133ms（贏過 rapid 169 與 omlx 148）
- **唯一支援多模態**：可吃 image / audio / video 輸入

### 弱項
- **Decode tps 全程墊底**：所有 context 都比另外三個框架慢 20–30%
  - 64 tokens：95.5 tps，比 rapid 慢 24%
  - 32K tokens：67.7 tps，比 omlx 慢 18%
- VLM stack 為支援多模態的額外開銷，純文字場景不划算

### 衰退率
從 64 → 32K：**95.5 → 67.7 tps（-29%）**——4 個框架最緩

---

## 適用情境

✅ **圖文 / 影音 / 文檔多模態應用**（無可替代）
✅ 對變異敏感、需要可預測的響應時間
✅ 中等 context（4K–8K）的純文字生成（decode 雖低但 prefill 強）

❌ 純文字 + 高吞吐量需求（rapid/omlx/dflash 都更快）
❌ 短互動 chat（其他框架 decode 快 25%）

---

## 進階：mlx-vlm + DFlash 組合

mlx-vlm 內建 `--draft-model` + `--draft-kind dflash` 支援，理論上可以同時得到：
- mlx-vlm 的 prefill 強項
- dflash 的小 context decode 加速

```bash
python3 -m mlx_vlm server \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --draft-model z-lab/Qwen3.6-35B-A3B-DFlash \
  --draft-kind dflash \
  --port 8765
```

**本次未測**——值得後續實驗。但要小心 32K 崩盤問題可能會繼承。

---

## 注意事項

1. 預設 port 8080，本測試手動指定 8765
2. 預設 `prefill-step-size=2048`，長 prompt 會分批 prefill
3. 沒有 `/health` endpoint，readiness 用 `GET /v1/models`
4. CLI 用法：`python3 -m mlx_vlm server`（也可用 `mlx_vlm.generate` 做單次生成）
5. 模型若有 vision components（如本測試的 Qwen3.6-35B-A3B-4bit 內含 `preprocessor_config.json`），可直接吃 image 輸入
