# 模型選型:Qwen2.5 7B vs 3B(AWQ-int4)

**結論:production 用 7B。** 3B 快一倍,但掉的分集中在這系統的核心差異化能力(圖推理),且該缺口無法用調參補回。

實驗均在 held-out 協定下進行(dev 選 / test 報),語料 = 3 SOP、29 節點、48 邊,題庫 = `fab_queries_dev.json`(8 answerable)+ `fab_queries_test.json`(19 answerable)。

---

## 1. 7B vs 3B(同 AWQ-int4、同題、runs=3)

| 指標(held-out TEST) | 7B | 3B | 備註 |
|------|:---:|:---:|------|
| Answer keyword-match | 100.0% | 94.2% | 生成側 |
| Answer correctness (LLM-judge) | 100.0% | ~94–100% | 生成側(跨 run 抖動) |
| Retrieval recall@k (model) | 96.5% | 96.5% | **0 變化** — 檢索與模型無關 |
| Retrieval recall (evidence) | 100.0% | 100.0% | **0 變化** |
| avg latency(full pipeline) | 8746 ms | 4178 ms | 3B ≈ 快一倍 |

**3B 退步集中在圖推理題:**

```
interlock_condition   kw 100→75%   judge 83→50%
multihop_dependency   kw 100→78%
cross_doc_dependency  kw 100→92%
```

單步題(`step_dependency`、`step_requires_status`)兩個模型都 100%。換 LLM **只動得了生成側**(keyword / judge),動不了檢索(graph + 向量是 model-agnostic)。

## 2. 那 3B 能不能調回來?(cap + prompt 消融,皆在 3B 上)

**(a) 進模型的 triple 量(rerank 後 top-N cap):**

| cap | test kw | avg #triples | latency |
|----:|:---:|:---:|:---:|
| 4 | 76.3% | 4.0 | 819 ms |
| 8 | 95.6% | 7.8 | 982 ms |
| None(現況) | 89–93% | 17.3 | 1173 ms |

cap≈8 方向性略好 + 延遲省 ~16%,但 **paired(cap8 − None)95% CI [−3.5, +17.5] 含 0 → 不顯著**;砍到 4 會崩。倒 U 形,下限約 6。

**(b) prompt(few-shot / concise vs 現用,kw + judge):**

| variant | test kw | test judge |
|------|:---:|:---:|
| baseline(現用) | 94.3% | **100.0%** |
| few-shot | 93.9% | 84.2% |
| concise | 95.6% | 94.7% |

**現用 prompt 最強**;加格式範例與精簡規則都讓 judge 退步(few-shot 把小模型帶歪、concise 把 `sop_step_sequence` 的 judge 弄到 0%)。

→ **interlock 在所有 cap / prompt 變體下都修不好** → 這是 3B 的推理上限,不是調參問題。

## 3. 決策依據

1. **3B 的弱點 = 本系統的賣點。** interlock / multihop 圖推理正是與「普通 vector RAG」拉開差距之處;3B 在這些題最弱,且無法用 cap / prompt 補回。
2. **延遲優勢未跨門檻。** 8s vs 4s 對 demo 都是「等幾秒」,非互動級;此專案是 portfolio / demo,非高 QPS 線上服務,correctness > latency。
3. **7B 在此 eval 飽和(全 100%)** 對 demo 而言是「它就是會動」,反而是優點。

## 4. 誠實 caveat

題庫小(8 dev / 19 test,部分類別 n=2),vLLM 有 run 間抖動,judge 與生成共用同一模型(自評偏差)。**以上差異多數無法達到統計顯著**;站得住的是**質性 trade-off**(3B 快一倍、圖推理類別明顯掉、且調不回來),而非精確數字。要做細部調參結論,需先擴大題庫。
