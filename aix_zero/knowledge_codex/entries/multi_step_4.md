# Knowledge Codex Entry: `MULTI_STEP_4` — Four-Step Compositional Reasoning Chains

- **Category:** Deep Multi-Step Composition (Knowledge Is Power v11 + Web)
- **Introduced In:** `AIX-Zero v11-KIP-Web`
- **Tokenizer Prefix:** `Q:Chain4:`
- **Governing Law / Formula:** `R = (((a * b) + c) - d) + e`
- **Expert snapshot:** Consult the versioned cartridge manifest for filename and source expert; specialist quality has not been evaluated.
- **Held-Out Verification Accuracy:** **`100.0%`** (5/5 unseen test problems)

## 1. Domain Concept & Symbolic Grammar

Chains four sequential operations with triple accumulator auto-carry across semicolons.

```text
Prompt Pattern     : Q:Chain4:{a}*{b}+{c}-{d}+{e}=
Reasoning Grammar  : <think>{a}*{b}={p};{p}+{c}={q};{q}-{d}={r};{r}+{e}={s}</think><ans>{s}</ans>
```

## 2. Live Internet Research Citations (`Wikipedia` + `arXiv`)

- **Wikipedia — [Order of operations](https://en.wikipedia.org/wiki/Order_of_operations):** In mathematics and computer programming, the order of operations is a collection of conventions about which arithmetic operations to perform first in order to evaluate a given mathematical expression.
- **arXiv — [The Deep Arbitrary Polynomial Chaos Neural Network or how Deep Artificial Neural Networks could benefit from Data-Driven Homogeneous Chaos Theory](http://arxiv.org/abs/2306.14753v1):** Artificial Intelligence and Machine learning have been widely used in various fields of mathematical computing, physical modeling, computational science, communication science, and stochastic analysis. Approaches based on Deep Artificial Neural Networks (DANN) are very popular in our days. Depending on the learning tas

## 3. Multi-LLM Symbiotic Council Conversation (`Knowledge Is Power`)

- **Turn 1 — Theorist (`aix-8b:latest`):** First, the user is asking about the order of operations in mathematics and computer programming, referencing Wikipedia. Then they mention an arXiv paper on polynomial chaos theory,
- **Turn 2 — Solver & Critic (`aix3:latest`):** Alright, so I need to solve this equation: 3 multiplied by 4 plus 5 minus 2 plus 6. Let me break it down step by step. First, let's do the multiplication part because that seems straightforward
- **Turn 3 — Curriculum Consensus (`aix-8b:latest`):** Okay, let me think about this problem. The user is asking for a concise explanation of why distilling the specific solver interaction into A
- **Turn 4 — `AIX-Zero v12` Student Verification:** `Q:Chain4:3*4+5-2+6=<think>3*4=12;12+5=17;17-2=15;15+6=21</think><ans>21</ans>` (`PASS`)
- **Recorded At:** `2026-11-05T07:26:41.273088+00:00`

## 4. Formally Verified Worked Examples (Live `AIX-Zero v12` Traces)

| Held-Out Prompt | Full Neuro-Symbolic `<think>` & `<ans>` Trace | Gold | Confidence | Ponder | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Q:Chain4:3*4+3-1+6=` | `Q:Chain4:3*4+3-1+6=<think>3*4=12;12+3=15;15-1=14;14+6=20</think><ans>20</ans>` | `20` (PASS) | `0.8761` | `1.0` | `299.09ms` |
| `Q:Chain4:2*2+3-2+6=` | `Q:Chain4:2*2+3-2+6=<think>2*2=4;4+3=7;7-2=5;5+6=11</think><ans>11</ans>` | `11` (PASS) | `0.8759` | `1.001` | `295.84ms` |
| `Q:Chain4:3*3+3-2+6=` | `Q:Chain4:3*3+3-2+6=<think>3*3=9;9+3=12;12-2=10;10+6=16</think><ans>16</ans>` | `16` (PASS) | `0.8725` | `1.0` | `336.64ms` |
| `Q:Chain4:4*3+5-1+6=` | `Q:Chain4:4*3+5-1+6=<think>4*3=12;12+5=17;17-1=16;16+6=22</think><ans>22</ans>` | `22` (PASS) | `0.8693` | `1.0` | `281.17ms` |
| `Q:Chain4:4*3+5-2+4=` | `Q:Chain4:4*3+5-2+4=<think>4*3=12;12+5=17;17-2=15;15+4=19</think><ans>19</ans>` | `19` (PASS) | `0.8636` | `1.0` | `282.97ms` |
