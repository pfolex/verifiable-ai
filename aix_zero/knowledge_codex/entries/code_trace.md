# Knowledge Codex Entry: `CODE_TRACE` — Python Function Execution Trace

- **Category:** Program Verification
- **Introduced In:** `AIX-Zero v4`
- **Tokenizer Prefix:** `Q:Code:`
- **Governing Law / Formula:** `def f(x): return a*x + b; f(k) = a*k + b`
- **Expert snapshot:** Consult the versioned cartridge manifest for filename and source expert; specialist quality has not been evaluated.
- **Held-Out Verification Accuracy:** **`100.0%`** (5/5 unseen test problems)

## 1. Domain Concept & Symbolic Grammar

Binds formal parameter x=k in Python function definition and traces return expression.

```text
Prompt Pattern     : Q:Code:def f(x):return {a}*x+{b};f({k})=
Reasoning Grammar  : <think>{a}*{k}={p};{p}+{b}={res}</think><ans>{res}</ans>
```

## 4. Formally Verified Worked Examples (Live `AIX-Zero v12` Traces)

| Held-Out Prompt | Full Neuro-Symbolic `<think>` & `<ans>` Trace | Gold | Confidence | Ponder | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Q:Code:def f(x):return 4*x+6;f(4)=` | `Q:Code:def f(x):return 4*x+6;f(4)=<think>4*4=16;16+6=22</think><ans>22</ans>` | `22` (PASS) | `0.8612` | `1.003` | `97.26ms` |
| `Q:Code:def f(x):return 6*x+6;f(5)=` | `Q:Code:def f(x):return 6*x+6;f(5)=<think>6*5=30;30+6=36</think><ans>36</ans>` | `36` (PASS) | `0.8346` | `1.0` | `85.02ms` |
| `Q:Code:def f(x):return 6*x+1;f(2)=` | `Q:Code:def f(x):return 6*x+1;f(2)=<think>6*2=12;12+1=13</think><ans>13</ans>` | `13` (PASS) | `0.8749` | `1.0` | `81.59ms` |
| `Q:Code:def f(x):return 5*x+4;f(4)=` | `Q:Code:def f(x):return 5*x+4;f(4)=<think>5*4=20;20+4=24</think><ans>24</ans>` | `24` (PASS) | `0.8487` | `1.0` | `86.91ms` |
| `Q:Code:def f(x):return 2*x+4;f(5)=` | `Q:Code:def f(x):return 2*x+4;f(5)=<think>2*5=10;10+4=14</think><ans>14</ans>` | `14` (PASS) | `0.8725` | `1.0` | `86.45ms` |
