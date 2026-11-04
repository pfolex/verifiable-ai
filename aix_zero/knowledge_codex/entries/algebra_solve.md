# Knowledge Codex Entry: `ALGEBRA_SOLVE` — Linear Single-Variable Algebraic Equations

- **Category:** Symbolic Algebra
- **Introduced In:** `AIX-Zero v1`
- **Tokenizer Prefix:** `Q:`
- **Governing Law / Formula:** `a*x + b = c  ==>  x = (c - b) / a`
- **Held-Out Verification Accuracy:** **`100.0%`** (5/5 unseen test problems)

## 1. Domain Concept & Symbolic Grammar

Isolates unknown variable x by subtracting constant offset b and dividing by coefficient a.

```text
Prompt Pattern     : Q:{a}x+{b}={c},x=
Reasoning Grammar  : <think>{c}-{b}={d};{d}/{a}={x}</think><ans>{x}</ans>
```

## 3. Formally Verified Worked Examples (Live `AIX-Zero v9` Traces)

| Held-Out Prompt | Full Neuro-Symbolic `<think>` & `<ans>` Trace | Gold | Confidence | Ponder | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Q:5x+1=21,x=` | `Q:5x+1=21,x=<think>21-1=20;20/5=4</think><ans>4</ans>` | `4` (PASS) | `0.8524` | `1.0` | `961.75ms` |
| `Q:5x+1=16,x=` | `Q:5x+1=16,x=<think>16-1=15;15/5=3</think><ans>3</ans>` | `3` (PASS) | `0.8519` | `1.0` | `186.82ms` |
| `Q:4x+2=18,x=` | `Q:4x+2=18,x=<think>18-2=16;16/4=4</think><ans>4</ans>` | `4` (PASS) | `0.8477` | `1.0` | `191.37ms` |
| `Q:2x+2=12,x=` | `Q:2x+2=12,x=<think>12-2=10;10/2=5</think><ans>5</ans>` | `5` (PASS) | `0.8466` | `1.0` | `181.92ms` |
| `Q:6x+4=34,x=` | `Q:6x+4=34,x=<think>34-4=30;30/6=5</think><ans>5</ans>` | `5` (PASS) | `0.8539` | `1.0` | `184.7ms` |
