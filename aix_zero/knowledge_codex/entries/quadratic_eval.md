# Knowledge Codex Entry: `QUADRATIC_EVAL` — Non-Linear Quadratic Polynomial Evaluation

- **Category:** Polynomial Algebra
- **Introduced In:** `AIX-Zero v6`
- **Tokenizer Prefix:** `Q:Quad:`
- **Governing Law / Formula:** `f(k) = a * k^2 + c = (a * k * k) + c`
- **Held-Out Verification Accuracy:** **`100.0%`** (5/5 unseen test problems)

## 1. Domain Concept & Symbolic Grammar

Computes ternary quadratic term a*k*k and adds constant intercept c.

```text
Prompt Pattern     : Q:Quad:{a}x^2+{c}@x={k}=
Reasoning Grammar  : <think>{a}*{k}*{k}={t1};{t1}+{c}={ans}</think><ans>{ans}</ans>
```

## 4. Formally Verified Worked Examples (Live `AIX-Zero v11` Traces)

| Held-Out Prompt | Full Neuro-Symbolic `<think>` & `<ans>` Trace | Gold | Confidence | Ponder | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Q:Quad:2x^2+1@x=5=` | `Q:Quad:2x^2+1@x=5=<think>2*5*5=50;50+1=51</think><ans>51</ans>` | `51` (PASS) | `0.8416` | `1.0` | `194.24ms` |
| `Q:Quad:3x^2+3@x=3=` | `Q:Quad:3x^2+3@x=3=<think>3*3*3=27;27+3=30</think><ans>30</ans>` | `30` (PASS) | `0.8403` | `1.0` | `176.99ms` |
| `Q:Quad:3x^2+1@x=5=` | `Q:Quad:3x^2+1@x=5=<think>3*5*5=75;75+1=76</think><ans>76</ans>` | `76` (PASS) | `0.8299` | `1.0` | `193.16ms` |
| `Q:Quad:4x^2+1@x=4=` | `Q:Quad:4x^2+1@x=4=<think>4*4*4=64;64+1=65</think><ans>65</ans>` | `65` (PASS) | `0.8503` | `1.0` | `200.27ms` |
| `Q:Quad:4x^2+5@x=4=` | `Q:Quad:4x^2+5@x=4=<think>4*4*4=64;64+5=69</think><ans>69</ans>` | `69` (PASS) | `0.834` | `1.0` | `198.1ms` |
