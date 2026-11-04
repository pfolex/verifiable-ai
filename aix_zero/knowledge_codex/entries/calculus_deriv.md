# Knowledge Codex Entry: `CALCULUS_DERIV` — Symbolic Polynomial Differentiation & Point Evaluation

- **Category:** Differential Calculus
- **Introduced In:** `AIX-Zero v3`
- **Tokenizer Prefix:** `Q:Calc:`
- **Governing Law / Formula:** `d/dx(a*x^2 + b*x) |_{x=k} = 2*a*k + b`
- **Held-Out Verification Accuracy:** **`100.0%`** (5/5 unseen test problems)

## 1. Domain Concept & Symbolic Grammar

Applies power rule d/dx(a*x^2 + b*x) = 2*a*x + b and evaluates at x = k.

```text
Prompt Pattern     : Q:Calc:d/dx({a}x^2+{b}x)@x={k}=
Reasoning Grammar  : <think>{a}*2*{k}={t1};{t1}+{b}={ans}</think><ans>{ans}</ans>
```

## 3. Formally Verified Worked Examples (Live `AIX-Zero v9` Traces)

| Held-Out Prompt | Full Neuro-Symbolic `<think>` & `<ans>` Trace | Gold | Confidence | Ponder | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Q:Calc:d/dx(5x^2+1x)@x=1=` | `Q:Calc:d/dx(5x^2+1x)@x=1=<think>5*2*1=10;10+1=11</think><ans>11</ans>` | `11` (PASS) | `0.8456` | `1.0` | `197.43ms` |
| `Q:Calc:d/dx(2x^2+7x)@x=3=` | `Q:Calc:d/dx(2x^2+7x)@x=3=<think>2*2*3=12;12+7=19</think><ans>19</ans>` | `19` (PASS) | `0.8444` | `1.0` | `148.18ms` |
| `Q:Calc:d/dx(5x^2+5x)@x=4=` | `Q:Calc:d/dx(5x^2+5x)@x=4=<think>5*2*4=40;40+5=45</think><ans>45</ans>` | `45` (PASS) | `0.8322` | `1.0` | `210.72ms` |
| `Q:Calc:d/dx(4x^2+5x)@x=2=` | `Q:Calc:d/dx(4x^2+5x)@x=2=<think>4*2*2=16;16+5=21</think><ans>21</ans>` | `21` (PASS) | `0.8404` | `1.0` | `226.5ms` |
| `Q:Calc:d/dx(3x^2+5x)@x=1=` | `Q:Calc:d/dx(3x^2+5x)@x=1=<think>3*2*1=6;6+5=11</think><ans>11</ans>` | `11` (PASS) | `0.8323` | `1.0` | `214.11ms` |
