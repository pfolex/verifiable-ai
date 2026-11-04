# Knowledge Codex Entry: `MODULAR_NT` — Modular Number Theory & Congruence

- **Category:** Number Theory
- **Introduced In:** `AIX-Zero v1`
- **Tokenizer Prefix:** `Q:(`
- **Governing Law / Formula:** `R = (a + b) mod m`
- **Held-Out Verification Accuracy:** **`100.0%`** (5/5 unseen test problems)

## 1. Domain Concept & Symbolic Grammar

Computes parenthetical sum followed by Euclidean remainder modulo m.

```text
Prompt Pattern     : Q:({a}+{b})%{m}=
Reasoning Grammar  : <think>{a}+{b}={s};{s}%{m}={r}</think><ans>{r}</ans>
```

## 3. Formally Verified Worked Examples (Live `AIX-Zero v9` Traces)

| Held-Out Prompt | Full Neuro-Symbolic `<think>` & `<ans>` Trace | Gold | Confidence | Ponder | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Q:(9+7)%3=` | `Q:(9+7)%3=<think>9+7=16;16%3=1</think><ans>1</ans>` | `1` (PASS) | `0.8276` | `1.05` | `131.35ms` |
| `Q:(7+5)%4=` | `Q:(7+5)%4=<think>7+5=12;12%4=0</think><ans>0</ans>` | `0` (PASS) | `0.8242` | `1.033` | `148.94ms` |
| `Q:(8+5)%4=` | `Q:(8+5)%4=<think>8+5=13;13%4=1</think><ans>1</ans>` | `1` (PASS) | `0.8312` | `1.034` | `142.67ms` |
| `Q:(11+2)%3=` | `Q:(11+2)%3=<think>11+2=13;13%3=1</think><ans>1</ans>` | `1` (PASS) | `0.8566` | `1.033` | `175.91ms` |
| `Q:(5+7)%4=` | `Q:(5+7)%4=<think>5+7=12;12%4=0</think><ans>0</ans>` | `0` (PASS) | `0.8265` | `1.055` | `151.93ms` |
