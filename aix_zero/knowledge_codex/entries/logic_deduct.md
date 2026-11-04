# Knowledge Codex Entry: `LOGIC_DEDUCT` — Relational Inequality & Boolean Deduction

- **Category:** Formal Logic
- **Introduced In:** `AIX-Zero v1`
- **Tokenizer Prefix:** `Q:`
- **Governing Law / Formula:** `B = 1 if (a * b > c) else 0`
- **Held-Out Verification Accuracy:** **`100.0%`** (5/5 unseen test problems)

## 1. Domain Concept & Symbolic Grammar

Evaluates product a*b and performs strict relational comparison against threshold c.

```text
Prompt Pattern     : Q:{a}*{b}>{c}?
Reasoning Grammar  : <think>{a}*{b}={p};{p}>{c}->{b_val}</think><ans>{b_val}</ans>
```

## 3. Formally Verified Worked Examples (Live `AIX-Zero v9` Traces)

| Held-Out Prompt | Full Neuro-Symbolic `<think>` & `<ans>` Trace | Gold | Confidence | Ponder | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Q:2*2>5?` | `Q:2*2>5?<think>2*2=4;4>5->0</think><ans>0</ans>` | `0` (PASS) | `0.8394` | `1.1` | `182.24ms` |
| `Q:2*4>2?` | `Q:2*4>2?<think>2*4=8;8>2->1</think><ans>1</ans>` | `1` (PASS) | `0.8537` | `1.1` | `146.03ms` |
| `Q:2*4>1?` | `Q:2*4>1?<think>2*4=8;8>1->1</think><ans>1</ans>` | `1` (PASS) | `0.8565` | `1.1` | `188.62ms` |
| `Q:2*8>8?` | `Q:2*8>8?<think>2*8=16;16>8->1</think><ans>1</ans>` | `1` (PASS) | `0.8182` | `1.046` | `160.44ms` |
| `Q:2*7>7?` | `Q:2*7>7?<think>2*7=14;14>7->1</think><ans>1</ans>` | `1` (PASS) | `0.8401` | `1.046` | `173.35ms` |
