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

## 4. Formally Verified Worked Examples (Live `AIX-Zero v11` Traces)

| Held-Out Prompt | Full Neuro-Symbolic `<think>` & `<ans>` Trace | Gold | Confidence | Ponder | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Q:2*2>5?` | `Q:2*2>5?<think>2*2=4;4>5->0</think><ans>0</ans>` | `0` (PASS) | `0.8358` | `1.202` | `124.3ms` |
| `Q:2*4>2?` | `Q:2*4>2?<think>2*4=8;8>2->1</think><ans>1</ans>` | `1` (PASS) | `0.863` | `1.122` | `174.56ms` |
| `Q:2*4>1?` | `Q:2*4>1?<think>2*4=8;8>1->1</think><ans>1</ans>` | `1` (PASS) | `0.8626` | `1.128` | `218.47ms` |
| `Q:2*8>8?` | `Q:2*8>8?<think>2*8=16;16>8->1</think><ans>1</ans>` | `1` (PASS) | `0.8254` | `1.303` | `145.8ms` |
| `Q:2*7>7?` | `Q:2*7>7?<think>2*7=14;14>7->1</think><ans>1</ans>` | `1` (PASS) | `0.8417` | `1.366` | `178.07ms` |
