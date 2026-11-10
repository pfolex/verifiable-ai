# Knowledge Codex Entry: `ARITH_CHAIN` — Two-Step Order-of-Operations Arithmetic

- **Category:** Foundational Mathematics
- **Introduced In:** `AIX-Zero v1`
- **Tokenizer Prefix:** `Q:`
- **Governing Law / Formula:** `R = (a * b) +/- c`
- **Expert snapshot:** Consult the versioned cartridge manifest for filename and source expert; specialist quality has not been evaluated.
- **Held-Out Verification Accuracy:** **`100.0%`** (5/5 unseen test problems)

## 1. Domain Concept & Symbolic Grammar

Evaluates precedence-ordered multiplication followed by addition or subtraction.

```text
Prompt Pattern     : Q:{a}*{b}+{c}=
Reasoning Grammar  : <think>{a}*{b}={p};{p}+{c}={r}</think><ans>{r}</ans>
```

## 4. Formally Verified Worked Examples (Live `AIX-Zero v12` Traces)

| Held-Out Prompt | Full Neuro-Symbolic `<think>` & `<ans>` Trace | Gold | Confidence | Ponder | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Q:(4*3)+3=` | `Q:(4*3)+3=<think>4*3=12;12+3=15</think><ans>15</ans>` | `15` (PASS) | `0.8802` | `1.0` | `88.27ms` |
| `Q:(7*7)+1=` | `Q:(7*7)+1=<think>7*7=49;49+1=50</think><ans>50</ans>` | `50` (PASS) | `0.8658` | `1.015` | `79.81ms` |
| `Q:(6*3)+1=` | `Q:(6*3)+1=<think>6*3=18;18+1=19</think><ans>19</ans>` | `19` (PASS) | `0.875` | `1.0` | `83.31ms` |
| `Q:(6*6)+3=` | `Q:(6*6)+3=<think>6*6=36;36+3=39</think><ans>39</ans>` | `39` (PASS) | `0.8499` | `1.0` | `87.13ms` |
| `Q:(4*2)+1=` | `Q:(4*2)+1=<think>4*2=8;8+1=9</think><ans>9</ans>` | `9` (PASS) | `0.8804` | `1.0` | `83.66ms` |
