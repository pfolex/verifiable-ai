# Knowledge Codex Entry: `SYSTEM_2X2` — Simultaneous 2x2 Linear Systems

- **Category:** Linear Algebra
- **Introduced In:** `AIX-Zero v4`
- **Tokenizer Prefix:** `Q:Sys:`
- **Governing Law / Formula:** `{ x + y = S, x - y = D }  ==>  x = (S + D) / 2`
- **Held-Out Verification Accuracy:** **`100.0%`** (5/5 unseen test problems)

## 1. Domain Concept & Symbolic Grammar

Eliminates y by summing simultaneous equations x+y=S and x-y=D, then divides by 2.

```text
Prompt Pattern     : Q:Sys:x+y={s},x-y={d},x=
Reasoning Grammar  : <think>{s}+{d}={sum};{sum}/2={x}</think><ans>{x}</ans>
```

## 4. Formally Verified Worked Examples (Live `AIX-Zero v11` Traces)

| Held-Out Prompt | Full Neuro-Symbolic `<think>` & `<ans>` Trace | Gold | Confidence | Ponder | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Q:Sys:x+y=14,x-y=4,x=` | `Q:Sys:x+y=14,x-y=4,x=<think>14+4=18;18/2=9</think><ans>9</ans>` | `9` (PASS) | `0.8507` | `1.172` | `199.81ms` |
| `Q:Sys:x+y=10,x-y=8,x=` | `Q:Sys:x+y=10,x-y=8,x=<think>10+8=18;18/2=9</think><ans>9</ans>` | `9` (PASS) | `0.842` | `1.165` | `165.72ms` |
| `Q:Sys:x+y=11,x-y=5,x=` | `Q:Sys:x+y=11,x-y=5,x=<think>11+5=16;16/2=8</think><ans>8</ans>` | `8` (PASS) | `0.8564` | `1.134` | `247.11ms` |
| `Q:Sys:x+y=10,x-y=4,x=` | `Q:Sys:x+y=10,x-y=4,x=<think>10+4=14;14/2=7</think><ans>7</ans>` | `7` (PASS) | `0.8604` | `1.166` | `241.15ms` |
| `Q:Sys:x+y=10,x-y=6,x=` | `Q:Sys:x+y=10,x-y=6,x=<think>10+6=16;16/2=8</think><ans>8</ans>` | `8` (PASS) | `0.8539` | `1.165` | `209.88ms` |
