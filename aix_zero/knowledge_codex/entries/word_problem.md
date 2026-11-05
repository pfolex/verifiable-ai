# Knowledge Codex Entry: `WORD_PROBLEM` — Natural-Language Inventory & Counting Word Problems

- **Category:** Applied Mathematics
- **Introduced In:** `AIX-Zero v3`
- **Tokenizer Prefix:** `Q:Word:`
- **Governing Law / Formula:** `Total = (packs * items_per_pack) +/- delta`
- **Held-Out Verification Accuracy:** **`100.0%`** (5/5 unseen test problems)

## 1. Domain Concept & Symbolic Grammar

Translates English word narratives into two-step multiplication and accumulation.

```text
Prompt Pattern     : Q:Word:Mia has {a} packs of {b} and buys {c} total=
Reasoning Grammar  : <think>{a}*{b}={p};{p}+{c}={tot}</think><ans>{tot}</ans>
```

## 4. Formally Verified Worked Examples (Live `AIX-Zero v11` Traces)

| Held-Out Prompt | Full Neuro-Symbolic `<think>` & `<ans>` Trace | Gold | Confidence | Ponder | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Q:Word:Sam has 5 packs of 5 and buys 2 total=` | `Q:Word:Sam has 5 packs of 5 and buys 2 total=<think>5*5=25;25+2=27</think><ans>27</ans>` | `27` (PASS) | `0.8226` | `1.003` | `127.68ms` |
| `Q:Word:Mia has 2 packs of 4 and buys 4 total=` | `Q:Word:Mia has 2 packs of 4 and buys 4 total=<think>2*4=8;8+4=12</think><ans>12</ans>` | `12` (PASS) | `0.8564` | `1.0` | `171.39ms` |
| `Q:Word:Mia has 4 packs of 4 and buys 4 total=` | `Q:Word:Mia has 4 packs of 4 and buys 4 total=<think>4*4=16;16+4=20</think><ans>20</ans>` | `20` (PASS) | `0.844` | `1.0` | `183.67ms` |
| `Q:Word:Mia has 4 packs of 5 and buys 5 total=` | `Q:Word:Mia has 4 packs of 5 and buys 5 total=<think>4*5=20;20+5=25</think><ans>25</ans>` | `25` (PASS) | `0.8208` | `1.0` | `248.52ms` |
| `Q:Word:Mia has 5 packs of 6 and buys 2 total=` | `Q:Word:Mia has 5 packs of 6 and buys 2 total=<think>5*6=30;30+2=32</think><ans>32</ans>` | `32` (PASS) | `0.8323` | `1.0` | `199.34ms` |
