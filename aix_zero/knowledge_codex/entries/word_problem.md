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

## 3. Formally Verified Worked Examples (Live `AIX-Zero v9` Traces)

| Held-Out Prompt | Full Neuro-Symbolic `<think>` & `<ans>` Trace | Gold | Confidence | Ponder | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Q:Word:Sam has 5 packs of 5 and buys 2 total=` | `Q:Word:Sam has 5 packs of 5 and buys 2 total=<think>5*5=25;25+2=27</think><ans>27</ans>` | `27` (PASS) | `0.8249` | `1.0` | `151.95ms` |
| `Q:Word:Mia has 2 packs of 4 and buys 4 total=` | `Q:Word:Mia has 2 packs of 4 and buys 4 total=<think>2*4=8;8+4=12</think><ans>12</ans>` | `12` (PASS) | `0.8321` | `1.0` | `180.53ms` |
| `Q:Word:Mia has 4 packs of 4 and buys 4 total=` | `Q:Word:Mia has 4 packs of 4 and buys 4 total=<think>4*4=16;16+4=20</think><ans>20</ans>` | `20` (PASS) | `0.8297` | `1.0` | `182.33ms` |
| `Q:Word:Mia has 4 packs of 5 and buys 5 total=` | `Q:Word:Mia has 4 packs of 5 and buys 5 total=<think>4*5=20;20+5=25</think><ans>25</ans>` | `25` (PASS) | `0.8292` | `1.0` | `156.48ms` |
| `Q:Word:Mia has 5 packs of 6 and buys 2 total=` | `Q:Word:Mia has 5 packs of 6 and buys 2 total=<think>5*6=30;30+2=32</think><ans>32</ans>` | `32` (PASS) | `0.8286` | `1.0` | `184.33ms` |
