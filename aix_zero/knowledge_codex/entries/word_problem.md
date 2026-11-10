# Knowledge Codex Entry: `WORD_PROBLEM` — Natural-Language Inventory & Counting Word Problems

- **Category:** Applied Mathematics
- **Introduced In:** `AIX-Zero v3`
- **Tokenizer Prefix:** `Q:Word:`
- **Governing Law / Formula:** `Total = (packs * items_per_pack) +/- delta`
- **Expert snapshot:** Consult the versioned cartridge manifest for filename and source expert; specialist quality has not been evaluated.
- **Held-Out Verification Accuracy:** **`100.0%`** (5/5 unseen test problems)

## 1. Domain Concept & Symbolic Grammar

Translates English word narratives into two-step multiplication and accumulation.

```text
Prompt Pattern     : Q:Word:Mia has {a} packs of {b} and buys {c} total=
Reasoning Grammar  : <think>{a}*{b}={p};{p}+{c}={tot}</think><ans>{tot}</ans>
```

## 4. Formally Verified Worked Examples (Live `AIX-Zero v12` Traces)

| Held-Out Prompt | Full Neuro-Symbolic `<think>` & `<ans>` Trace | Gold | Confidence | Ponder | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Q:Word:Sam has 5 packs of 5 and buys 2 total=` | `Q:Word:Sam has 5 packs of 5 and buys 2 total=<think>5*5=25;25+2=27</think><ans>27</ans>` | `27` (PASS) | `0.8353` | `1.0` | `85.62ms` |
| `Q:Word:Mia has 2 packs of 4 and buys 4 total=` | `Q:Word:Mia has 2 packs of 4 and buys 4 total=<think>2*4=8;8+4=12</think><ans>12</ans>` | `12` (PASS) | `0.8745` | `1.0` | `84.17ms` |
| `Q:Word:Mia has 4 packs of 4 and buys 4 total=` | `Q:Word:Mia has 4 packs of 4 and buys 4 total=<think>4*4=16;16+4=20</think><ans>20</ans>` | `20` (PASS) | `0.8341` | `1.0` | `87.92ms` |
| `Q:Word:Mia has 4 packs of 5 and buys 5 total=` | `Q:Word:Mia has 4 packs of 5 and buys 5 total=<think>4*5=20;20+5=25</think><ans>25</ans>` | `25` (PASS) | `0.8233` | `1.0` | `90.14ms` |
| `Q:Word:Mia has 5 packs of 6 and buys 2 total=` | `Q:Word:Mia has 5 packs of 6 and buys 2 total=<think>5*6=30;30+2=32</think><ans>32</ans>` | `32` (PASS) | `0.8422` | `1.0` | `92.52ms` |
