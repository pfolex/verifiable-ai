# Knowledge Codex Entry: `MULTI_STEP_3` — Three-Step Compositional Reasoning Chains

- **Category:** Multi-Step Composition
- **Introduced In:** `AIX-Zero v8`
- **Tokenizer Prefix:** `Q:Chain3:`
- **Governing Law / Formula:** `R = ((a * b) + c) - d`
- **Held-Out Verification Accuracy:** **`100.0%`** (5/5 unseen test problems)

## 1. Domain Concept & Symbolic Grammar

Chains three sequential operations with double accumulator auto-carry across semicolons.

```text
Prompt Pattern     : Q:Chain3:{a}*{b}+{c}-{d}=
Reasoning Grammar  : <think>{a}*{b}={p};{p}+{c}={q};{q}-{d}={r}</think><ans>{r}</ans>
```

## 3. Formally Verified Worked Examples (Live `AIX-Zero v9` Traces)

| Held-Out Prompt | Full Neuro-Symbolic `<think>` & `<ans>` Trace | Gold | Confidence | Ponder | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Q:Chain3:2*3+5-2=` | `Q:Chain3:2*3+5-2=<think>2*3=6;6+5=11;11-2=9</think><ans>9</ans>` | `9` (PASS) | `0.8358` | `1.0` | `137.94ms` |
| `Q:Chain3:5*4+3-1=` | `Q:Chain3:5*4+3-1=<think>5*4=20;20+3=23;23-1=22</think><ans>22</ans>` | `22` (PASS) | `0.8416` | `1.034` | `134.98ms` |
| `Q:Chain3:5*5+7-2=` | `Q:Chain3:5*5+7-2=<think>5*5=25;25+7=32;32-2=30</think><ans>30</ans>` | `30` (PASS) | `0.8414` | `1.0` | `142.19ms` |
| `Q:Chain3:6*5+3-1=` | `Q:Chain3:6*5+3-1=<think>6*5=30;30+3=33;33-1=32</think><ans>32</ans>` | `32` (PASS) | `0.8408` | `1.0` | `188.79ms` |
| `Q:Chain3:5*5+5-1=` | `Q:Chain3:5*5+5-1=<think>5*5=25;25+5=30;30-1=29</think><ans>29</ans>` | `29` (PASS) | `0.84` | `1.0` | `245.38ms` |
