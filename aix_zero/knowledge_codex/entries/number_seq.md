# Knowledge Codex Entry: `NUMBER_SEQ` — Arithmetic Progression n-th Term

- **Category:** Discrete Mathematics
- **Introduced In:** `AIX-Zero v7`
- **Tokenizer Prefix:** `Q:Seq:`
- **Governing Law / Formula:** `a_n = a_0 + d * n`
- **Expert snapshot:** Consult the versioned cartridge manifest for filename and source expert; specialist quality has not been evaluated.
- **Held-Out Verification Accuracy:** **`100.0%`** (5/5 unseen test problems)

## 1. Domain Concept & Symbolic Grammar

Projects n-th term of an arithmetic sequence from step difference d, index n, and base term a0.

```text
Prompt Pattern     : Q:Seq:d={d},n={n},a0={a0},an=
Reasoning Grammar  : <think>{d}*{n}={step};{step}+{a0}={an}</think><ans>{an}</ans>
```

## 4. Formally Verified Worked Examples (Live `AIX-Zero v12` Traces)

| Held-Out Prompt | Full Neuro-Symbolic `<think>` & `<ans>` Trace | Gold | Confidence | Ponder | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Q:Seq:d=2,n=6,a0=3,an=` | `Q:Seq:d=2,n=6,a0=3,an=<think>2*6=12;12+3=15</think><ans>15</ans>` | `15` (PASS) | `0.8657` | `1.073` | `89.73ms` |
| `Q:Seq:d=4,n=4,a0=1,an=` | `Q:Seq:d=4,n=4,a0=1,an=<think>4*4=16;16+1=17</think><ans>17</ans>` | `17` (PASS) | `0.8505` | `1.074` | `89.23ms` |
| `Q:Seq:d=2,n=4,a0=4,an=` | `Q:Seq:d=2,n=4,a0=4,an=<think>2*4=8;8+4=12</think><ans>12</ans>` | `12` (PASS) | `0.8657` | `1.078` | `82.94ms` |
| `Q:Seq:d=5,n=3,a0=3,an=` | `Q:Seq:d=5,n=3,a0=3,an=<think>5*3=15;15+3=18</think><ans>18</ans>` | `18` (PASS) | `0.8503` | `1.074` | `96.61ms` |
| `Q:Seq:d=2,n=3,a0=1,an=` | `Q:Seq:d=2,n=3,a0=1,an=<think>2*3=6;6+1=7</think><ans>7</ans>` | `7` (PASS) | `0.8464` | `1.079` | `86.32ms` |
