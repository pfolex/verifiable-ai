# Knowledge Codex Entry: `ELECTRICAL_POWER` — Electrical Circuit Net Power & Thermal Loss

- **Category:** Electrical Engineering (Knowledge Is Power v9)
- **Introduced In:** `AIX-Zero v9-KIP`
- **Tokenizer Prefix:** `Q:Elec:`
- **Governing Law / Formula:** `P_net = (V * I) - P_loss`
- **Held-Out Verification Accuracy:** **`100.0%`** (5/5 unseen test problems)

## 1. Domain Concept & Symbolic Grammar

Applies Joule-Watt law P_gross = V*I and subtracts parasitic thermal dissipation loss.

```text
Prompt Pattern     : Q:Elec:V={v},I={i},loss={l},P=
Reasoning Grammar  : <think>{v}*{i}={gross};{gross}-{l}={net}</think><ans>{net}</ans>
```

## 2. Multi-LLM Council Research Notes (`Knowledge Is Power`)

- **Researcher (`aix-8b:latest`):** [Council Synthesis (aix-8b:latest): verified symbolic problem batch (timed out)]
- **Formalizer (`aix3:latest`):** Alright, so I'm trying to figure out how to approach this problem. It's about council synthesis, which is a system that aggregates
- **Recorded At:** `2026-11-04T20:11:50.189746+00:00`

## 3. Formally Verified Worked Examples (Live `AIX-Zero v9` Traces)

| Held-Out Prompt | Full Neuro-Symbolic `<think>` & `<ans>` Trace | Gold | Confidence | Ponder | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Q:Elec:V=7,I=3,loss=1,P=` | `Q:Elec:V=7,I=3,loss=1,P=<think>7*3=21;21-1=20</think><ans>20</ans>` | `20` (PASS) | `0.8425` | `1.0` | `180.72ms` |
| `Q:Elec:V=7,I=2,loss=2,P=` | `Q:Elec:V=7,I=2,loss=2,P=<think>7*2=14;14-2=12</think><ans>12</ans>` | `12` (PASS) | `0.8408` | `1.0` | `154.51ms` |
| `Q:Elec:V=5,I=5,loss=1,P=` | `Q:Elec:V=5,I=5,loss=1,P=<think>5*5=25;25-1=24</think><ans>24</ans>` | `24` (PASS) | `0.8491` | `1.0` | `137.11ms` |
| `Q:Elec:V=5,I=5,loss=2,P=` | `Q:Elec:V=5,I=5,loss=2,P=<think>5*5=25;25-2=23</think><ans>23</ans>` | `23` (PASS) | `0.8444` | `1.0` | `140.61ms` |
| `Q:Elec:V=5,I=4,loss=4,P=` | `Q:Elec:V=5,I=4,loss=4,P=<think>5*4=20;20-4=16</think><ans>16</ans>` | `16` (PASS) | `0.8469` | `1.0` | `137.82ms` |
