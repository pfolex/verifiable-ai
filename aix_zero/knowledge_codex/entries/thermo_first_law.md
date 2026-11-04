# Knowledge Codex Entry: `THERMO_FIRST_LAW` — First Law of Thermodynamics & Internal Energy

- **Category:** Thermodynamics (Knowledge Is Power v10)
- **Introduced In:** `AIX-Zero v10-KIP`
- **Tokenizer Prefix:** `Q:Thermo:`
- **Governing Law / Formula:** `dU = (q * t) - W`
- **Held-Out Verification Accuracy:** **`100.0%`** (5/5 unseen test problems)

## 1. Domain Concept & Symbolic Grammar

Computes total heat influx Q = q*t and subtracts mechanical work W performed by the system.

```text
Prompt Pattern     : Q:Thermo:q={q},t={t},W={w},dU=
Reasoning Grammar  : <think>{q}*{t}={Q};{Q}-{w}={dU}</think><ans>{dU}</ans>
```

## 2. Multi-LLM Symbiotic Council Conversation (`Knowledge Is Power`)

- **Turn 1 — Theorist (`aix-8b:latest`):** We are given an equation: dU = (q * T) - W However, note that the standard first law is usually written as ΔU = Q - W, where: U is internal energy, Q is heat transferred to
- **Turn 2 — Solver & Critic (`aix3:latest`):** Alright, so I've got this problem here about calculating something related to thermodynamics. Let me see if I can wrap my head around it. The user mentioned an equation: dU = (q *
- **Turn 3 — Curriculum Consensus (`aix-8b:latest`):** First, the user is referencing a Solver's verification where they were trying to calculate something related to thermodynamics. The equation
- **Turn 4 — `AIX-Zero v9` Student Verification:** `Q:Thermo:q=6,t=4,W=3,dU=<think>6*4=24;24-3=21</think><ans>21</ans>` (`PASS`)
- **Recorded At:** `2026-11-04T21:14:29.074758+00:00`

## 3. Formally Verified Worked Examples (Live `AIX-Zero v9` Traces)

| Held-Out Prompt | Full Neuro-Symbolic `<think>` & `<ans>` Trace | Gold | Confidence | Ponder | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Q:Thermo:q=6,t=3,W=1,dU=` | `Q:Thermo:q=6,t=3,W=1,dU=<think>6*3=18;18-1=17</think><ans>17</ans>` | `17` (PASS) | `0.8427` | `1.001` | `109.11ms` |
| `Q:Thermo:q=7,t=4,W=1,dU=` | `Q:Thermo:q=7,t=4,W=1,dU=<think>7*4=28;28-1=27</think><ans>27</ans>` | `27` (PASS) | `0.8517` | `1.002` | `154.36ms` |
| `Q:Thermo:q=7,t=2,W=3,dU=` | `Q:Thermo:q=7,t=2,W=3,dU=<think>7*2=14;14-3=11</think><ans>11</ans>` | `11` (PASS) | `0.8583` | `1.002` | `177.08ms` |
| `Q:Thermo:q=3,t=5,W=1,dU=` | `Q:Thermo:q=3,t=5,W=1,dU=<think>3*5=15;15-1=14</think><ans>14</ans>` | `14` (PASS) | `0.8387` | `1.006` | `126.53ms` |
| `Q:Thermo:q=3,t=3,W=4,dU=` | `Q:Thermo:q=3,t=3,W=4,dU=<think>3*3=9;9-4=5</think><ans>5</ans>` | `5` (PASS) | `0.8442` | `1.006` | `105.92ms` |
