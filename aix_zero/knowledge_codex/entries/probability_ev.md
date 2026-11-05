# Knowledge Codex Entry: `PROBABILITY_EV` — Three-Step Discrete Expected Value with Bonus & Fee

- **Category:** Probability & Decision Theory (Knowledge Is Power v10)
- **Introduced In:** `AIX-Zero v10-KIP`
- **Tokenizer Prefix:** `Q:Prob:`
- **Governing Law / Formula:** `EV = ((p * v) + bonus) - fee`
- **Held-Out Verification Accuracy:** **`100.0%`** (5/5 unseen test problems)

## 1. Domain Concept & Symbolic Grammar

Executes a 3-step decision chain: base payoff p*v, plus bonus b, minus entry fee f.

```text
Prompt Pattern     : Q:Prob:p={p},v={v},bonus={b},fee={f},EV=
Reasoning Grammar  : <think>{p}*{v}={g};{g}+{b}={tot};{tot}-{f}={ev}</think><ans>{ev}</ans>
```

## 3. Multi-LLM Symbiotic Council Conversation (`Knowledge Is Power`)

- **Turn 1 — Theorist (`aix-8b:latest`):** First, I'm addressing a user query about explaining an economic or probabilistic concept using specific values: probability (p) is 3, value (v) is 5, bonus is 4, and fee is 2. The
- **Turn 2 — Solver & Critic (`aix3:latest`):** Alright, let's tackle this query about calculating the Expected Value (EV) of a deal based on some given values. The user is asking me to explain how to compute EV using specific numbers
- **Turn 3 — Curriculum Consensus (`aix-8b:latest`):** Okay, let me think about this problem. The user is asking how to calculate Expected Value (EV) for a deal with specific parameters: probability
- **Turn 4 — `AIX-Zero v11` Student Verification:** `Q:Prob:p=3,v=5,bonus=4,fee=2,EV=<think>3*5=15;15+4=19;19-2=17</think><ans>17</ans>` (`PASS`)
- **Recorded At:** `2026-11-04T21:16:26.056924+00:00`

## 4. Formally Verified Worked Examples (Live `AIX-Zero v11` Traces)

| Held-Out Prompt | Full Neuro-Symbolic `<think>` & `<ans>` Trace | Gold | Confidence | Ponder | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Q:Prob:p=5,v=2,bonus=5,fee=2,EV=` | `Q:Prob:p=5,v=2,bonus=5,fee=2,EV=<think>5*2=10;10+5=15;15-2=13</think><ans>13</ans>` | `13` (PASS) | `0.8471` | `1.0` | `198.28ms` |
| `Q:Prob:p=3,v=3,bonus=3,fee=2,EV=` | `Q:Prob:p=3,v=3,bonus=3,fee=2,EV=<think>3*3=9;9+3=12;12-2=10</think><ans>10</ans>` | `10` (PASS) | `0.8396` | `1.0` | `217.01ms` |
| `Q:Prob:p=4,v=2,bonus=5,fee=1,EV=` | `Q:Prob:p=4,v=2,bonus=5,fee=1,EV=<think>4*2=8;8+5=13;13-1=12</think><ans>12</ans>` | `12` (PASS) | `0.8524` | `1.0` | `221.81ms` |
| `Q:Prob:p=4,v=4,bonus=3,fee=2,EV=` | `Q:Prob:p=4,v=4,bonus=3,fee=2,EV=<think>4*4=16;16+3=19;19-2=17</think><ans>17</ans>` | `17` (PASS) | `0.849` | `1.0` | `232.67ms` |
| `Q:Prob:p=2,v=3,bonus=3,fee=1,EV=` | `Q:Prob:p=2,v=3,bonus=3,fee=1,EV=<think>2*3=6;6+3=9;9-1=8</think><ans>8</ans>` | `8` (PASS) | `0.832` | `1.0` | `218.94ms` |
