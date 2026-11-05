# Knowledge Codex Entry: `COMBINATORICS_GRID` — Restricted Combinatorial Pairing Space

- **Category:** Combinatorics & Discrete Math (Knowledge Is Power v10)
- **Introduced In:** `AIX-Zero v10-KIP`
- **Tokenizer Prefix:** `Q:Comb:`
- **Governing Law / Formula:** `N_valid = (n * m) - k_excluded`
- **Held-Out Verification Accuracy:** **`100.0%`** (5/5 unseen test problems)

## 1. Domain Concept & Symbolic Grammar

Calculates Cartesian product size n*m and subtracts forbidden/excluded configurations.

```text
Prompt Pattern     : Q:Comb:n={n},m={m},ex={ex},N=
Reasoning Grammar  : <think>{n}*{m}={tot};{tot}-{ex}={valid}</think><ans>{valid}</ans>
```

## 3. Multi-LLM Symbiotic Council Conversation (`Knowledge Is Power`)

- **Turn 1 — Theorist (`aix-8b:latest`):** We are given: n = 5, m = 6, and k_excluded is the number of excluded pairs. The formula for restricted combinatorial pairings is: N_valid = (n * m) - k_excluded This means
- **Turn 2 — Solver & Critic (`aix3:latest`):** Okay, so I need to figure out how to solve this problem where they're talking about combinatorial pairings with some restrictions. The formula given is N_valid = (n * m) - k_excluded
- **Turn 3 — Curriculum Consensus (`aix-8b:latest`):** First, the user is asking me to state why distilling that specific example (n=5, m=6, ex=4) into AIX-Zero's weights teaches it the concept of
- **Turn 4 — `AIX-Zero v11` Student Verification:** `Q:Comb:n=5,m=6,ex=4,N=<think>5*6=30;30-4=26</think><ans>26</ans>` (`PASS`)
- **Recorded At:** `2026-11-04T21:15:30.761473+00:00`

## 4. Formally Verified Worked Examples (Live `AIX-Zero v11` Traces)

| Held-Out Prompt | Full Neuro-Symbolic `<think>` & `<ans>` Trace | Gold | Confidence | Ponder | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Q:Comb:n=7,m=3,ex=1,N=` | `Q:Comb:n=7,m=3,ex=1,N=<think>7*3=21;21-1=20</think><ans>20</ans>` | `20` (PASS) | `0.8485` | `1.11` | `151.11ms` |
| `Q:Comb:n=6,m=2,ex=2,N=` | `Q:Comb:n=6,m=2,ex=2,N=<think>6*2=12;12-2=10</think><ans>10</ans>` | `10` (PASS) | `0.8581` | `1.11` | `159.88ms` |
| `Q:Comb:n=4,m=6,ex=4,N=` | `Q:Comb:n=4,m=6,ex=4,N=<think>4*6=24;24-4=20</think><ans>20</ans>` | `20` (PASS) | `0.8609` | `1.11` | `157.14ms` |
| `Q:Comb:n=4,m=3,ex=4,N=` | `Q:Comb:n=4,m=3,ex=4,N=<think>4*3=12;12-4=8</think><ans>8</ans>` | `8` (PASS) | `0.8499` | `1.11` | `182.97ms` |
| `Q:Comb:n=6,m=3,ex=1,N=` | `Q:Comb:n=6,m=3,ex=1,N=<think>6*3=18;18-1=17</think><ans>17</ans>` | `17` (PASS) | `0.8444` | `1.11` | `183.23ms` |
