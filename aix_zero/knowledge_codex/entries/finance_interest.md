# Knowledge Codex Entry: `FINANCE_INTEREST` — Financial Yield & Principal Accumulation

- **Category:** Quantitative Finance (Knowledge Is Power v9)
- **Introduced In:** `AIX-Zero v9-KIP`
- **Tokenizer Prefix:** `Q:Fin:`
- **Governing Law / Formula:** `B = (r * t) + P_0`
- **Expert snapshot:** Consult the versioned cartridge manifest for filename and source expert; specialist quality has not been evaluated.
- **Held-Out Verification Accuracy:** **`100.0%`** (5/5 unseen test problems)

## 1. Domain Concept & Symbolic Grammar

Calculates cumulative yield r*t over t periods and adds initial principal P_0.

```text
Prompt Pattern     : Q:Fin:r={r},t={t},P={p},B=
Reasoning Grammar  : <think>{r}*{t}={yield};{yield}+{p}={bal}</think><ans>{bal}</ans>
```

## 3. Multi-LLM Symbiotic Council Conversation (`Knowledge Is Power`)

- **Turn 1 — Theorist (`aix-8b:latest`):** We are given a formula for linear financial yield accumulation: B = (r * t) + P_0. This can be interpreted as the total amount B after time t at a constant interest rate r per unit
- **Turn 2 — Solver & Critic (`aix3:latest`):** Okay, so I'm trying to figure out this problem about linear financial yield accumulation. The formula given is B = (r * t) + P_0. Let me break it down step by step. First, let's identify
- **Turn 3 — Curriculum Consensus (`aix-8b:latest`):** Okay, let me analyze this request. The user wants a concise explanation about why distilling the specific problem instance (Q:Fin:r=5,t=6,P=
- **Turn 4 — `AIX-Zero v12` Student Verification:** `Q:Fin:r=5,t=6,P=8,B=<think>5*6=30;30+8=38</think><ans>38</ans>` (`PASS`)
- **Recorded At:** `2026-11-04T20:32:28.370172+00:00`

## 4. Formally Verified Worked Examples (Live `AIX-Zero v12` Traces)

| Held-Out Prompt | Full Neuro-Symbolic `<think>` & `<ans>` Trace | Gold | Confidence | Ponder | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Q:Fin:r=5,t=6,P=8,B=` | `Q:Fin:r=5,t=6,P=8,B=<think>5*6=30;30+8=38</think><ans>38</ans>` | `38` (PASS) | `0.8409` | `1.0` | `81.92ms` |
| `Q:Fin:r=4,t=3,P=8,B=` | `Q:Fin:r=4,t=3,P=8,B=<think>4*3=12;12+8=20</think><ans>20</ans>` | `20` (PASS) | `0.8717` | `1.0` | `84.16ms` |
| `Q:Fin:r=5,t=6,P=10,B=` | `Q:Fin:r=5,t=6,P=10,B=<think>5*6=30;30+10=40</think><ans>40</ans>` | `40` (PASS) | `0.8564` | `1.0` | `95.02ms` |
| `Q:Fin:r=2,t=2,P=5,B=` | `Q:Fin:r=2,t=2,P=5,B=<think>2*2=4;4+5=9</think><ans>9</ans>` | `9` (PASS) | `0.8619` | `1.0` | `87.55ms` |
| `Q:Fin:r=5,t=5,P=8,B=` | `Q:Fin:r=5,t=5,P=8,B=<think>5*5=25;25+8=33</think><ans>33</ans>` | `33` (PASS) | `0.8413` | `1.0` | `88.85ms` |
