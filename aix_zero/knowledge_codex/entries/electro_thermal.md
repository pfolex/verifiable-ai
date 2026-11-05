# Knowledge Codex Entry: `ELECTRO_THERMAL` — Cross-Domain Electro-Thermal Joule Heating & Internal Energy

- **Category:** Multi-Physics Cross-Domain Synthesis (Knowledge Is Power v11 + Web)
- **Introduced In:** `AIX-Zero v11-KIP-Web`
- **Tokenizer Prefix:** `Q:BridgeET:`
- **Governing Law / Formula:** `dU = ((V * I) * t) - W`
- **Held-Out Verification Accuracy:** **`100.0%`** (5/5 unseen test problems)

## 1. Domain Concept & Symbolic Grammar

Fuses Electrical Power (P=V*I) with First Law of Thermodynamics (dU=P*t-W) in a 3-step chain.

```text
Prompt Pattern     : Q:BridgeET:V={v},I={i},t={t},W={w},dU=
Reasoning Grammar  : <think>{v}*{i}={P};{P}*{t}={Q};{Q}-{w}={dU}</think><ans>{dU}</ans>
```

## 2. Live Internet Research Citations (`Wikipedia` + `arXiv`)

- **Wikipedia — [Joule heating](https://en.wikipedia.org/wiki/Joule_heating):** Joule heating is the process by which the passage of an electric current through a conductor produces heat.
- **arXiv — [Finite Element Convergence for the Joule Heating Problem with Mixed Boundary Conditions](http://arxiv.org/abs/1203.6306v1):** We prove strong convergence of conforming finite element approximations to the stationary Joule heating problem with mixed boundary conditions on Lipschitz domains in three spatial dimensions. We show optimal global regularity estimates on creased domains and prove a priori and a posteriori bounds for shape regular mes

## 3. Multi-LLM Symbiotic Council Conversation (`Knowledge Is Power`)

- **Turn 1 — Theorist (`aix-8b:latest`):** We are given: 1. Joule heating: Q = V * I * t (This is a standard formula for energy dissipated as heat in an electrical conductor) 2. The First Law of Thermodynamics: dU = Q - W
- **Turn 2 — Solver & Critic (`aix3:latest`):** Okay, so I need to figure out this problem where they're talking about Joule heating and the First Law of Thermodynamics. Let me start by understanding what each term means. Joule
- **Turn 3 — Curriculum Consensus (`aix-8b:latest`):** Okay, the user is asking me to explain why distilling a specific problem involving Joule heating and thermodynamics into AIX-Zero's weight system
- **Turn 4 — `AIX-Zero v11` Student Verification:** `Q:BridgeET:V=4,I=3,t=5,W=6,dU=<think>4*3=12;12*5=60;60-6=54</think><ans>54</ans>` (`PASS`)
- **Recorded At:** `2026-11-05T07:24:19.973238+00:00`

## 4. Formally Verified Worked Examples (Live `AIX-Zero v11` Traces)

| Held-Out Prompt | Full Neuro-Symbolic `<think>` & `<ans>` Trace | Gold | Confidence | Ponder | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Q:BridgeET:V=5,I=3,t=4,W=3,dU=` | `Q:BridgeET:V=5,I=3,t=4,W=3,dU=<think>5*3=15;15*4=60;60-3=57</think><ans>57</ans>` | `57` (PASS) | `0.835` | `1.009` | `232.36ms` |
| `Q:BridgeET:V=4,I=4,t=2,W=1,dU=` | `Q:BridgeET:V=4,I=4,t=2,W=1,dU=<think>4*4=16;16*2=32;32-1=31</think><ans>31</ans>` | `31` (PASS) | `0.852` | `1.0` | `363.88ms` |
| `Q:BridgeET:V=2,I=2,t=4,W=3,dU=` | `Q:BridgeET:V=2,I=2,t=4,W=3,dU=<think>2*2=4;4*4=16;16-3=13</think><ans>13</ans>` | `13` (PASS) | `0.8621` | `1.0` | `170.39ms` |
| `Q:BridgeET:V=4,I=4,t=3,W=1,dU=` | `Q:BridgeET:V=4,I=4,t=3,W=1,dU=<think>4*4=16;16*3=48;48-1=47</think><ans>47</ans>` | `47` (PASS) | `0.8488` | `1.0` | `165.15ms` |
| `Q:BridgeET:V=2,I=2,t=2,W=3,dU=` | `Q:BridgeET:V=2,I=2,t=2,W=3,dU=<think>2*2=4;4*2=8;8-3=5</think><ans>5</ans>` | `5` (PASS) | `0.8489` | `1.001` | `160.65ms` |
