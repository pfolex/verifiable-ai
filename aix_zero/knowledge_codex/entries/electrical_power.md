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

## 2. Multi-LLM Symbiotic Council Conversation (`Knowledge Is Power`)

- **Turn 1 — Theorist (`aix-8b:latest`):** We are given an equation: P_net = (V * I) - P_loss This can be broken down as: - The net power (P_net) is the product of voltage (V) and current (I), minus any losses (P_loss). Now
- **Turn 2 — Solver & Critic (`aix3:latest`):** Alright, so I need to figure out this equation they gave us about electric power. Let me start by understanding what each part of it means. First, there's P_net, which is the net power
- **Turn 3 — Curriculum Consensus (`aix-8b:latest`):** First, the user is referencing a Solver's verification where they're analyzing an equation for electric power. They mention P_net as net power
- **Turn 4 — `AIX-Zero v9` Student Verification:** `Q:Elec:V=7,I=3,loss=1,P=<think>7*3=21;21-1=20</think><ans>20</ans>` (`PASS`)
- **Recorded At:** `2026-11-04T20:31:27.212409+00:00`

## 3. Formally Verified Worked Examples (Live `AIX-Zero v9` Traces)

| Held-Out Prompt | Full Neuro-Symbolic `<think>` & `<ans>` Trace | Gold | Confidence | Ponder | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Q:Elec:V=7,I=3,loss=1,P=` | `Q:Elec:V=7,I=3,loss=1,P=<think>7*3=21;21-1=20</think><ans>20</ans>` | `20` (PASS) | `0.838` | `1.0` | `159.66ms` |
| `Q:Elec:V=7,I=2,loss=2,P=` | `Q:Elec:V=7,I=2,loss=2,P=<think>7*2=14;14-2=12</think><ans>12</ans>` | `12` (PASS) | `0.8449` | `1.0` | `179.7ms` |
| `Q:Elec:V=5,I=5,loss=1,P=` | `Q:Elec:V=5,I=5,loss=1,P=<think>5*5=25;25-1=24</think><ans>24</ans>` | `24` (PASS) | `0.832` | `1.0` | `118.12ms` |
| `Q:Elec:V=5,I=5,loss=2,P=` | `Q:Elec:V=5,I=5,loss=2,P=<think>5*5=25;25-2=23</think><ans>23</ans>` | `23` (PASS) | `0.8378` | `1.0` | `108.27ms` |
| `Q:Elec:V=5,I=4,loss=4,P=` | `Q:Elec:V=5,I=4,loss=4,P=<think>5*4=20;20-4=16</think><ans>16</ans>` | `16` (PASS) | `0.8444` | `1.0` | `111.15ms` |
