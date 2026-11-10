# Knowledge Codex Entry: `CAUSAL_DO` — Pearl Structural Causal Intervention (do-Calculus)

- **Category:** Symbiotic Superintelligence Triad (Knowledge Is Power v12 + Web + 1T Cartridges)
- **Introduced In:** `AIX-Zero v12-KIP-Symbiont`
- **Tokenizer Prefix:** `Q:Causal:`
- **Governing Law / Formula:** `Y_do = ((a * doX) + Z_med) - C_conf`
- **Expert snapshot:** Consult the versioned cartridge manifest for filename and source expert; specialist quality has not been evaluated.
- **Held-Out Verification Accuracy:** **`100.0%`** (5/5 unseen test problems)

## 1. Domain Concept & Symbolic Grammar

Computes causal intervention outcome Y under do(X) by combining direct causal effect a*doX plus mediator Z minus backdoor confounder C.

```text
Prompt Pattern     : Q:Causal:a={a},doX={x},Z={z},conf={c},Y=
Reasoning Grammar  : <think>{a}*{x}={dir};{dir}+{z}={tot};{tot}-{c}={Y}</think><ans>{Y}</ans>
```

## 2. Live Internet Research Citations (`Wikipedia` + `arXiv`)

- **Wikipedia — [Causal model](https://en.wikipedia.org/wiki/Causal_model):** In metaphysics and statistics, a causal model is a conceptual model that represents the causal mechanisms of a system. Causal models often employ formal causal notation, such as structural equation modeling or causal directed acyclic graphs (DAGs), to describe relationships among variables and to guide inference.
- **arXiv — [Intuitionistic $j$-Do-Calculus in Topos Causal Models](http://arxiv.org/abs/2510.17944v2):** In this paper, we generalize Pearl's do-calculus to an Intuitionistic setting called $j$-stable causal inference inside a topos of sheaves. Our framework is an elaboration of the recently proposed framework of Topos Causal Models (TCMs), where causal interventions are defined as subobjects. We generalize the original s

## 3. Multi-LLM Symbiotic Council Conversation (`Knowledge Is Power`)

- **Turn 1 — Theorist (`aix-8b:latest`):** First, the user is asking about Judea Pearl's do-calculus and specifically to explain the 3-step backdoor-adjusted intervention law: Y_do = ((a * doX) + Z) - conf. From the context
- **Turn 2 — Solver & Critic (`aix3:latest`):** Okay, so I need to figure out what Judea Pearl's do-calculus says about this causal question. The user is asking about Y_do, which should be calculated using the formula they provided
- **Turn 3 — Curriculum Consensus (`aix-8b:latest`):** Okay, let me think through this carefully. The user is asking about distilling a specific causal knowledge chunk (Q:Causal:a=3,doX=6,Z=5,conf
- **Turn 4 — `AIX-Zero v12` Student Verification:** `Q:Causal:a=3,doX=6,Z=5,conf=4,Y=<think>3*6=18;18+5=23;23-4=19</think><ans>19</ans>` (`PASS`)
- **Recorded At:** `2026-11-10T21:23:19.788913+00:00`

## 4. Formally Verified Worked Examples (Live `AIX-Zero v12` Traces)

| Held-Out Prompt | Full Neuro-Symbolic `<think>` & `<ans>` Trace | Gold | Confidence | Ponder | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Q:Causal:a=4,doX=2,Z=4,conf=3,Y=` | `Q:Causal:a=4,doX=2,Z=4,conf=3,Y=<think>4*2=8;8+4=12;12-3=9</think><ans>9</ans>` | `9` (PASS) | `0.8421` | `1.023` | `150.24ms` |
| `Q:Causal:a=4,doX=3,Z=4,conf=3,Y=` | `Q:Causal:a=4,doX=3,Z=4,conf=3,Y=<think>4*3=12;12+4=16;16-3=13</think><ans>13</ans>` | `13` (PASS) | `0.8486` | `1.022` | `155.12ms` |
| `Q:Causal:a=4,doX=5,Z=4,conf=1,Y=` | `Q:Causal:a=4,doX=5,Z=4,conf=1,Y=<think>4*5=20;20+4=24;24-1=23</think><ans>23</ans>` | `23` (PASS) | `0.8527` | `1.022` | `428.78ms` |
| `Q:Causal:a=6,doX=5,Z=7,conf=1,Y=` | `Q:Causal:a=6,doX=5,Z=7,conf=1,Y=<think>6*5=30;30+7=37;37-1=36</think><ans>36</ans>` | `36` (PASS) | `0.8322` | `1.022` | `254.31ms` |
| `Q:Causal:a=3,doX=4,Z=2,conf=1,Y=` | `Q:Causal:a=3,doX=4,Z=2,conf=1,Y=<think>3*4=12;12+2=14;14-1=13</think><ans>13</ans>` | `13` (PASS) | `0.8502` | `1.017` | `260.58ms` |
