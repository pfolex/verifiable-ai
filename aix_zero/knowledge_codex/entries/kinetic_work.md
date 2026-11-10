# Knowledge Codex Entry: `KINETIC_WORK` — Cross-Domain Mechanical Work-Energy & Friction Conservation

- **Category:** Multi-Physics Cross-Domain Synthesis (Knowledge Is Power v11 + Web)
- **Introduced In:** `AIX-Zero v11-KIP-Web`
- **Tokenizer Prefix:** `Q:BridgeKW:`
- **Governing Law / Formula:** `E_final = ((F * d) + K_0) - L_friction`
- **Expert snapshot:** Consult the versioned cartridge manifest for filename and source expert; specialist quality has not been evaluated.
- **Held-Out Verification Accuracy:** **`100.0%`** (5/5 unseen test problems)

## 1. Domain Concept & Symbolic Grammar

Fuses mechanical work W=F*d with initial kinetic energy K0 and dissipative friction loss L.

```text
Prompt Pattern     : Q:BridgeKW:F={f},d={d},K0={k},loss={l},E=
Reasoning Grammar  : <think>{f}*{d}={W};{W}+{k}={Etot};{Etot}-{l}={E}</think><ans>{E}</ans>
```

## 2. Live Internet Research Citations (`Wikipedia` + `arXiv`)

- **Wikipedia — [Work (physics)](https://en.wikipedia.org/wiki/Work_(physics)):** In science, work is the energy transferred to or from an object via the application of force along a displacement. In its simplest form, for a constant force aligned with the direction of motion, the work equals the product of the force strength and the distance traveled. A force is said to do positive work if it has a component in the direction of the displ
- **arXiv — [Large Synoptic Survey Telescope: Dark Energy Science Collaboration](http://arxiv.org/abs/1211.0310v1):** This white paper describes the LSST Dark Energy Science Collaboration (DESC), whose goal is the study of dark energy and related topics in fundamental physics with data from the Large Synoptic Survey Telescope (LSST). It provides an overview of dark energy science and describes the current and anticipated state of the 

## 3. Multi-LLM Symbiotic Council Conversation (`Knowledge Is Power`)

- **Turn 1 — Theorist (`aix-8b:latest`):** First, the user is referencing two sources: Wikipedia for a definition of work in physics, and arXiv for something about the Large Synoptic Survey Telescope (LSST), but it's not directly
- **Turn 2 — Solver & Critic (`aix3:latest`):** Alright, so I need to figure out how to solve this problem about calculating work done on a bridge using some given parameters. The user mentioned two sources: Wikipedia for physics
- **Turn 3 — Curriculum Consensus (`aix-8b:latest`):** Okay, let me think about this problem carefully. The user is asking how to calculate work done on a bridge using given parameters: F=5 (force
- **Turn 4 — `AIX-Zero v12` Student Verification:** `Q:BridgeKW:F=5,d=4,K0=7,loss=3,E=<think>5*4=20;20+7=27;27-3=24</think><ans>24</ans>` (`PASS`)
- **Recorded At:** `2026-11-05T07:25:31.098884+00:00`

## 4. Formally Verified Worked Examples (Live `AIX-Zero v12` Traces)

| Held-Out Prompt | Full Neuro-Symbolic `<think>` & `<ans>` Trace | Gold | Confidence | Ponder | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Q:BridgeKW:F=4,d=3,K0=5,loss=2,E=` | `Q:BridgeKW:F=4,d=3,K0=5,loss=2,E=<think>4*3=12;12+5=17;17-2=15</think><ans>15</ans>` | `15` (PASS) | `0.8418` | `1.0` | `244.2ms` |
| `Q:BridgeKW:F=4,d=2,K0=5,loss=1,E=` | `Q:BridgeKW:F=4,d=2,K0=5,loss=1,E=<think>4*2=8;8+5=13;13-1=12</think><ans>12</ans>` | `12` (PASS) | `0.868` | `1.0` | `243.78ms` |
| `Q:BridgeKW:F=2,d=4,K0=7,loss=2,E=` | `Q:BridgeKW:F=2,d=4,K0=7,loss=2,E=<think>2*4=8;8+7=15;15-2=13</think><ans>13</ans>` | `13` (PASS) | `0.8528` | `1.0` | `242.44ms` |
| `Q:BridgeKW:F=4,d=2,K0=3,loss=2,E=` | `Q:BridgeKW:F=4,d=2,K0=3,loss=2,E=<think>4*2=8;8+3=11;11-2=9</think><ans>9</ans>` | `9` (PASS) | `0.8573` | `1.0` | `252.3ms` |
| `Q:BridgeKW:F=5,d=4,K0=3,loss=2,E=` | `Q:BridgeKW:F=5,d=4,K0=3,loss=2,E=<think>5*4=20;20+3=23;23-2=21</think><ans>21</ans>` | `21` (PASS) | `0.857` | `1.0` | `235.52ms` |
