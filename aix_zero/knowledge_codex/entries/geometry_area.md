# Knowledge Codex Entry: `GEOMETRY_AREA` — 2D Rectangular Area with Boundary Padding

- **Category:** Euclidean Geometry
- **Introduced In:** `AIX-Zero v7`
- **Tokenizer Prefix:** `Q:Geom:`
- **Governing Law / Formula:** `A_total = (w * h) + p`
- **Expert snapshot:** Consult the versioned cartridge manifest for filename and source expert; specialist quality has not been evaluated.
- **Held-Out Verification Accuracy:** **`100.0%`** (5/5 unseen test problems)

## 1. Domain Concept & Symbolic Grammar

Computes base rectangular area w*h and adds boundary padding p.

```text
Prompt Pattern     : Q:Geom:w={w},h={h},p={p},A=
Reasoning Grammar  : <think>{w}*{h}={area};{area}+{p}={tot}</think><ans>{tot}</ans>
```

## 4. Formally Verified Worked Examples (Live `AIX-Zero v12` Traces)

| Held-Out Prompt | Full Neuro-Symbolic `<think>` & `<ans>` Trace | Gold | Confidence | Ponder | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Q:Geom:w=2,h=2,p=5,A=` | `Q:Geom:w=2,h=2,p=5,A=<think>2*2=4;4+5=9</think><ans>9</ans>` | `9` (PASS) | `0.8598` | `1.0` | `88.63ms` |
| `Q:Geom:w=2,h=6,p=5,A=` | `Q:Geom:w=2,h=6,p=5,A=<think>2*6=12;12+5=17</think><ans>17</ans>` | `17` (PASS) | `0.8635` | `1.0` | `87.58ms` |
| `Q:Geom:w=4,h=2,p=1,A=` | `Q:Geom:w=4,h=2,p=1,A=<think>4*2=8;8+1=9</think><ans>9</ans>` | `9` (PASS) | `0.8644` | `1.0` | `81.51ms` |
| `Q:Geom:w=2,h=6,p=3,A=` | `Q:Geom:w=2,h=6,p=3,A=<think>2*6=12;12+3=15</think><ans>15</ans>` | `15` (PASS) | `0.8754` | `1.0` | `90.94ms` |
| `Q:Geom:w=2,h=7,p=1,A=` | `Q:Geom:w=2,h=7,p=1,A=<think>2*7=14;14+1=15</think><ans>15</ans>` | `15` (PASS) | `0.874` | `1.0` | `84.03ms` |
