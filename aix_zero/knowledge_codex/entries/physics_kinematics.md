# Knowledge Codex Entry: `PHYSICS_KINEMATICS` — 1D Constant-Acceleration Kinematics

- **Category:** Classical Mechanics
- **Introduced In:** `AIX-Zero v6`
- **Tokenizer Prefix:** `Q:Phys:`
- **Governing Law / Formula:** `v = u + a * t`
- **Held-Out Verification Accuracy:** **`100.0%`** (5/5 unseen test problems)

## 1. Domain Concept & Symbolic Grammar

Computes velocity change dv = a*t and adds initial velocity u.

```text
Prompt Pattern     : Q:Phys:a={a},t={t},u={u},v=
Reasoning Grammar  : <think>{a}*{t}={dv};{dv}+{u}={v}</think><ans>{v}</ans>
```

## 3. Formally Verified Worked Examples (Live `AIX-Zero v9` Traces)

| Held-Out Prompt | Full Neuro-Symbolic `<think>` & `<ans>` Trace | Gold | Confidence | Ponder | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Q:Phys:a=4,t=5,u=1,v=` | `Q:Phys:a=4,t=5,u=1,v=<think>4*5=20;20+1=21</think><ans>21</ans>` | `21` (PASS) | `0.8506` | `1.0` | `169.56ms` |
| `Q:Phys:a=5,t=3,u=2,v=` | `Q:Phys:a=5,t=3,u=2,v=<think>5*3=15;15+2=17</think><ans>17</ans>` | `17` (PASS) | `0.8446` | `1.0` | `164.06ms` |
| `Q:Phys:a=3,t=5,u=2,v=` | `Q:Phys:a=3,t=5,u=2,v=<think>3*5=15;15+2=17</think><ans>17</ans>` | `17` (PASS) | `0.8431` | `1.0` | `174.8ms` |
| `Q:Phys:a=3,t=2,u=4,v=` | `Q:Phys:a=3,t=2,u=4,v=<think>3*2=6;6+4=10</think><ans>10</ans>` | `10` (PASS) | `0.843` | `1.0` | `179.26ms` |
| `Q:Phys:a=4,t=6,u=1,v=` | `Q:Phys:a=4,t=6,u=1,v=<think>4*6=24;24+1=25</think><ans>25</ans>` | `25` (PASS) | `0.845` | `1.0` | `176.92ms` |
