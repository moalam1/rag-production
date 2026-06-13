# RAG Platform Admin Console — Technical & Maintenance Documentation

**Version:** 1.0 · **Date:** 12 Jun 2026 · **System:** Equinix RAG Production (tenant #001)
**Audience:** Tech team (sections 1–5, 8–9) · Maintenance team (sections 6–7)

---

## 1. What this is

The admin console is the **control plane** for the RAG buyer-intelligence platform: it manages
*how* the system behaves (signals, prompts, costs, keys) without touching the serving path
(the **data plane**: search API, retrieval, generation, caches).

Core property: **configuration changes propagate through data, not code.** Editing a tag or a
prompt in the browser writes to DynamoDB; the API picks it up within 5 minutes (config TTL) or
on the next query (prompts). No deploys, no restarts, no engineers required for routine changes.

Failure isolation: if the console or `/admin/*` endpoints are down, **search keeps serving** —
every consumer has a hardcoded code fallback.

| Component | Location |
|---|---|
| Console UI | Private HF Space (static SDK): `huggingface.co/spaces/perwaizalam/rag-admin-console` |
| Backend API | `https://lxhxqqh3r8.execute-api.us-east-1.amazonaws.com` → EC2 `rag-api` (systemd) |
| Config store | DynamoDB table `rag-config` (us-east-1) |
| Code | EC2 `/home/ssm-user/rag-production/` |
| Auth | `X-API-Key` header (same key as search API — separation is on the to-do list) |

---

## 2. Architecture

```
Browser (HF Space, static HTML+JS)
   │  fetch + X-API-Key            ← CORS allowlist on FastAPI (main.py)
   ▼
API Gateway → EC2 FastAPI (api/search.py)
   │
   ├── /admin/config        ──► rag-config (6+ taxonomy keys)
   ├── /admin/prompts/*     ──► rag-config (prompt#generation / #intent / #profiles)
   ├── /analytics/*         ──► DynamoDB rag-episodic, Pinecone visitor-profiles
   └── /search (test panel) ──► full live pipeline
   
Consumers (data plane) read registry/config first, code literal as fallback:
   pipeline/prompt_registry.py   ← 5-min TTL cache, bust() on admin writes
   pipeline/generator.py         ← get_prompt("generation", SYSTEM_PROMPT)
   pipeline/intent_detector.py   ← (_gp("intent","") or INTENT_PROMPT)
   scripts/nightly_consolidation.py ← (_gp("profiles","") or """literal""")
   api/search.py                 ← get_config(key, fallback) for taxonomies
```

### 2.1 Cache-invalidation contract (the important one)

> **Save any prompt → version bumps → L1 keys rotate AND semantic entries self-reject →
> first query regenerates → second query caches cleanly.**

Mechanics:
- `pipeline/semantic_cache.py`: `_pv()` reads live version from registry; **PV-MISS guards on
  all 3 stored-return paths** reject entries whose stored `prompt_version` ≠ current.
- `pipeline/generator.py` line ~260: L1 answer-cache key includes
  `"pv": get_prompt_version("generation", 2)` — dynamic, so a version bump changes the key.
- Verified live 12 Jun: v4 save → `cached: False` → `PV-MISS stored=None current=4` in journal
  → `cached: True` on re-run.

**The old manual checklist (bump PROMPT_VERSION, bump pv=, bump CACHE_VERSION, restart, run
clear script) is RETIRED for prompt changes.** It still applies to *code* changes in the
generation path.

---

## 3. Tab reference (tech team)

### 📦 Projects
Tenant cards with live stats (docs / vectors / profiles via `/analytics/*`), queries/day from
`/analytics/stats` (sums `hourly_volume[].count`). "New project" card documents the 6-step
onboarding flow — steps 1–2 ship in WS1, step 3 (AI taxonomy extraction) is productization backlog.

### 🏷️ Signals & Tags
Editable chip UI over **7 rag-config keys**. Chips colour-coded per key. Mappings use
`keyword=Label ⏎` format; lists use plain text. Dirty-bar shows unsaved keys; Save does
`PUT /admin/config/{key}` per dirty key and busts the API's in-process config cache.

| Key | Type | Drives |
|---|---|---|
| `competitor_signals` | list (15 seeded) | competitor flagging → weekly sales brief (WS3) |
| `product_signals` | map keyword→product | product affinity scoring |
| `workload_signals` | map keyword→workload | memory-panel workload tags |
| `commercial_keywords` | list | commercial CTA / SOLID_LEAD trigger |
| `equinix_products` | list (13) | valid products for intent model |
| `equinix_use_cases` | list (17) | use-case detection |
| `workload_badge_styles` | map label→{icon,bg,color} | UI badge styling (also served at `/config/badge-styles`) |

Propagation: **≤ 5 minutes** (config TTL=300s in `api/search.py::_load_config`).

### ✍️ Prompt Studio
Registry of 3 prompts, each independently versioned in rag-config as `prompt#<id>`:

| id | Model | Consumer | Save effect |
|---|---|---|---|
| `generation` | GPT-4o | generator.py | version bump → **full cache invalidation** (see 2.1) |
| `intent` | GPT-4o-mini | intent_detector.py | next query; **PUT rejects** if `{products}`/`{use_cases}` placeholders missing |
| `profiles` | GPT-4o-mini | nightly_consolidation.py | next consolidation run (timer currently **paused** — run manually) |

Pills show `v<N>` (registry-controlled) or `code` (fallback). Test panel runs real queries
through `/search` showing intent/products/workloads/cache status.
Rollback: paste previous text → Save (forward-only versions; snapshot/restore is on the to-do).

### 💰 Cost Center
Estimates = live query volume × published model pricing, cache-adjusted (generation+rerank only
on misses). Components: GPT-4o gen $0.012/q · mini intent $0.0009/q · embed $0.00002/q ·
Cohere rerank $0.002/q · profiles $0.35/day (currently $0 — timer paused). Exact metering =
productization billing work.

### 🔑 API Keys
UI scaffolded; `/admin/keys` endpoints **not yet built** (to-do). Current truth: one key in
EC2 `.env`, used by search + admin + HF Space.

---

## 4. Backend endpoint reference

| Endpoint | Method | Purpose | Notes |
|---|---|---|---|
| `/api/v1/admin/config` | GET | all 7 taxonomy keys | |
| `/api/v1/admin/config/{key}` | PUT | update one key | body `{"data": ...}`; busts config cache |
| `/api/v1/admin/prompts` | GET | list 3 prompts + meta | version, chars, source |
| `/api/v1/admin/prompts/{id}` | GET/PUT | read / save+bump | PUT guards: len≥50; intent placeholders |
| `/api/v1/admin/prompt` | GET/PUT | **legacy** single-prompt | superseded; kept for compat |
| `/api/v1/config/badge-styles` | GET | badge styles for UIs | used by HF Space app |
| `/api/v1/analytics/stats` | GET | volume/hourly/cache-rate | console metrics + costs |
| `/api/v1/analytics/visitor-profiles` | GET | profile cards | |
| `/api/v1/search` | POST | live pipeline | test panel |

Key files: `api/search.py` (endpoints, `get_config`), `pipeline/prompt_registry.py`
(get_prompt / get_prompt_version / bust), `pipeline/semantic_cache.py` (`_pv()`, PV-MISS guards),
`pipeline/generator.py` (L1 key), `main.py` (CORS allowlist incl. the HF Space origin).

---

## 5. Engineering change discipline (hard-won, mandatory)

Adopted after the 12-Jun four-bug incident (docstring patch landing, substring-unsafe replace,
frozen alias, hardcoded L1 pv). **All EC2 patches must:**

1. **Parse BEFORE write** — `ast.parse(new_src)` before `open(...).write()`. Never the reverse.
2. **Validate by IMPORT, not just parse** — subprocess `import <module>` with auto-revert on
   failure. (`ast.parse` passed every one of the four bugs.) Exception: scripts with top-level
   execution (`nightly_consolidation.py`) use `py_compile` — importing them runs them.
3. **No substring-unsafe replaces** — `"as _pv"` ate `"as _pv_code"`. Anchor with regex `\b`
   boundaries or full-line patterns.
4. **Browser-bound HTML/JS**: run inline scripts through `node --check` before shipping
   (apostrophes in single-quoted JS strings via Python patching = silent script death).
5. **Evidence before patches** — grep ground truth first; print diagnostics with the critical
   section LAST (terminal pastes truncate from the top).
6. **Long EC2 sessions**: start `tmux` immediately after SSM connect; sessions idle-out at
   ~20 min and kill running scripts (`tmux attach` to resume).
7. **No silent compatibility shims** — the frozen `PROMPT_VERSION` alias served stale values
   for hours. Prefer loud import failures.

---

## 6. Maintenance runbook (operations team)

### 6.1 Routine tasks

| Task | How | Takes effect |
|---|---|---|
| Add/remove a signal tag | Console → Signals & Tags → edit chips → **Save to rag-config** | ≤ 5 min |
| Add a competitor to track | Same, `competitor_signals` key | ≤ 5 min |
| Edit any prompt | Console → Prompt Studio → select pill → edit → **Save & bump version** | next query (generation/intent) |
| Refresh buyer profiles (pre-demo!) | EC2: `python3.11 scripts/nightly_consolidation.py` (~90 s) | immediate |
| Resume nightly profiles | EC2: `sudo systemctl enable --now rag-consolidation.timer` | 2am nightly |
| Pause nightly profiles | EC2: `sudo systemctl stop rag-consolidation.timer && sudo systemctl disable rag-consolidation.timer` | immediate |
| Restart the API | EC2: `sudo systemctl restart rag-api` (≈6 s downtime) | immediate |
| Re-warm demo cache | Run the 8-query demo sequence in the HF Space once | immediate |
| Update the console UI | HF Space → upload new `index.html` → **hard-refresh (Ctrl+Shift+R)** — static Spaces cache aggressively | ~1 min |

### 6.2 Current operational state (as of 12 Jun 2026)

- ⏸ **Consolidation timer PAUSED** (cost saving). Profiles frozen at last manual run.
  **Run manually before any demo of Buyer Intelligence.**
- Prompts: generation **v4**, intent **v1**, profiles **v1** — all registry-controlled.
- Demo answer cache was emptied during the 12-Jun work — re-warm before leadership demos.

### 6.3 Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Console: "Blocked by the browser (CORS)" | API doesn't allow the Space origin | Add Space URL to CORS allowlist in `main.py`, restart rag-api |
| Console: "/admin/config not deployed" | Endpoints missing after a rebuild | Re-run admin endpoint deploy (tech team) |
| Console: "API key rejected (401/403)" | Wrong key | Use the key from EC2 `.env` |
| Console loads but nothing works, no errors | JS syntax error killed the script block | F12 console; tech team validates with `node --check` |
| Queries/day shows garbage like `[object Object]` | Stats shape change | Tech team: check `/analytics/stats` `hourly_volume` shape |
| Old answer after a prompt change | Should be impossible (see 2.1) | Check journal: `sudo journalctl -u rag-api \| grep PV-MISS`; if PV-MISS **keeps firing for just-cached queries**, a cache write path isn't stamping `prompt_version` — escalate with `semantic_cache.set` |
| Same wrong answer persists across "cache clears" | A layer not covered by the clear | Surgical: cache-clear utility `MODE="query"`; nuclear: `MODE="all"` (≈$2.50 + slow demos while re-warming) |
| Profiles stale in dashboard | Timer paused | Manual run (6.1) or resume timer |
| Search down entirely | rag-api crashed | `systemctl status rag-api`; `journalctl -u rag-api -n 50`; restart; if import error → tech team |
| Search up, buyer features dark | DynamoDB/OpenAI-mini issue | Search degrades gracefully; check dependency health (Ops tab when built; until then: journal) |
| SSM session keeps dying mid-task | 20-min idle timeout | `tmux` after connect; `tmux attach` to resume |

### 6.4 Health quick-checks (copy-paste)

```bash
cd /home/ssm-user/rag-production && export $(grep -v '^#' .env | grep -v '^$' | xargs)
curl -s http://localhost:8000/api/v1/health                       # {"status":"ok",...}
systemctl is-active rag-api rag-consolidation.timer               # active / inactive
sudo journalctl -u rag-api --since "1 hour ago" | grep -c PV-MISS # small+shrinking = healthy
curl -s http://localhost:8000/api/v1/admin/prompts -H "X-API-Key: $API_KEY" | python3 -m json.tool
```

---

## 7. Security notes

- HF Space is **private** (operator login) + API key = two gates. Keep it private.
- One shared API key currently serves search AND admin — **admin/tenant key separation is
  mandatory before a second tenant** (to-do).
- Planned restart endpoint must use a sudoers entry scoped to the **exact** command
  (`NOPASSWD: /usr/bin/systemctl restart rag-api`) — never a parameterised shell.
- Secrets still in `.env` → Secrets Manager migration is WS2 week 1 (InfoSec gate).

---

## 8. To-do list

### 8.1 Operations tab (next console build, ~2 wks)
1. **Cache flush** (2d) — `POST /admin/cache/flush {mode}` reusing clear-utility MODEs
   (answers / semantic / bm25 / query / all) + stats panel (key counts, hit rate,
   **PV-MISS 24h counter**) + blast-radius confirm + ops audit log.
2. **Service restart + timer control** (2d) — scoped-sudoers restart; consolidation
   pause/resume/**run-now** buttons; uptime + timer-state badges.
3. **Dependency health panel** (2d) — Pinecone/OpenAI/Cohere/Redis/DynamoDB ping cards
   (WS2 canary, given a face); degradation-mode indicator post circuit-breakers.
4. **Logs viewer** (1d) — `GET /admin/logs?lines&filter`, saved filters for known signatures.
5. **Config snapshot & restore** (2d) — auto-snapshot before every console save →
   `rag-config-history`; one-click restore = undo for bad edits **and prompt rollback**.
   *Build immediately after cache flush — highest safety value.*
6. **Ops audit log** (1d) — who/what/when/old→new on every control-plane write (SOC2 trail).
7. **GDPR/data maintenance** — visitor-erasure UI over the WS2 endpoint; tester-filter
   (`@equinix.com`) as an editable rag-config key.
8. **Maintenance-mode toggle** (1d) — rag-config flag → banner in Space/widget, CTA paused.

### 8.2 Console gaps
- `/admin/keys` endpoints (issue/revoke/usage) for the API Keys tab.
- Admin vs tenant **key separation**.
- Cost Center: wire profiles row to actual timer state; real metering (productization).
- Onboarding wizard steps 1–2 live (WS1) → step 3 AI taxonomy extraction (productization).

### 8.3 Watch items
- **PV-MISS on just-cached queries** → a set-path missing the version stamp (see 6.3).
- Legacy `/admin/prompt` endpoint — remove once nothing references it.
- After Lambda migration (WS2): restart semantics change to deploy/alias-flip; timer →
  EventBridge; EC2 runbook entries retire.

---

## 9. Change log (how we got here — 11–12 Jun 2026)

- Console built (5 tabs) and deployed as private static HF Space; CORS added.
- Connection diagnostics (CORS/404/auth); stats endpoint fixed (`/analytics/stats`);
  reduce-concat bug fixed (`count:0` falsy trap).
- 7th config key `competitor_signals` seeded (15 names) — WS3 wk-1 item done early.
- Prompt registry: module + seeding (extracted live prompts from code) + 3 consumers wired
    + `/admin/prompts` CRUD. Generation prompt updated to v3 (pricing-redirect rules) live.
- **Cache-invalidation chain completed** after 4-bug incident: PV-MISS guards (3 paths),
  dynamic `_pv()`, L1 key pv-dynamic, frozen alias deleted. Contract proven at v4.
- Consolidation timer paused (cost). Manual prompt-change checklist retired.
- Engineering discipline codified (section 5).