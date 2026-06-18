import sys as _s; _s.path.insert(0, "/home/ssm-user/rag-production")
from pipeline.prompt_registry import get_prompt as _gp
#!/usr/bin/env python3.11
"""Nightly visitor profile consolidation — runs at 2am via cron."""
import sys, os, json, time
sys.path.insert(0, "/home/ssm-user/rag-production")
os.chdir("/home/ssm-user/rag-production")
from dotenv import load_dotenv; load_dotenv(".env")
import boto3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from openai import OpenAI
from pinecone import Pinecone
from config import settings

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc     = Pinecone(api_key=settings.PINECONE_API_KEY)
index  = pc.Index(settings.PINECONE_INDEX)

client   = boto3.client("dynamodb", region_name="us-east-1")
pages    = client.get_paginator("scan").paginate(TableName="rag-episodic")
visitors = defaultdict(list)

for page in pages:
    for item in page["Items"]:
        vid = item.get("visitor_id", {}).get("S", "")
        if vid and not vid.startswith(("v_debug","v_test","v_prod_guest")):
            q = {}
            for k, v in item.items():
                typ = list(v.keys())[0]
                val = list(v.values())[0]
                if typ == "L":
                    q[k] = [list(x.values())[0] for x in val if isinstance(x, dict)]
                elif typ == "N":
                    try: q[k] = float(val)
                    except: q[k] = val
                else:
                    q[k] = val
            visitors[vid].append(q)

print(f"[{datetime.now()}] Consolidating {len(visitors)} visitors...")
upserted = 0

for vid, queries in visitors.items():
    if len(queries) < 3:
        continue
    try:
        # Extract identity if visitor submitted their details
        identity_records = [q for q in queries if q.get("intent","") == "identity_capture"]
        name  = identity_records[0].get("name","")  if identity_records else ""
        email = identity_records[0].get("email","") if identity_records else ""
        company = identity_records[0].get("company","") if identity_records else ""
        country = identity_records[0].get("country","") if identity_records else ""
        # Source 2 fallback: derive company from work-email domain when the form
        # company is blank. Skip personal-email providers.
        if not company and email and "@" in email:
            _dom = email.split("@")[1].lower().strip()
            if _dom and _dom not in ("gmail.com","yahoo.com","hotmail.com","outlook.com","icloud.com","aol.com","proton.me","protonmail.com"):
                company = _dom

        # Build query history (exclude identity capture records)
        search_queries = [q for q in queries if q.get("intent","") != "identity_capture"]
        window = (search_queries[:5] + search_queries[-25:]) if len(search_queries) > 30 else search_queries
        history = "\n".join(
            f"- [{q.get('intent','?')}] {q.get('query','')}"
            for q in window
        )
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":_gp("profiles", "") or """You are a B2B sales intelligence analyst writing a pre-call brief for a sales rep. Based on this visitor's actual search history, write exactly 4 sentences:

Sentence 1 — WHAT THEY DID: Name the exact products, exact questions asked, and sequence of research. Use specifics: product names, features, specs mentioned.
Sentence 2 — WHAT THEY ARE SOLVING: Infer the business problem or technical use case they are trying to address. Be specific — don't say 'network needs', say 'SD-WAN branch deployment for hybrid multicloud'.
Sentence 3 — WHERE THEY ARE IN THE JOURNEY: Awareness / Consideration / Evaluation / Ready to buy. Cite the evidence — e.g. 'pricing and contract questions indicate they are close to procurement'.
Sentence 4 — SALES ANGLE: One specific recommendation for the sales rep — what to lead with, what to offer, what to watch out for.

Rules:
- Use the visitor's name ONLY if explicitly provided in "Identified visitor:" line above.
- If no name is provided, say "This visitor" — NEVER invent or assume a name like Jordan, Alex, Sarah etc.
- Do NOT use placeholder names. Do NOT hallucinate identity.
- Never use these generic phrases: 'strong interest in', 'indicating they are', 'likely in the evaluation phase', 'indicating a thorough', 'likely assessing', 'suggesting they are', 'appears to be', 'seems to be', 'it seems', 'it appears'.
- Every sentence must contain at least one specific fact — a product name, a feature, a query they ran, or a number.
- Never start with 'The visitor' or 'The ideal buyer profile'.
- Be direct, specific, and actionable. Write like a senior SDR briefing an AE before a call.

Example output:
'Mohammed asked about Equinix Fabric port speeds and SLAs, then compared Fabric vs Network Edge for SD-WAN deployment, then asked about cross-connect pricing and enterprise contracts — all within one session. He is solving a hybrid multicloud networking challenge, likely evaluating Equinix as the primary interconnect layer for a multi-region SD-WAN rollout. Pricing and contract questions in query 5 confirm he is in active procurement — budget is allocated, decision is imminent. Lead with a Fabric + Network Edge bundle proposal and offer a Solutions Architect call to design the deployment blueprint.'"""},
                {"role":"user","content":(
                f"{'Identified visitor: ' + name + (' <' + email + '>' if email else '') + chr(10) if name or email else ''}"
                f"Query history:\n{history}"
            )}
            ],
            max_tokens=300, temperature=0.2,
        )
        profile = resp.choices[0].message.content.strip()
        emb = openai.embeddings.create(
            model="text-embedding-3-small", input=profile, dimensions=1024
        ).data[0].embedding

        all_products = []
        for q in queries:
            try:
                p = q.get("products","[]")
                if isinstance(p, str): p = json.loads(p)
                elif not isinstance(p, list): p = []
                all_products.extend([x for x in p if isinstance(x, str)])
            except: pass

        top_products = [p for p,_ in Counter(all_products).most_common(3)]
        tags = [q.get("lead_quality_tag","") for q in queries if isinstance(q.get("lead_quality_tag",""), str) and q.get("lead_quality_tag","").strip()]
        # Highest-tier wins — don't let noisy EARLY_EXPLORER or DEAD_END records
        # override a genuine commercial or tech-pilot signal
        TIER_RANK = {"SOLID_LEAD_COMMERCIAL":4,"TECH_PILOT_ENGAGED":3,"EARLY_EXPLORER":2,"DEAD_END_SUPPORT":1}
        best_tag = max(set(tags), key=lambda t: TIER_RANK.get(t,0)) if tags else "EARLY_EXPLORER"

        # Infer stage — align with best_tag and intents
        _n = len(queries)
        _intents = [q.get("intent","") for q in queries]
        if best_tag == "SOLID_LEAD_COMMERCIAL":
            _vstage = "intent"
        elif best_tag == "TECH_PILOT_ENGAGED":
            _vstage = "evaluation" if "compare" in _intents else "consideration"
        elif any(i in _intents for i in ["evaluate_specs","compare","troubleshoot"]):
            _vstage = "consideration"
        else:
            _vstage = "awareness"

        index.upsert(vectors=[{
            "id": vid,
            "values": emb,
            "metadata": {
                "visitor_id":   vid,
                "profile":      profile[:800],
                "top_products": json.dumps(top_products),
                "lead_tag":     best_tag,
                "stage":        _vstage,
                "query_count":  str(len(search_queries)),
                "name":         name,
                "email":        email,
                "company":      company,
                "country":      country,
                "identified":   "true" if (name or email) else "false",
                "updated_at":   datetime.now(timezone.utc).isoformat(),
            }
        }], namespace="visitor-profiles")
        upserted += 1
        time.sleep(0.2)
    except Exception as e:
        print(f"  ✗ {vid[:20]}: {e}")

print(f"[{datetime.now()}] Done: {upserted} profiles upserted")
