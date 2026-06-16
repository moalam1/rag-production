import re
_FB = {
  "Megaport": ["megaport"],
  "Console Connect": ["console connect", "consoleconnect"],
  "AWS Direct Connect": ["direct connect", "directconnect"],
  "Azure ExpressRoute": ["expressroute", "express route"],
  "Google Cloud Interconnect": ["cloud interconnect", "partner interconnect"],
  "Digital Realty": ["digital realty", "interxion"],
  "NTT": ["ntt", "ntt global", "ntt data"],
  "CoreSite": ["coresite"],
  "Cyxtera": ["cyxtera"],
  "Lumen": ["lumen", "centurylink"],
  "Cologix": ["cologix"],
  "CyrusOne": ["cyrusone"],
}

def detect_competitors(query, competitor_config=None):
    if not query:
        return []
    cfg = competitor_config or _FB
    q = query.lower()
    hits = set()
    for canonical, aliases in cfg.items():
        for alias in aliases:
            a = str(alias).lower().strip()
            if not a:
                continue
            if re.search(r"\b" + re.escape(a) + r"\b", q):
                hits.add(canonical)
                break
    return sorted(hits)
