#!/usr/bin/env python3.11
"""Read the full Equinix IBX JSON and write a slim metro_map.json with only
the fields the resolver + regional-heatmap rollup need. Re-run when IBX
refreshes. Rule #5: writes on target, asserts the result.

Usage: python3.11 extract_metro_map.py <IBX_Latest.json> <out metro_map.json>
"""
import json, sys
from collections import Counter

src = sys.argv[1] if len(sys.argv) > 1 else "IBX_Latest.json"
out = sys.argv[2] if len(sys.argv) > 2 else "data/metro_map.json"

d = json.load(open(src))
recs = d.get('ibxCertificationDetailsList', []) + d.get('ibxCertDetailsNotLiveList', [])
assert recs, "no IBX records found - wrong file?"

city_to_metro, iso_to_metros, metro_meta = {}, {}, {}
metro_counts = Counter()
for r in recs:
    metro  = (r.get('metro') or '').strip()
    code   = (r.get('metroCode') or '').strip()
    region = (r.get('region') or '').strip()
    addr   = r.get('address', {}) or {}
    iso    = (addr.get('countryCode') or '').strip().upper()
    city   = (addr.get('city') or '').strip()
    if metro:
        metro_counts[metro] += 1
        metro_meta.setdefault(metro, {'code': code, 'region': region, 'iso': iso})
    if city and metro:
        city_to_metro[city.lower()] = metro
    if iso and metro:
        iso_to_metros.setdefault(iso, []).append(metro)

flagship = {iso: max(set(ms), key=lambda m: metro_counts[m])
            for iso, ms in iso_to_metros.items()}

minimal = {
    "_generated_from": "IBX_Latest.json",
    "_counts": {"cities": len(city_to_metro), "countries": len(flagship), "metros": len(metro_meta)},
    "city_to_metro": city_to_metro,
    "country_flagship": flagship,
    "metro_meta": metro_meta,
}
text = json.dumps(minimal, indent=2, sort_keys=True)
open(out, 'w').write(text)
assert len(text) > 2000, f"output suspiciously small ({len(text)} bytes)"
print(f"WROTE {out}: {len(text)} bytes")
print(f"  cities={len(city_to_metro)} countries={len(flagship)} metros={len(metro_meta)}")
