# Independent facet-plan review

## Scope and firewall

The independent content reviewer inspected all 145 proposed facets using only:

- the 22 supplied narratives;
- `facet_prompt.md`;
- the proposed `facet_manifest.json`; and
- the approved all-topic validation design.

The reviewer checked obligation completeness, narrative order, invented
subtopics, query tethering, bridge safety, deterministic ordering, and likely
BM25 lexical specificity. No qrels, nuggets, retrieved documents, candidate
results, evaluation metrics, model output, or web sources were opened or used.

A separate code/safety review inspected manifest schema validation, local cache
discovery, the planning CLI, and offline-firewall test coverage.

## Findings and resolutions

All Important and Minor findings were resolved before the planning freeze:

- Topic 58: split nuclear-energy safety from accident risks/Chernobyl.
- Topic 144: split financial-institution safety from trust.
- Topic 200: separated historical and societal impact, restricted European
  Jewry to those two impact facets, removed it from unrelated queries, and
  retained the audited `concluded` relation bridge.
- Topic 213: combined US involvement and Cold War strategy into one obligation.
- Topic 219: split positive/negative effects for daily life, government, and
  business/telehealth; retained technical societies and device rationing.
- Topic 161: removed an unnecessary women-domain tether from current options.
- Topic 477: changed the definition query to the narrative-specific
  `concept of race how defined` wording.
- Topic 499: restored the explicit `influence` relation for Western-country
  perspectives.
- Manifest validation now rejects coercible numeric/boolean IDs, boolean orders
  and counts, extra schema fields, and non-string bridge provenance.
- Offline tests now block standard-library network entry points, qrels-named
  file access, and model-loading imports; missing and escaping cache cases are
  tested directly.

The reviewer approved every other facet without changes. In particular, topic
225 retains explicit human/child/adult tethers; topic 707 adds no pet/human
claim or unsupported implementation domain; and no query adds dictionary,
Scrabble, essay-writing, or generic-process wording.

## Final exact-22 inventory

| Topic | Facets |
|---:|---:|
| 14 | 7 |
| 31 | 8 |
| 37 | 9 |
| 58 | 9 |
| 72 | 7 |
| 84 | 7 |
| 144 | 7 |
| 161 | 7 |
| 200 | 9 |
| 213 | 6 |
| 219 | 8 |
| 224 | 7 |
| 225 | 7 |
| 233 | 3 |
| 273 | 7 |
| 300 | 4 |
| 407 | 5 |
| 477 | 7 |
| 499 | 6 |
| 515 | 6 |
| 707 | 5 |
| 897 | 7 |
| **Total** | **148** |

These resolutions were independently re-reviewed before the real planning
directory was frozen. The canonical planning root is
`bc1351cca8aa05dd0979a342c8dd5f72395f2b7a9207ae20668aba05f1ab85a2`.
