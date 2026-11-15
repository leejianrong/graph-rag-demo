# A short form in one document no longer pulls a different person into the entity (GRA-178)

Done on 2026-11-16. The cause was found in GRA-172 ([real-2wiki-medium-50.md](real-2wiki-medium-50.md)): entity linking's exact-key block merges on any surface form or coref cluster member that equals a name or alias of an existing same-type entity, before the name guard or the cosine is consulted. A bare short form such as "Frederick" that one document's coref clustered with a person therefore became an identity key in the next document.

## The change

In `graph_rag/stages/entity_linking.py`, a block hit that rests only on a bare short form of the document entity (a single-token member that is not the entity's own surface) is kept only if the entity's name relates to the candidate's name under the same rule a similarity merge needs (`_pair_compatible`: one name's distinctive tokens inside the other's, or an acronym). So "Saba" still finds "Saba Industries", and "Frederick Calvert, 6th Baron Baltimore" no longer absorbs "Frederick, Hereditary Prince of Baden-Durlach". An exact match on the entity's own name, or on a multi-token member, is unchanged. The `_names_compatible` guard now uses the same pair rule.

Unit tests ([test_el_name_guard.py](../../tests/unit/test_el_name_guard.py)): the two-document repro that was a strict xfail now passes; two different people sharing a surname through their short form stay apart; "Saba" still finds "Saba Industries". Against the old block the first two fail and the third passes under both, which guards against over-tightening.

## What it did

| measure | before | after |
|---|---|---|
| `hard-small`, `en_core_web_sm`, recorded Haiku stages (mention, entity, entity-pair, triple, linking F1) | 0.856, 0.728, 0.796, 0.789, 0.859 | identical; every recorded prompt still hits the cache, so linking did not change |
| `hard-small`, `en_core_web_trf` | 0.900, 0.737, 0.881, 0.876, 0.870 | 0.900, 0.738, 0.884, 0.879, 0.870 |
| B-cubed linking, hashing embedder, `medium` | | identical (that benchmark has no coref clusters, so it cannot see this change) |
| `hard-small` whole-path@5 rows | | identical |

On `trf` the change touched 12 of the 145 documents (4 distinct recorded graph-build prompts changed and were re-recorded; the committed cache now holds 294). It cost nothing measurable on `hard-small`, and entity-pair and triple F1 rose by 0.003. The synthetic worlds are where this change is least likely to matter: short forms there are the generator's deliberate aliases, and they satisfy the rule.

**The 2Wiki-50 run through the real stores** (the corpus of [real-2wiki-medium-50.md](real-2wiki-medium-50.md), the same questions, `en_core_web_trf`, replayed from the cache plus 94 newly recorded answers):

| entity | before | after |
|---|---|---|
| Maurice de Prendergast | also "Maurice, Count of Saxony", "de Saxe" | only "Maurice" |
| Frederick Calvert, 6th Baron Baltimore | also "Frederick, Hereditary Prince of Baden-Durlach" | only "Frederick" |
| Rajaraja Chola I | also "Arinjaya Chola", "Sundara Chola" | only "Rajaraja chola I." |
| Santa Catarina | also four rivers | alone |
| Sir Henry Curwen | also "Nicholas Curwen", "Joan Curwen" | **unchanged** |
| Mary | also "Mary, Queen of Scots" | **unchanged** |
| the St John entity | seven merged barons | no longer an entity under that id (not examined further) |

- Four of the six reported over-merges are fixed; two are not. Curwen and Mary reach their aliases by other paths (multi-token cluster members and a generic given name), which this change does not touch.
- **There are 130 more entities** (2,565 before, 2,695 after). Some are correct splits; I did not measure how many are wrong splits of one real entity. Supporting-fact scores are identical (F1 0.335).
- The in-memory link-only replay (no real stores) gave 2,762 and 2,765 entities and a suspicious-alias proxy of 48 and 46. The proxy is noisy: many of the remaining 46 are legitimate variants such as "Kizhan Adikal" and "Kizhan Adigal". It did not reproduce the Curwen merge, because the real Elasticsearch and Neo4j stores link some documents differently from the in-memory fakes. That is also the likely reason a replay of the real run through `kgbench real` missed the cache; I did not confirm it.

## The recorded answers

The 94 new 2Wiki answers and the 4 `hard-small` answers were written by a Haiku agent and carry its judgement calls. It reported: relations left out where the object was typed as a place (HAS_ROLE needs an organisation); conditional or negated statements omitted; address-style `LOCATED_IN` edges added, one factually doubtful (Kedah in Indonesia, as the source states); `RELATED_TO` for spouse, sibling and parent; a `RELATED_TO` edge to the Battle of Lepanto carrying the battle's year as its date, which is wrong; and one `Oliver St John` ID that names two different people. I did not read every answer.

## Not done

- Curwen, Mary and the multi-token merges.
- The 130 extra entities were not checked for wrong splits. A check would need entity gold for the 2Wiki slice, which does not exist.
- A flag: this changes linking for every user with coref clusters. I did not add one because it is a correctness fix with no measured cost, but the owner may want one.

## Reproduce

The caches replay only with the new code: `bench/kgbench/data/llm-cache-2wiki-medium50/` (953 files; 859 recorded under the old linking, 94 added) and `bench/kgbench/data/llm-cache-hard-small-trf/` (294 files; 290 plus 4). The commands are in [hard-small-trf.md](hard-small-trf.md) and [real-2wiki-medium-50.md](real-2wiki-medium-50.md); both need `PATH_RETRIEVAL` unset or `false` for the graph-view numbers.
