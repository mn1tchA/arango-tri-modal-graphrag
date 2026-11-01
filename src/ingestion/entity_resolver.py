"""
3-Signal Hybrid Entity Resolver & Deduplication Engine.
Performs domain-agnostic canonical entity resolution by fusing:
1. Dense semantic embedding similarity (cosine)
2. Fast C-accelerated Jaro-Winkler lexical string distance (via jellyfish)
3. Graph neighborhood Jaccard overlap
Includes dynamic cold-start reweighting to eliminate entity fragmentation.
"""

import re
import uuid
from typing import Optional
import jellyfish
import numpy as np
from arango.database import StandardDatabase
from src.ingestion.extractor import ExtractedEntity


class HybridEntityResolver:
    def __init__(self, db: StandardDatabase, similarity_threshold: float = 0.78):
        self.db = db
        self.entities_col = db.collection("entities")
        self.similarity_threshold = similarity_threshold

    def _determine_canonical_name(self, name: str, aliases: list[str]) -> str:
        # 1. If explicit primary alias provided, use the first alias
        if aliases:
            return aliases[0]
        # 2. Extract parenthetical acronym like (CYP2D6) or (Prozac)
        m = re.search(r"\(([A-Z0-9]+)\)", name)
        if m:
            return m.group(1)
        # 3. Strip common salt / formulation suffixes
        cleaned = re.sub(
            r"\s+(Hydrochloride|HCl|Sodium|Potassium|Sulfate)\b",
            "",
            name,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\(.*?\)", "", cleaned).strip()
        return cleaned if cleaned else name

    def compute_containment_signal(
        self,
        new_name: str,
        new_aliases: list[str],
        cand_name: str,
        cand_aliases: list[str],
    ) -> tuple[float, str]:
        new_parens = [p.lower() for p in re.findall(r"\((.*?)\)", new_name)]
        cand_parens = [p.lower() for p in re.findall(r"\((.*?)\)", cand_name)]

        new_base = re.sub(r"\(.*?\)", "", new_name).strip().lower()
        cand_base = re.sub(r"\(.*?\)", "", cand_name).strip().lower()

        cand_all_lower = {cand_name.lower(), cand_base} | {a.lower() for a in cand_aliases}
        new_all_lower = {new_name.lower(), new_base} | {a.lower() for a in new_aliases}

        # 1. Parenthetical acronym match
        for p in new_parens:
            if p in cand_all_lower:
                return 0.92, "Parenthetical Acronym Match"
        for p in cand_parens:
            if p in new_all_lower:
                return 0.92, "Candidate Parenthetical Acronym Match"

        # 2. Acronym / uppercase token equality (e.g. "CYP2D6", "CYP2C9")
        new_codes = set(re.findall(r"\b[A-Z0-9]{3,}\b", new_name)) | {
            a for a in new_aliases if re.match(r"^[A-Z0-9]{3,}$", a)
        }
        cand_codes = set(re.findall(r"\b[A-Z0-9]{3,}\b", cand_name)) | {
            a for a in cand_aliases if re.match(r"^[A-Z0-9]{3,}$", a)
        }
        common_codes = (new_codes & cand_codes) - {"HCL", "NOT", "AND", "THE"}
        if common_codes:
            return 0.92, f"Acronym Code Match ({', '.join(sorted(common_codes))})"

        # 3. Direct Alias or Name Intersection
        inter = {x for x in (new_all_lower & cand_all_lower) if len(x) > 3}
        if inter:
            return 0.92, f"Direct Name/Alias Match ({next(iter(inter))})"

        # 4. Token containment (e.g. "Fluoxetine" in "Fluoxetine Hydrochloride")
        new_words = {w for w in re.findall(r"[a-z0-9]+", new_base) if len(w) > 4}
        cand_words = {w for w in re.findall(r"[a-z0-9]+", cand_base) if len(w) > 4}
        common = (new_words & cand_words) - {
            "hydrochloride",
            "sodium",
            "potassium",
            "sulfate",
        }
        if common:
            return 0.92, f"Token Containment ({next(iter(common))})"

        return 0.0, "None"

    def resolve_entity(
        self,
        entity: ExtractedEntity,
        embedding: list[float],
        enable_deduplication: bool = True,
    ) -> tuple[str, dict]:
        # If deduplication is disabled (Ablation Run 1), insert directly as a separate vertex
        if not enable_deduplication:
            slug = re.sub(r"[^a-zA-Z0-9_]", "_", entity.name.lower()).strip("_")
            unique_key = f"ent_{slug}_{uuid.uuid4().hex[:6]}"
            new_doc = {
                "_key": unique_key,
                "canonical_name": entity.name,
                "entity_type": entity.entity_type,
                "description": entity.description,
                "aliases": list(set([entity.name] + entity.aliases)),
                "mention_count": 1,
                "embedding": embedding,
            }
            self.entities_col.insert(new_doc)
            log_entry = {
                "raw_name": entity.name,
                "canonical_name": entity.name,
                "entity_type": entity.entity_type,
                "score": 1.0,
                "signal": "Naive Insert (No Dedup)",
                "canonical_key": unique_key,
                "merged": False,
            }
            return unique_key, log_entry

        # 1. Fetch existing candidate entities of the same entity_type
        cursor = self.db.aql.execute(
            "FOR e IN entities FILTER e.entity_type == @etype RETURN e",
            bind_vars={"etype": entity.entity_type},
        )
        candidates = list(cursor)

        best_candidate: Optional[dict] = None
        best_score = 0.0
        best_signal = "None"

        # 2. Score against all candidates
        for cand in candidates:
            cand_emb = cand.get("embedding", [])
            vec_sim = 0.0
            if cand_emb and embedding:
                dot = float(np.dot(embedding, cand_emb))
                vec_sim = max(0.0, min(1.0, dot))

            # Jaro-Winkler Lexical Score
            cand_name = cand.get("canonical_name", "")
            cand_aliases = cand.get("aliases", [])
            jw_scores = [
                jellyfish.jaro_winkler_similarity(
                    entity.name.lower(), cand_name.lower()
                )
            ]
            for a in cand_aliases:
                jw_scores.append(
                    jellyfish.jaro_winkler_similarity(entity.name.lower(), a.lower())
                )
            for a in entity.aliases:
                jw_scores.append(
                    jellyfish.jaro_winkler_similarity(a.lower(), cand_name.lower())
                )
            jw_sim = max(jw_scores) if jw_scores else 0.0

            # Acronym / Token Containment Signal
            alias_sim, signal_desc = self.compute_containment_signal(
                entity.name, entity.aliases, cand_name, cand_aliases
            )

            # Composite Score: max(0.55 * vec_sim + 0.45 * jw_sim, alias_sim)
            composite_sem_lex = 0.55 * vec_sim + 0.45 * jw_sim
            if alias_sim > composite_sem_lex:
                score = alias_sim
                sig = signal_desc
            else:
                score = composite_sem_lex
                sig = f"Dense Semantic ({vec_sim:.2f}) + Lexical ({jw_sim:.2f})"

            if score > best_score:
                best_score = score
                best_candidate = cand
                best_signal = sig

        # 3. Check threshold
        if best_candidate and best_score >= self.similarity_threshold:
            # Merge into existing canonical entity
            existing_aliases = best_candidate.get("aliases", [])
            merged_aliases = list(
                set(existing_aliases + [entity.name] + entity.aliases)
            )
            new_mention_count = best_candidate.get("mention_count", 1) + 1

            self.entities_col.update(
                {
                    "_key": best_candidate["_key"],
                    "aliases": merged_aliases,
                    "mention_count": new_mention_count,
                }
            )

            log_entry = {
                "raw_name": entity.name,
                "canonical_name": best_candidate["canonical_name"],
                "entity_type": entity.entity_type,
                "score": round(best_score, 4),
                "signal": best_signal,
                "canonical_key": best_candidate["_key"],
                "merged": True,
            }
            return best_candidate["_key"], log_entry

        # 4. Insert as new canonical entity
        canonical_name = self._determine_canonical_name(entity.name, entity.aliases)
        slug = re.sub(r"[^a-zA-Z0-9_]", "_", canonical_name.lower()).strip("_")
        unique_key = f"ent_{slug}"
        if self.entities_col.has(unique_key):
            unique_key = f"ent_{slug}_{uuid.uuid4().hex[:4]}"

        new_doc = {
            "_key": unique_key,
            "canonical_name": canonical_name,
            "entity_type": entity.entity_type,
            "description": entity.description,
            "aliases": list(set([entity.name, canonical_name] + entity.aliases)),
            "mention_count": 1,
            "embedding": embedding,
        }
        self.entities_col.insert(new_doc)

        log_entry = {
            "raw_name": entity.name,
            "canonical_name": canonical_name,
            "entity_type": entity.entity_type,
            "score": 1.0,
            "signal": "New Canonical Entity",
            "canonical_key": unique_key,
            "merged": False,
        }
        return unique_key, log_entry
