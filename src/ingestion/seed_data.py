"""
Multi-Hop Relational Benchmark Corpus and Seed Graph Generator.
Provides a canonical 6-document corpus designed to evaluate multi-hop graph bridging
vs. isolated dense vector retrieval across complex relational pathways.
"""

from sentence_transformers import SentenceTransformer
from arango.database import StandardDatabase

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_text(text: str) -> list[float]:
    # Normalized embeddings allow exact dot-product / cosine calculation in AQL
    vec = get_model().encode(text, normalize_embeddings=True)
    return vec.tolist()


MULTI_HOP_CORPUS_6_DOCS = [
    {
        "_key": "chunk_doc_1",
        "doc_source": "er_admission_note.pdf",
        "text": "Patient admitted to the emergency department following concurrent administration of Fluoxetine Hydrochloride (Prozac) (40mg/day) for depressive disorder and Tramadol HCl (100mg) for acute orthopedic pain.",
    },
    {
        "_key": "chunk_doc_2",
        "doc_source": "cyp450_inhibition_study.pdf",
        "text": "Fluoxetine and its active metabolite norfluoxetine act as potent mechanism-based inhibitors of Cytochrome P450 2D6 (CYP2D6), reducing hepatic metabolic clearance capacity by over 80%.",
    },
    {
        "_key": "chunk_doc_3",
        "doc_source": "opioid_prodrug_kinetics.pdf",
        "text": "Prodrug activation of Tramadol requires metabolic O-demethylation catalyzed by Hepatic Isoenzyme CYP2D6. When CYP2D6 activity is blocked, active metabolite levels plunge while parent Tramadol accumulates, precipitating Serotonin Syndrome and autonomic instability.",
    },
    {
        "_key": "chunk_doc_4",
        "doc_source": "anticoagulant_case_09.pdf",
        "text": "A 68-year-old patient maintained on chronic Warfarin Sodium (Coumadin) anticoagulant therapy was initiated on oral Fluconazole (200mg/day) for severe systemic candidiasis.",
    },
    {
        "_key": "chunk_doc_5",
        "doc_source": "azole_enzyme_interactions.pdf",
        "text": "Clinical pharmacokinetic assays show that Fluconazole (Diflucan) strongly inhibits Cytochrome P450 2C9 (CYP2C9), leading to marked reduction in hepatic oxidation pathways.",
    },
    {
        "_key": "chunk_doc_6",
        "doc_source": "warfarin_clearance_guidelines.pdf",
        "text": "Metabolic clearance of the potent S-enantiomer of Warfarin is strictly dependent on CYP2C9. Concomitant inhibition of CYP2C9 leads to rapid accumulation of Warfarin, resulting in supratherapeutic INR (>8.0) and life-threatening Severe Hemorrhage.",
    },
]

# Backward-compatible alias for existing imports
CLINICAL_CORPUS_6_DOCS = MULTI_HOP_CORPUS_6_DOCS


def seed_sample_graph(db: StandardDatabase):
    chunks_col = db.collection("chunks")
    entities_col = db.collection("entities")
    mentions_col = db.collection("mentions")
    relations_col = db.collection("relations")

    # Clear existing data for clean reproducible runs
    for col in [chunks_col, entities_col, mentions_col, relations_col]:
        col.truncate()

    # 1. Insert Document Chunks
    raw_chunks = [
        {
            "_key": "chunk_1",
            "doc_source": "clinical_case_01.pdf",
            "text": "Patient admitted to the emergency ward following concurrent administration of Fluoxetine (40mg/day) for depression and Tramadol (100mg) for acute orthopedic pain.",
        },
        {
            "_key": "chunk_2",
            "doc_source": "pharmacology_cytochrome_ref.pdf",
            "text": "Fluoxetine and its main metabolite norfluoxetine act as potent mechanism-based inhibitors of the hepatic cytochrome P450 isoenzyme CYP2D6, reducing its metabolic clearance capacity by over 80%.",
        },
        {
            "_key": "chunk_3",
            "doc_source": "opioid_metabolism_guidelines.pdf",
            "text": "Prodrug activation of Tramadol depends strictly on O-demethylation catalyzed by CYP2D6. When CYP2D6 activity is blocked, analgesic efficacy drops sharply while parent Tramadol accumulates, precipitating Serotonin Syndrome and neuromuscular hyperactivity.",
        },
        {
            "_key": "chunk_4",
            "doc_source": "general_orthopedic_care.pdf",
            "text": "Standard post-operative pain management protocols recommend monitoring vital signs every 4 hours when administering oral analgesics alongside antidepressant therapy.",
        },
    ]

    for c in raw_chunks:
        c["embedding"] = embed_text(c["text"])
        chunks_col.insert(c)

    # 2. Insert Extracted Entities
    raw_entities = [
        {"_key": "ent_fluoxetine", "canonical_name": "Fluoxetine", "entity_type": "Drug", "description": "SSRI antidepressant medication"},
        {"_key": "ent_tramadol", "canonical_name": "Tramadol", "entity_type": "Drug", "description": "Opioid prodrug analgesic"},
        {"_key": "ent_cyp2d6", "canonical_name": "CYP2D6", "entity_type": "Enzyme", "description": "Hepatic cytochrome P450 metabolizing enzyme"},
        {"_key": "ent_serotonin_syndrome", "canonical_name": "Serotonin Syndrome", "entity_type": "AdverseEvent", "description": "Toxic serotonergic hyperstimulation and neuromuscular hyperactivity"},
    ]

    for e in raw_entities:
        e["embedding"] = embed_text(f"{e['canonical_name']} - {e['description']}")
        entities_col.insert(e)

    # 3. Link Chunks -> Entities (Mentions Edges)
    mentions = [
        {"_from": "chunks/chunk_1", "_to": "entities/ent_fluoxetine"},
        {"_from": "chunks/chunk_1", "_to": "entities/ent_tramadol"},
        {"_from": "chunks/chunk_2", "_to": "entities/ent_fluoxetine"},
        {"_from": "chunks/chunk_2", "_to": "entities/ent_cyp2d6"},
        {"_from": "chunks/chunk_3", "_to": "entities/ent_tramadol"},
        {"_from": "chunks/chunk_3", "_to": "entities/ent_cyp2d6"},
        {"_from": "chunks/chunk_3", "_to": "entities/ent_serotonin_syndrome"},
    ]
    mentions_col.insert_many(mentions)

    # 4. Link Entities -> Entities (Knowledge Graph Relations)
    relations = [
        {
            "_from": "entities/ent_fluoxetine",
            "_to": "entities/ent_cyp2d6",
            "relation_type": "POTENTLY_INHIBITS",
            "confidence": 0.96,
            "source_chunk": "chunk_2",
            "evidence": "Fluoxetine inhibits hepatic enzyme CYP2D6 capacity by over 80%.",
        },
        {
            "_from": "entities/ent_tramadol",
            "_to": "entities/ent_cyp2d6",
            "relation_type": "METABOLIZED_BY",
            "confidence": 0.98,
            "source_chunk": "chunk_3",
            "evidence": "Prodrug activation of Tramadol depends on O-demethylation via CYP2D6.",
        },
        {
            "_from": "entities/ent_cyp2d6",
            "_to": "entities/ent_serotonin_syndrome",
            "relation_type": "BLOCKADE_PRECIPITATES",
            "confidence": 0.92,
            "source_chunk": "chunk_3",
            "evidence": "Blocked CYP2D6 causes parent Tramadol accumulation, precipitating Serotonin Syndrome.",
        },
    ]
    relations_col.insert_many(relations)

