# Project Quick-Reference: PhD Interview Prep
*University of Edinburgh — Dr. Aneta Mikulasova*

---

## 1. GNN-Based Somatic Mutation Classification (gnn_cancer)
*The published paper project*

**What it is**: A research system that uses graph neural networks to classify cancer driver mutations in breast cancer, built on real TCGA patient data.

**The core insight**: Somatic mutations don't act in isolation — they exist in a network of protein interactions, pathway memberships, and co-expression relationships. Encoding that relational structure as a graph lets GNNs learn features that flat feature vectors can't capture.

**The graph**: 967,189 nodes (patient samples), 2,134,841 edges across three types — protein-protein interactions (STRING), pathway co-membership (KEGG/Reactome), and co-expression correlations from the data itself. 19 multi-omics features per node (mutation counts, expression, CNV, protein abundance).

**The three architectures**:
- **GAT** (Graph Attention Networks): Learns which neighbors to attend to — interpretable, 100% recall on test set
- **GraphSAGE**: Inductive learning, generalizes to new patients — best overall, 99.87% F1 (+6.76% over baseline)
- **GCN** (Graph Convolutional Networks): Spectral diffusion, foundational architecture — solid baseline

**The hard part**: 50,903:1 class imbalance (19 positive samples, 967,170 negative). Solved with Focal Loss + SMOTE + stratified splits.

**Results**: GraphSAGE achieved 99.87% accuracy, 100% precision, 99.74% recall — exceeded the paper baseline by ~6% across all metrics. Perfect test confusion matrix (120/120).

**Why relevant to Edinburgh**: Complex genome rearrangements in myeloma (chromothripsis, chromoplexy) are literally graph structures. Breakpoints are nodes, rearrangement events are edges. This is the same architectural family.

---

## 2. Cancer Genomics Analysis Suite
*The full-stack research platform*

**What it is**: An end-to-end platform for cancer genomics research, integrating 127 data sources into a unified system with mutation analysis, biomarker discovery, drug-gene analysis, ML outcome prediction, and interactive dashboards.

**Why it was built**: Cancer genomics research typically requires stitching together dozens of incompatible tools. This collapses that fragmented stack into one coherent system with shared data models and a consistent API.

**Key architectural choices**:
- Flask + Dash for the web layer
- Celery + Redis for async long-running analyses
- **Neo4j** for mutation relationship graphs (same graph-first thinking as the GNN project)
- **Kafka** for real-time streaming mutation detection
- Kubernetes + Docker for production deployment

**Data scale**: 127 sources including TCGA, GEO, COSMIC, ClinVar, OncoKB, PharmGKB, CCLE, GDSC. Master orchestrator handles parallel collection with retry logic and rate limiting. Performance: 10,000+ mutations/second, 1M+ clinical records/hour.

**ML layer**: Survival prediction (Cox proportional hazards, Random Survival Forest), treatment response classification, risk stratification — with SHAP/LIME explanations and Optuna hyperparameter tuning.

**Workflow orchestration**: Unified executor abstracts Nextflow, Snakemake, R/Seurat, HDOCK, HADDOCK — all called via the same interface.

**Why relevant to Edinburgh**: The data integration infrastructure (multi-omics across genomics + epigenomics + clinical) directly maps onto what Aneta's lab does. The Kafka-based streaming architecture was built with the same real-time genomic analysis use case in mind.

---

## 3. Cancer Biomarker Identifier
*The focused biomarker discovery app*

**What it is**: A full-stack web app with a React frontend and FastAPI backend, purpose-built to take a gene expression matrix + clinical metadata and output a validated, clinically annotated biomarker list.

**The pipeline** (fully automated):
Upload → Quality Control → Normalization (ComBat batch correction) → Differential Expression (FDR-corrected) → Consensus Feature Selection → Model Training + Cross-Validation → Permutation Testing → SHAP Explainability → Pathway Analysis (GSEA) → Clinical Annotation (COSMIC, ClinVar, OncoKB) → HTML/PDF Report

**Consensus feature selection**: Uses three independent method classes (filter: variance/correlation, wrapper: RFE, embedded: LASSO/Ridge) simultaneously and takes the consensus. Genes selected by all three are the most trustworthy candidates.

**Real-time progress**: WebSocket connections stream live updates to the frontend during long analyses — critical UX when a run takes 10–30 minutes.

**SHAP explainability**: Not just accuracy — each biomarker comes with a SHAP-based explanation of why the model flagged it. Essential for clinical credibility.

**Clinical annotation**: Live API calls to COSMIC, ClinVar, and OncoKB link each candidate biomarker to known clinical evidence and treatment implications.

**Federated learning module**: Allows training across multiple institutions without sharing raw patient data — only gradients shared. Directly addresses privacy constraints in clinical genomics.

**Why relevant to Edinburgh**: The outcome prediction framing, the survival analysis component, and the clinical annotation workflow are directly transferable to predicting myeloma outcomes from genomic data.

---

## 4. Multi-Omics Analysis Suite
*The general-purpose platform*

**What it is**: A unified bioinformatics platform supporting 50+ omics disciplines (genomics, transcriptomics, proteomics, metabolomics, epigenomics, metagenomics, spatial transcriptomics, single-cell, radiomics, and ~40 more) through one consistent plugin-based architecture.

**The plugin system**: Every omics module inherits from `OmicsModuleBase` — standardized `load_data`, `preprocess`, `quality_control`, `normalize`, `analyze`, `visualize` interface. The `OmicsRegistry` auto-discovers modules on startup. Adding a new omics type requires only writing a conforming module; the core never changes.

**Bioinformatics core (built from scratch)**:
- Sequence alignment: Needleman-Wunsch, Smith-Waterman, FM-index/BWT (same basis as BWA)
- Genome assembly: De Bruijn graph (short reads), OLC (long reads), hybrid, Hi-C scaffolding
- File parsers: FASTA, FASTQ, BAM/SAM, VCF, GFF/GTF with lazy loading

**ML/AI engine**: Traditional ML (RF, XGBoost, LightGBM), deep learning (PyTorch), GNNs (PyTorch Geometric — GCN, GAT, GraphSAGE), AutoML with Optuna, SHAP + LIME explainability, survival analysis.

**Multi-omics integration**: Early, intermediate, and late fusion strategies; network-based integration via Neo4j; dimensionality reduction (PCA, UMAP, t-SNE).

**Interfaces**: FastAPI (async REST) + GraphQL (Strawberry) + WebSocket + CLI (`moas` command) + React/TypeScript frontend + Dash dashboards.

**Data collection**: Master Orchestrator for 50+ public databases (TCGA, GEO, COSMIC, PDB, KEGG, STRING, UniProt, MetaboLights, DrugBank, etc.) with retry, rate limiting, caching, parallel execution.

**Why relevant to Edinburgh**: Aneta's recent work integrates multi-omics data (WGS + epigenomics) across the complete myeloma genome. This suite was built to handle exactly that kind of heterogeneous integration problem at scale.

---

## The Thread That Connects All Four

| Project | Layer | Key Contribution |
|---|---|---|
| GNN Cancer | Science | Proves graph architectures are optimal for cancer mutation analysis |
| Cancer Genomics Suite | Platform | Productizes that into clinical-scale data integration + ML |
| Biomarker Identifier | Methodology | Rigorous, explainable, clinically annotated biomarker discovery |
| Multi-Omics Suite | Generalization | Extends the architecture to all omics disciplines |

The Edinburgh project sits squarely in the intersection: it needs the graph-based thinking from the GNN project, the multi-omics integration from the suite, and the outcome-prediction rigor from the biomarker identifier.
