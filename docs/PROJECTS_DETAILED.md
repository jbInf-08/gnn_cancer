# Project Deep-Dive Reference: Prepared for PhD Interview with Dr. Aneta Mikulasova
*University of Edinburgh — ML Prediction of Multiple Myeloma Outcomes*

---

## 1. GNN-Based Somatic Mutation Classification (gnn_cancer)

### What It Is
A graph neural network research system for classifying cancer driver mutations in breast cancer patient cohorts. It implements and compares three GNN architectures — Graph Attention Networks (GAT), GraphSAGE, and Graph Convolutional Networks (GCN) — and was the basis for a published paper. The central thesis: somatic mutations and their molecular context form a natural graph structure, and GNNs are better suited to this domain than flat feature-vector approaches.

### The Biological Problem It Solves
Distinguishing *driver* mutations (those that confer selective growth advantage and actively cause cancer) from *passenger* mutations (neutral bystanders) is one of the hardest open problems in cancer genomics. The signal is buried in extreme class imbalance — only 19 positive (driver) samples among 967,189 nodes in the final graph, a 50,903:1 ratio. Traditional ML collapses to predicting all negatives under those conditions.

### Data Sources
- **TCGA** (The Cancer Genome Atlas): Primary source — somatic mutation calls, gene expression, CNV
- **CPTAC** (Clinical Proteomic Tumor Analysis Consortium): Protein abundance measurements
- **STRING database**: Protein-protein interaction (PPI) edges with confidence > 0.7
- **KEGG & Reactome**: Pathway co-occurrence edges
- **Co-expression analysis**: Pearson correlation edges (|r| > 0.7) derived from the expression data itself

All data is real, authenticated clinical genomics data — no synthetic augmentation of the graph structure.

### Graph Construction
- **967,189 nodes**: Each represents a patient sample
- **2,134,841 edges** across three types: PPI interactions, pathway co-membership, co-expression
- **19 features per node**: Mutation counts, gene expression statistics (mean/std/max/min), CNV measurements, protein abundance, driver mutation label
- Nodes without sufficient data were excluded rather than imputed, maintaining biological fidelity

The reason for the multi-edge-type approach: PPI edges encode experimentally validated physical interactions; pathway edges encode functional relationships; co-expression edges are data-driven and capture patient-cohort-specific patterns. Together they produce a richer relational representation than any single source.

### The Three GNN Architectures

**GAT (Graph Attention Network)**
- 3 layers, 8 attention heads per layer, 64 hidden units
- Attention mechanism: each node learns different weights for each neighbor, which maps directly onto the biological intuition that some molecular interactions matter more than others in a given context
- ELU activation, 0.5 dropout on attention coefficients
- Best result: 100% recall (zero false negatives), ROC-AUC 99.98% — ideal for a clinical screening scenario where missing a true driver mutation is the worst outcome

**GraphSAGE (Graph Sample and Aggregate)**
- 3 layers, mean neighborhood aggregation, samples 25 neighbors per node
- Inductive learning: can generalize to new nodes not seen at training time — this matters for extending to new patient cohorts without retraining
- Skip/residual connections for gradient stability
- Best overall result: 99.87% accuracy, 99.87% F1, 100% precision — exceeded the paper baseline by ~6% across metrics
- Achieved perfect test confusion matrix: 120/120 correct

**GCN (Graph Convolutional Network)**
- Spectral convolution using normalized graph Laplacian
- Theoretical basis: signal diffusion through a graph, analogous to how mutations propagate effects through interaction networks
- Foundational architecture; ROC-AUC 94.60%, PR-AUC 86.55%

### Class Imbalance Strategy — The Technical Core
Three-layer approach:
1. **Focal Loss**: Adaptive down-weighting of easy negatives; parameters α=1.0, γ=2.0, β=0.25 — focuses training signal on the hard examples near the decision boundary
2. **SMOTE**: Synthetic Minority Oversampling Technique applied at training time to balance the positive/negative ratio
3. **Stratified splits**: 70/15/15 train/val/test with stratification to maintain class distribution — essential when you have 19 positive samples

Cosine annealing with warm restarts (T₀=10, T_mult=2) was chosen over standard learning rate decay because the restarts allow the optimizer to escape local minima — particularly important when the loss landscape is dominated by imbalanced class signals.

### Performance Results

| Metric | Paper Baseline | Our Best (GraphSAGE) |
|---|---|---|
| Accuracy | 93.8% | 99.87% |
| F1-Score | 93.1% | 99.87% |
| Precision | 93.4% | 100.00% |
| Recall | 92.8% | 99.74% |
| ROC-AUC | N/A | 99.76% |
| PR-AUC | N/A | 99.87% |

12 evaluation metrics were tracked in total — far beyond accuracy alone — because accuracy is misleading on imbalanced data (a naive all-negative classifier would achieve 99.998%).

### Why GNNs for This Problem
Chromosomal rearrangements and somatic mutations don't act in isolation. A mutation in gene A has context: its interaction partners, its pathway membership, the co-expression neighborhood of the affected gene. Flat feature vectors discard this relational structure. Graph neural networks preserve and exploit it. This is the same architectural argument that applies to Aneta's work on complex genome rearrangements in myeloma — breakpoints and rearrangement events are nodes and edges; their topology encodes biology.

### Design Philosophy
The project was built with scientific reproducibility as a first principle: deterministic seeding throughout, no data leakage between splits, all hyperparameters documented, 100% real data. The goal was to produce results that could survive peer review and extend to clinical use, not just maximize a held-out leaderboard number.

---

## 2. Cancer Genomics Analysis Suite (cancer_genomics_analysis_suite)

### What It Is
A full-stack, production-grade bioinformatics platform for cancer genomics research and clinical decision support. The goal was to build a single unified system that handles the entire analytical workflow — from raw data ingestion across 127 external sources to ML outcome prediction to interactive clinical dashboards — in one coherent architecture instead of the fragmented toolchain that most cancer genomics labs run.

### The Problem It Solves
Cancer genomics research typically requires stitching together dozens of incompatible tools: separate pipelines for variant calling, gene expression, clinical annotation, drug analysis, and visualization. Each tool has different data formats, dependencies, and interfaces. The result is brittle, hard to reproduce, and difficult to extend. This suite collapses that stack into one application with shared data models, consistent APIs, and a unified orchestration layer.

### Architecture
The system uses a hybrid architecture: Flask (REST API core) + Dash (interactive dashboards) on top of a polyglot data layer. The key architectural decision was separating concerns sharply:

- **Web layer**: Flask with blueprints handles API routing; Gunicorn for production serving
- **Task layer**: Celery workers with Redis as the broker handle all long-running analyses asynchronously — critical when analyses can run for hours
- **Data layer**: PostgreSQL (primary relational store), Neo4j (graph analytics for mutation networks), Redis (caching + task queue), Kafka (streaming mutation detection in real-time)
- **Orchestration layer**: A unified WorkflowExecutor abstracts Nextflow, Snakemake, Perl, R (Seurat), HDOCK, HADDOCK, SATurn, and SeqAnt — so the application can invoke any of these via a single interface without caring about their internal APIs

The plugin registry system allows new analysis modules to be added at runtime without modifying core application code — important for a research platform that needs to grow with new methods.

### Data Ingestion at Scale
127 data sources organized into categories:
- Genomic: TCGA, GEO, ICGC, EGA, GDC, NCBI
- Clinical: SEER, NCDB, CDC
- Mutation annotation: COSMIC, ClinVar, OncoKB
- Drug/cell lines: CCLE, GDSC, NCI-60
- Drugs: 32 sources including PharmGKB, FDA FAERS, DrugBank, RxNorm, DailyMed

All collection goes through a Master Orchestrator + Base Collector framework with retry logic, rate limiting, parallel execution, and error handling. Performance target: 10,000+ mutations/second for mutation processing, 1M+ clinical records/hour ingestion.

### Key Analysis Modules

**Mutation Analysis**
- Real-time detection from streaming genomic data via Kafka
- Mutation Effect Predictor: estimates structural/functional impact
- Integration with external variant databases (COSMIC, ClinVar)
- Neo4j-based relationship graphs for mutation network analysis
- Supports VCF, BAM, FASTQ inputs

**Biomarker Discovery Engine**
- Statistical methods: t-tests, Mann-Whitney U, correlation, FDR multiple testing correction
- ML methods: Random Forest, SVM, Logistic Regression, XGBoost, LightGBM
- Feature selection: SelectKBest, mutual information, recursive feature elimination
- Validation: Cross-validation, ROC/AUC, independent dataset validation
- Multi-omics: genomics + transcriptomics + proteomics fusion

**ML Outcome Predictor**
- Survival prediction: Cox proportional hazards, Random Survival Forest
- Treatment response classification
- Risk stratification (high/medium/low)
- SHAP/LIME explanations for all predictions
- Optuna for automated hyperparameter tuning

**Graph Analytics (Neo4j)**
- Gene interaction networks
- Pathway analysis (KEGG/Reactome)
- Drug-gene-disease relationship mapping
- Patient cohort similarity grouping

### Technology Stack (Selected Key Choices)
- **Kafka**: Chose this for mutation detection streaming because it provides true fault-tolerant event streaming, not just message passing — important when you can't afford to miss a mutation event
- **Neo4j**: Chose graph database over relational for mutation networks because Cypher queries over biological relationships are orders of magnitude faster than JOIN-heavy SQL on networks with millions of edges
- **Celery + Redis**: Async task processing to keep the API responsive while long-running analyses execute in worker processes
- **Multi-ML library approach**: scikit-learn + XGBoost + LightGBM + PyTorch — different algorithms excel on different data shapes, so the system tries multiple and selects the best

### Deployment
Kubernetes-native with Helm charts, Terraform IaC for AWS/GCP/Azure, ArgoCD for GitOps CD. Monitoring via Prometheus + Grafana. The production configuration runs 2+ application replicas, multi-node PostgreSQL with replication, Kafka with 3+ broker replication. This was designed to run in a clinical environment where downtime has direct consequences.

### Why It Was Built This Way
The design principle was to build a system that a bioinformatician and a clinician could both use — the bioinformatician through the API and CLI, the clinician through Dash dashboards. The plugin architecture ensures that as new analytical methods are published, they can be added without redesigning the core. The security architecture (OAuth2, RBAC, TLS) was built in from the start because clinical genomic data requires it.

---

## 3. Cancer Biomarker Identifier (biomarker_identifier)

### What It Is
A production-grade, full-stack web application for identifying, validating, and analyzing cancer biomarkers from multi-omics data. Where the Cancer Genomics Suite is a broad research platform, this project is tightly focused: its single purpose is taking a gene expression matrix + clinical metadata as input and producing a ranked, validated list of candidate biomarkers with statistical confidence, ML evidence, SHAP interpretability, and clinical annotation — all in one automated pipeline.

### Architecture
Modern microservices design:
- **Backend**: FastAPI (async Python) + SQLAlchemy 2.0 with async PostgreSQL
- **Frontend**: React 18, TypeScript, Tailwind CSS, Recharts for visualization
- **Background jobs**: Celery 5.3.4 with Redis
- **Real-time updates**: WebSocket connections for live pipeline progress — analyses can take 10+ minutes and users need visibility
- **Database**: PostgreSQL 13+ with Alembic migrations; SQLite for development

The choice of FastAPI over Flask was deliberate: async/await throughout the backend means I/O-bound operations (database reads, external API calls to COSMIC/ClinVar/OncoKB) don't block each other. A standard Flask app would serialize those calls.

### The End-to-End Pipeline

1. **Upload**: Gene expression matrix (CSV/TSV, genes × samples), clinical metadata, optional mutation data
2. **Validation**: File format checking, dimension verification, data type checks before anything expensive runs
3. **Quality Control**: Sample/gene filtering, outlier detection, distribution analysis — with QC report generated
4. **Normalization**: Log transformation, quantile normalization, batch effect correction using ComBat algorithm
5. **Differential Expression**: t-tests, Welch's test, Mann-Whitney U, FDR correction (Benjamini-Hochberg), Cohen's d effect size
6. **Feature Selection (Consensus)**: This is the methodological core — uses multiple independent methods simultaneously:
   - Filter methods: variance filtering, correlation analysis
   - Wrapper method: Recursive Feature Elimination (RFE)
   - Embedded methods: LASSO and Ridge regularization
   - Consensus voting across methods with stability scores — reduces risk of a single selection method's biases
7. **Model Training**: Automated selection across Logistic Regression, SVM, Random Forest, XGBoost, LightGBM — stratified k-fold cross-validation, hyperparameter optimization, AUC as primary metric
8. **Permutation Testing**: Validates feature stability and significance against null distribution
9. **SHAP Explainability**: Global feature importance plots + local sample-level explanations — addresses the "why" question that pure accuracy doesn't
10. **Pathway Analysis**: GSEA and over-representation analysis (gseapy) to place biomarkers in biological context
11. **Clinical Annotation**: Real API calls to COSMIC, ClinVar, OncoKB to link each biomarker to known clinical evidence
12. **Report Generation**: HTML reports (Jinja2) + PDF (ReportLab) with all embedded visualizations

### Key Design Decisions

**Why consensus feature selection?** A single feature selection method can overfit to quirks of the data. Using three classes of methods (filter, wrapper, embedded) and taking the consensus reduces variance in the selection. The stability score tells you which biomarkers are consistently selected across methods — those are the ones you trust.

**Why SHAP at the model level?** In a clinical context, a black-box accuracy number is not sufficient. A clinician needs to know: "why is this gene flagged as a biomarker?" SHAP provides mathematically grounded feature attribution at the individual sample level, not just global averages.

**Why Celery for background jobs?** Some analyses — particularly permutation testing with thousands of iterations and cross-validation over many model types — take 10–30 minutes. HTTP requests can't hold a connection that long. Celery submits the job, returns a job ID immediately, and the frontend polls or receives WebSocket updates as progress happens.

**Federated learning module**: A distinct component that allows training biomarker models across multiple institutions without sharing raw patient data — only model gradients are shared. This directly addresses privacy constraints in clinical genomic research.

### The Clinical Annotation Layer
The system makes live API calls to:
- **COSMIC**: Somatic mutation database — links variants to known cancer genes
- **ClinVar**: Clinical significance annotations from clinical labs
- **OncoKB**: Precision oncology knowledge base — provides evidence level and treatment implications

A Clinical Decision Support component takes the annotated biomarkers and generates evidence-based recommendations. This was built because the goal isn't just discovering biomarkers in the abstract — it's producing results actionable for clinical decisions.

### Multi-tenancy and Enterprise Features
The system has database-level tenant isolation, which means it was designed from the start for deployment across multiple research institutions, each with their own data and users, sharing the same application infrastructure. This reflects a forward-looking design choice: building for eventual SaaS deployment or multi-site clinical consortium use.

---

## 4. Multi-Omics Analysis Suite (multi_omics_analysis_suite)

### What It Is
An enterprise-grade bioinformatics platform supporting 50+ omics disciplines in a unified, plugin-based architecture. If the Cancer Genomics Suite is deep on cancer-specific analysis, the Multi-Omics Suite is broad: it covers genomics, transcriptomics, proteomics, metabolomics, epigenomics, metagenomics, pharmacogenomics, lipidomics, spatial transcriptomics, single-cell multi-omics, radiomics, and ~35 additional specialized disciplines, all through one consistent API.

### The Architectural Problem It Solves
Each omics type has traditionally required its own tool ecosystem — samtools/GATK for genomics, Salmon/DESeq2 for transcriptomics, MaxQuant for proteomics, MetaboAnalyst for metabolomics. Integrating these in a single analysis workflow requires manual format conversions, environment management, and bespoke glue code. The Multi-Omics Suite replaces that fragmented approach with a unified abstraction layer.

### Core Architecture: The Plugin System
Every omics module inherits from `OmicsModuleBase`, an abstract base class that enforces a standard interface:

```
load_data()         # Load from files or databases
preprocess()        # Filter, clean, transform
quality_control()   # Validate and report metrics
normalize()         # Standardize values
analyze()           # Run statistical/ML analysis
visualize()         # Generate plots and figures
```

The `OmicsRegistry` (singleton) auto-discovers and registers all 50+ modules on application startup. This means adding a new omics type — say, a new flavor of spatial transcriptomics — requires only writing a module that implements the interface and placing it in the correct directory. The registry picks it up without any other changes.

**Why this design?** Supporting 50+ omics types with tight coupling would make the codebase unmaintainable. The plugin pattern inverts this: the core never changes when a new module is added, only the module directory grows.

### Data Flow
The pipeline architecture uses a `PipelineContext` object that carries data through a Directed Acyclic Graph (DAG) of `PipelineStep` objects. Each step:
- Receives the context (contains all prior intermediate results)
- Performs its operation
- Writes results back to the context
- Can be skipped conditionally (e.g., skip normalization if data is pre-normalized)

Real-time progress is streamed to the frontend via WebSocket as each step completes.

### Bioinformatics Core (Built From Scratch)
Rather than wrapping external tools for everything, the suite implements foundational algorithms directly:

**Sequence Processing**: DNA/RNA/Protein sequence classes with GC content, codon translation, reverse complement, Phred quality scoring, quality trimming

**Alignment Algorithms**:
- Needleman-Wunsch: Global pairwise alignment (O(mn) dynamic programming)
- Smith-Waterman: Local alignment for finding conserved regions
- FM-index and Burrows-Wheeler Transform for fast string search (same algorithmic basis as BWA/bowtie)

**Genome Assembly**:
- De Bruijn graph assembler for short reads (Eulerian path-finding)
- Overlap-Layout-Consensus (OLC) for long reads
- Hybrid assembly combining both
- Scaffolding strategies: paired-end, mate-pair, Hi-C (3D genome structure)
- Quality assessment: N50 statistics, BUSCO completeness

**File Format Parsers**: FASTA, FASTQ, GFF/GTF, BED, SAM/BAM, VCF, GenBank — with lazy loading for large files

The rationale for implementing these from scratch rather than wrapping samtools/bwa: the suite needs to run in environments where those binaries may not be available (cloud containers, Windows), and having pure-Python implementations means the suite is self-contained and portable.

### ML/AI Engine
The ML component is modular and sits independently of the omics modules:

**Model Types**:
- Traditional: Random Forest, XGBoost, LightGBM, SVM, Elastic Net
- Deep Learning (PyTorch): DNNs, CNNs, RNNs, attention mechanisms
- Graph Neural Networks (PyTorch Geometric): GCN, GAT, GraphSAGE — for biological network analysis
- Survival Analysis: Cox regression, Kaplan-Meier, log-rank tests

**AutoML Pipeline**: Systematic search over candidate models and hyperparameters (grid, random, Bayesian via Optuna). Returns best model + hyperparameters without manual tuning.

**Feature Engineering**: Statistical features, interaction features, polynomial features, domain-specific features (GC content, codon bias)

**Explainability**: SHAP and LIME integrated throughout — not as an afterthought but as a required output of every ML analysis step.

### Multi-Omics Data Integration
The integration module handles the methodologically hard problem of combining heterogeneous data types:
- **Early fusion**: Concatenate feature matrices before modeling — simple, loses some structure
- **Intermediate fusion**: Learn joint embeddings per omics type, then combine — preserves modality-specific patterns
- **Late fusion**: Train per-modality models, combine predictions — most robust to missing data
- **Network integration**: Pathway-based and network-based integration using the graph database (Neo4j)
- **Dimensionality reduction**: PCA, UMAP, t-SNE across integrated matrices

The choice of integration strategy depends on data availability and missingness patterns — the system exposes all three options rather than picking one.

### Data Collection: 50+ Public Databases
The Master Orchestrator manages parallel data collection from databases including TCGA, GEO, COSMIC, PDB (protein structures), KEGG, Reactome, STRING, BioGRID, UniProt, HMDB, MetaboLights, DrugBank, OncoKB, and others. Rate limiting, retry logic with exponential backoff, result caching, and dependency resolution are all handled by the orchestrator.

### Infrastructure and Interfaces
- **FastAPI** (async REST) + **Strawberry GraphQL** + **WebSocket** for real-time updates
- **Typer CLI** (`moas` command): `moas analyze genomics --input data.vcf`, `moas ml automl`, etc.
- **Dash dashboards**: Interactive visualization layer
- **React/TypeScript frontend**: Full UI
- **PostgreSQL** (structured data), **Neo4j** (biological networks), **Redis** (caching/Celery), **MinIO** (large binary data like alignments)
- **Kubernetes + Docker Compose** for deployment

### Why This Project Exists Alongside the Others
The Cancer Genomics Suite and Biomarker Identifier are purpose-built for specific cancer research workflows. The Multi-Omics Suite is the general-purpose platform that those projects' methodologies could eventually be ported onto. It also handles omics types — metabolomics, metagenomics, spatial transcriptomics — that aren't covered by the cancer-specific tools. The goal was to build something that could support the full breadth of a modern computational biology research group, not just cancer genomics specifically.

---

## Cross-Project Connecting Thread

These four projects form a coherent technical progression:

- The **GNN Cancer project** is the scientific core: it establishes that graph neural networks are the right architectural choice for cancer genomics classification, provides the empirical results, and demonstrates the value of multi-omics graph construction.
- The **Cancer Genomics Suite** productizes that insight into an end-to-end platform with clinical data integration, real-time processing, and deployment infrastructure.
- The **Biomarker Identifier** goes deep on one specific problem (biomarker discovery) with rigorous statistical methodology, clinical annotation, and explainability — the kind of output that actually moves from research to clinical use.
- The **Multi-Omics Suite** generalizes the architectural patterns across all omics disciplines, demonstrating that the approach scales beyond cancer genomics to the full space of modern molecular biology.

For the Edinburgh project specifically: the GNN architecture knowledge transfers directly. Complex chromosomal rearrangements in myeloma — chromothripsis, chromoplexy, templated insertions — are literally graph structures where breakpoints are nodes and rearrangement events are edges. GAT/GraphSAGE/GCN architectures are the natural fit. The multi-omics integration experience from all four projects is also directly relevant, given Aneta's ASH 2024 work integrating multi-omics data across the complete myeloma genome.
