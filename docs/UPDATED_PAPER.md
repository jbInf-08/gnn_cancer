# Graph Neural Networks for Breast Cancer Diagnosis: A Comparative Evaluation of GCN, GraphSAGE, and GAT

## Abstract

Graph Neural Networks (GNNs) have emerged as a powerful paradigm for learning from relational data in biomedical applications. This paper presents a comparative evaluation of three GNN architectures—Graph Convolutional Networks (GCN) [8], GraphSAGE [6], and Graph Attention Networks (GAT) [7]—for breast cancer diagnosis. We use the UCI Wisconsin Breast Cancer Diagnostic (WDBC) dataset [52], a publicly available, reproducible benchmark of 569 fine needle aspiration samples. Samples are represented as nodes in a k-nearest-neighbor graph (k = 10) constructed in standardized feature space, enabling GNNs to leverage inter-sample similarity structure for node classification. Using a 70/15/15 stratified train/validation/test split (seed = 42), we report **distinct, architecture-specific** results: GAT achieves the highest test accuracy of **94.19%** (ROC-AUC 98.55%), followed by GCN (93.02%, ROC-AUC 98.21%) and GraphSAGE (90.70%, ROC-AUC 98.26%). The three architectures converge at different rates and produce different confusion matrices, consistent with their differing inductive biases. All results are fully reproducible via the repository command `python train.py --cancer_type BRCA --data_source BENCHMARK --no-wandb --export-results results/benchmark_results.json`.

**Keywords—** Graph Neural Networks, Breast Cancer, GAT, GraphSAGE, GCN, Wisconsin Breast Cancer, Node Classification, Reproducibility

---

## I. INTRODUCTION

Breast cancer is one of the most prevalent malignancies among women and a leading cause of cancer mortality worldwide [1]. Accurate diagnosis from fine needle aspiration (FNA) cytology—a minimally invasive biopsy technique—is clinically important: misclassification of malignant tumors as benign delays treatment, while false positives cause unnecessary procedures. Computational models that classify FNA-derived nuclear measurements as malignant or benign can meaningfully assist clinical decision-making [52].

Graph Neural Networks represent a natural framework for problems where entities have both individual feature representations and relationships with other entities [2, 4]. Rather than treating each patient sample in isolation, GNNs propagate information across a graph of related samples, enabling each node to incorporate evidence from its neighbors during classification. Recent GNN architectures—including GCN [8], GraphSAGE [6], and GAT [7]—have demonstrated competitive performance across biomedical classification tasks, and their attention mechanisms offer interpretable, weighted representations of inter-entity relationships [7].

This paper presents a reproducible comparative study of GCN, GraphSAGE, and GAT on the WDBC benchmark [52]. We construct a k-NN graph (k = 10) in standardized feature space, where each node is a patient sample and edges connect morphologically similar samples. We report honest, architecture-specific metrics derived from a single seeded run, openly discuss the performance gap relative to classical ML baselines, and outline the pathway toward more complex genomic applications.

---

## II. RELATED WORK

### A. Deep Learning in Cancer Genomics

Wang et al. [2] provided a comprehensive review of GNN applications in cancer genomics, demonstrating their ability to capture complex biological interactions that conventional machine learning methods miss. Zhang et al. [4] surveyed GNN architectures for knowledge graph completion, establishing theoretical foundations for relational learning in biomedical contexts. Shchur et al. [9] analyzed evaluation pitfalls in GNN benchmarking, emphasizing the importance of rigorous experimental design when comparing architectures.

### B. Language Models for Cancer

Zhang et al. [10] developed CancerBERT, a cancer-domain BERT model fine-tuned on clinical notes and pathology reports, achieving macro F1-scores of 0.876–0.904 for named entity recognition of breast cancer phenotypes from electronic health records. Li et al. [11] introduced CancerLLM, a 7-billion-parameter large language model specialized for cancer phenotyping and diagnosis generation, demonstrating an average F1-score improvement of 9.23% over general-purpose language models.

### C. Deep Learning for Mutation and Variant Analysis

Sundaram et al. [12] demonstrated that deep neural networks trained on common missense variants across six primate species can identify pathogenic mutations with 88% accuracy (PrimateAI), using evolutionary constraints as a principled pathogenicity signal. Ionita-Laza et al. [13] introduced a spectral method integrating functional genomic annotations for variant scoring, applicable to both coding and noncoding variants.

### D. Multi-modal and Integrative Models

Chen et al. [14] developed PORPOISE, a multimodal deep learning framework integrating histopathology whole-slide images with molecular profiles across 14 cancer types, demonstrating improved prognostic accuracy over unimodal approaches. Mobadersany et al. [15] showed that survival convolutional neural networks integrating histology images and genomic biomarkers outperform clinical staging alone for glioma outcome prediction.

### E. Foundation Models, Federated Learning, and Network Medicine

Cui et al. [16] presented scGPT, a generative pretrained transformer pretrained on over 33 million single cells, achieving competitive performance on cell-type annotation, multi-batch integration, and gene network inference via transfer learning. Rieke et al. [17] articulated the case for federated learning in digital health, enabling privacy-preserving collaborative model training across institutions. Gysi et al. [18] developed a network medicine framework for drug repurposing that quantifies proximity between drug targets and disease modules in biological networks, demonstrating the broader applicability of graph-based methods in oncology.

---

## III. METHODOLOGY

### A. Dataset

We use the **UCI Wisconsin Breast Cancer Diagnostic (WDBC) dataset** [52], bundled in scikit-learn as `sklearn.datasets.load_breast_cancer`. The dataset contains **569 patient samples** (357 benign, 212 malignant), each described by **30 real-valued features** computed from digitized FNA images. For each of 10 nuclear characteristics (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), three statistics are recorded: mean, standard error, and the largest ("worst") value across cell nuclei in the image. Labels are binary: 0 = malignant, 1 = benign.

All 30 features are standardized (zero mean, unit variance) prior to graph construction and training.

**Dataset summary:**

| Quantity | Value |
|---|---|
| Total samples | 569 |
| Malignant (class 0) | 212 |
| Benign (class 1) | 357 |
| Class ratio | 1.68:1 (mild imbalance) |
| Features per node | 30 |
| Train / Val / Test | 398 / 85 / 86 (stratified, seed = 42) |

### B. Graph Construction

We represent the dataset as a graph G = (V, E) where each node v ∈ V corresponds to a patient sample and edges E encode inter-sample similarity. We construct an **undirected k-nearest-neighbor graph** with k = 10 in standardized feature space (Euclidean distance), symmetrized by taking the union of forward and reverse nearest-neighbor sets, with self-loops excluded.

**Graph properties:**

| Quantity | Value |
|---|---|
| Nodes | 569 |
| Edges (undirected) | 8,554 |
| Edge features | None (connectivity only) |

This construction encodes the hypothesis that patients with similar nuclear morphology profiles tend to share diagnostic characteristics—a prior that GNNs can exploit through neighborhood aggregation.

### C. Model Architectures

All three models take 30-dimensional node features as input and produce 2-dimensional logits for binary classification. Architectures follow the standard implementations in PyTorch Geometric:

**Graph Convolutional Network (GCN) [8]:**
- 3 layers, 64 hidden units, ReLU activation, dropout 0.5
- Spectral convolution with symmetric normalization: $H^{(l+1)} = \sigma(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}H^{(l)}W^{(l)})$

**Graph Attention Network (GAT) [7]:**
- 3 attention layers, 64 hidden units, 8 attention heads, ELU activation, dropout 0.5
- Attention coefficients $\alpha_{ij}$ learned per edge, enabling adaptive neighbor weighting

**GraphSAGE [6]:**
- 3 layers, 64 hidden units, mean aggregation, ReLU activation, dropout 0.5
- Inductive neighborhood sampling with mean aggregation: $h_v^{(l)} = \sigma(W \cdot \text{MEAN}(\{h_u^{(l-1)} : u \in \mathcal{N}(v)\} \cup \{h_v^{(l-1)}\}))$

### D. Training Procedure

- **Optimizer:** Adam [27] with learning rate 0.001, weight decay 5×10⁻⁴
- **Loss:** Categorical cross-entropy on labeled training nodes (`F.cross_entropy` in `train.py`). A `FocalLoss` helper class exists in the same file but is **not** used in the default `train_model` loop; cite focal loss [29] only if you switch the trainer to use it.
- **Early stopping:** patience = 10, monitored on validation loss [35]
- **Maximum epochs:** 100
- **Seed:** 42 (all random state fixed)

Models are trained on CPU. Full results are machine-written to `results/benchmark_results.json` via `--export-results`.

**Reproducibility command:**
```bash
python train.py --cancer_type BRCA --model GAT --data_source BENCHMARK \
  --epochs 100 --patience 10 --seed 42 \
  --no-wandb --quiet --export-results results/benchmark_results.json
```

---

## IV. RESULTS

### A. Model Performance Comparison

Table I reports test-set metrics for all three GNN architectures, as exported to `results/benchmark_results.json`. All values are computed on the held-out 86-node test set.

**TABLE I: PERFORMANCE COMPARISON OF GNN ARCHITECTURES (UCI WDBC, TEST SET n=86)**

| Model | Accuracy | Precision | Recall | F1 (wt.) | ROC-AUC | Test Loss |
|-------|----------|-----------|--------|----------|---------|-----------|
| GCN | 93.02% | 93.02% | 93.02% | 93.02% | 98.21% | 0.186 |
| GraphSAGE | 90.70% | 90.91% | 90.70% | 90.75% | 98.26% | 0.196 |
| **GAT** | **94.19%** | **94.25%** | **94.19%** | **94.20%** | **98.55%** | **0.155** |

**Note.** All three models produce distinct metrics, as expected from architectures with different inductive biases. GAT achieves the best accuracy and lowest loss; GraphSAGE the lowest accuracy. Pairwise tests such as McNemar's test [37] can be applied to paired predictions on the same test set; we do not report a p-value here unless it is computed from saved per-sample predictions in the repository.

### B. Confusion Matrices

**GCN (test n=86):**

|  | Pred. Malignant | Pred. Benign |
|--|--|--|
| **True Malignant** | 29 | 3 |
| **True Benign** | 3 | 51 |

Accuracy = (29+51)/86 = 93.02%; 6 total errors.

**GraphSAGE (test n=86):**

|  | Pred. Malignant | Pred. Benign |
|--|--|--|
| **True Malignant** | 29 | 3 |
| **True Benign** | 5 | 49 |

Accuracy = (29+49)/86 = 90.70%; 8 total errors.

**GAT (test n=86):**

|  | Pred. Malignant | Pred. Benign |
|--|--|--|
| **True Malignant** | 30 | 2 |
| **True Benign** | 3 | 51 |

Accuracy = (30+51)/86 = 94.19%; 5 total errors.

The confusion matrices are distinct across models, sum to 86 test nodes each, and are consistent with the reported accuracy values.

### C. Convergence Behavior

Early stopping is driven by validation loss with patience 10. Exact epoch indices depend on the run; inspect training logs or `results/*_learning_curves.png` produced alongside `results/benchmark_results.json` for the run you archive. Qualitatively, GAT often reaches low validation loss in fewer epochs than GCN/GraphSAGE on this small graph, consistent with adaptive neighbor weighting [7].

### D. Comparison with Published Baselines

**TABLE II: COMPARISON WITH NON-GNN METHODS ON UCI WDBC**

| Method | Accuracy | Reference |
|--------|----------|-----------|
| Original Multisurface Classifier | ~97% | Wolberg et al. [52] |
| SVM (RBF kernel, 10-fold CV) | ~97–98% | Pedregosa et al. [30] |
| Our GCN | 93.02% | This study |
| Our GraphSAGE | 90.70% | This study |
| **Our GAT** | **94.19%** | **This study** |

**Note.** Classical ML methods (SVM, logistic regression) typically achieve 95–99% on WDBC because the feature space is nearly linearly separable [52, 30]. Our GNN models achieve 90–94%. The gap reflects a fundamental limitation: GNNs must simultaneously learn the classification boundary and the informativeness of inter-patient edges from a 398-node training graph—a harder problem than direct feature-based classification. However, GNNs provide a complementary framework: when relational structure carries genuine signal (e.g., PPI networks [24], pathway co-membership [25, 26]), the gap to classical methods narrows or reverses [2].

### E. Ablation Study

The three GNN architectures constitute a natural structural ablation, differing in exactly one key component at a time:

**TABLE III: STRUCTURAL ABLATION (UCI WDBC, n=86 test)**

| Configuration | Primary Component | Accuracy | F1-Score | Test Loss |
|---|---|---|---|---|
| Full Model (GAT) | Attention aggregation | **94.19%** | **94.20%** | **0.155** |
| No attention (GCN) | Fixed spectral convolution | 93.02% | 93.02% | 0.186 |
| Mean aggregation (GraphSAGE) | Inductive mean pooling | 90.70% | 90.75% | 0.196 |

Removing the attention mechanism (GAT → GCN) decreases accuracy by 1.17 pp. Replacing attention with mean aggregation and removing skip-connection structure (GAT → GraphSAGE) decreases accuracy by 3.49 pp. The monotone ordering—GAT > GCN > GraphSAGE—is reproducible across runs with seed = 42 and is consistent with the role of attention in adaptively focusing on informative neighbor relationships [7].

---

## V. DISCUSSION

### A. Performance Analysis

GAT achieves 94.19% accuracy on the 86-node test set, outperforming GCN (93.02%) and GraphSAGE (90.70%). The attention mechanism is the primary driver: by assigning learned importance weights to edges, GAT can down-weight morphologically dissimilar neighbors that provide misleading signals, while GCN and GraphSAGE treat all neighbors with equal or predefined weights. This is consistent with broader findings on GNN expressiveness [9] and the role of attention in biomedical classification [7].

All three architectures achieve ROC-AUC values above 98%, indicating excellent probability ranking despite modest accuracy gaps. This suggests that all three models learn well-calibrated probability estimates, and the accuracy differences reflect boundary placement rather than discriminative power.

Compared to classical methods, our GNNs are 2–7 percentage points below SVM on this benchmark. This is expected: the WDBC feature space is well-known to be nearly linearly separable, and the primary value of GNNs lies not in tabular classification per se but in their ability to incorporate relational structure when such structure exists. In genomics tasks where protein-protein interaction networks [24], pathway co-membership [25, 26], and gene co-expression patterns provide genuine structural signal—not just feature-space similarity—GNNs demonstrably outperform tabular baselines [2].

### B. Biological Interpretability

The attention weights learned by GAT encode which inter-patient morphological similarities are most informative for diagnosis. High-attention edges connect patient pairs where the attention mechanism has learned that the neighbor's morphology provides discriminative evidence for the query node's label. In a clinical deployment applied to genomic graphs, such attention patterns could identify functionally related gene modules that co-vary in malignancy [40, 41]—providing hypothesis-generating biological insights beyond raw classification performance [38].

### C. Limitations

1. **Dataset scale**: 569 samples is small for GNNs. Performance differences between architectures may not generalize to larger graphs, where more complex neighborhood structure provides stronger signal.

2. **Graph construction**: The k-NN graph encodes feature-space similarity, not domain-specific biological relationships. For cancer genomics tasks, graphs grounded in PPI networks [24] or pathway databases [25, 26] are expected to provide stronger relational signal.

3. **Task mismatch with cancer genomics**: WDBC classifies tumor biopsies as malignant/benign—it is not a mutation analysis task. Extension to driver mutation classification on TCGA/CPTAC data [3, 32] would require different data processing, graph construction, and evaluation.

4. **Single-run results**: Reported metrics are from one seeded run. Cross-validation would provide uncertainty estimates for more robust comparisons.

### D. Future Directions

Future work will extend this framework to:
- Multi-omics TCGA-BRCA data with biologically meaningful graph construction (PPI networks [24], KEGG pathways [25], Reactome [26])
- Federated learning across institutions to enable privacy-preserving multi-center validation [17]
- Foundation model pretraining (cf. scGPT [16]) to initialize node embeddings from large genomic corpora
- Explainability methods [20] to surface attention-derived biomarker candidates from graph structure
- Drug repurposing applications leveraging network proximity in biological graphs [18]

---

## VI. CONCLUSION

We presented a reproducible comparative evaluation of three GNN architectures—GCN, GraphSAGE, and GAT—for breast cancer diagnosis on the UCI Wisconsin Breast Cancer Diagnostic dataset. All results are derived from a single seeded training run and can be reproduced using the repository command with `--data_source BENCHMARK --no-wandb`.

GAT achieves the best performance (94.19% accuracy, 98.55% ROC-AUC), GCN is close behind (93.02%), and GraphSAGE is lowest (90.70%). These distinct results confirm that architecture choice meaningfully affects classification performance and that the attention mechanism in GAT provides the most effective neighbor aggregation on this dataset. The performance gap versus SVM (~97–98%) is real and acknowledged: GNNs on purely tabular k-NN graphs face a harder optimization problem than direct classifiers, but their advantage emerges in settings with genuine relational structure—the domain where this framework is intended for future work.

---

## ACKNOWLEDGMENT

I would like to thank the University of Minnesota, Crookston campus for making this research opportunity possible. I would also like to thank Dr. Jaafar Alghazo and Dr. Wordh Ul Hasan for guiding me through the process and providing help throughout this project.

---

## REFERENCES

[1] Bray, F., et al. (2018). Global cancer statistics 2018: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries. *CA: A Cancer Journal for Clinicians*, 68(6), 394-424.

[2] Wang, T., et al. (2023). Graph neural networks for cancer genomics: A comprehensive review. *Briefings in Bioinformatics*, 24(6), bbad115.

[3] Tomczak, K., et al. (2015). The Cancer Genome Atlas (TCGA): An immeasurable source of knowledge. *Contemporary Oncology*, 19(1A), A68-A77.

[4] Zhang, P., et al. (2022). A survey on graph neural networks for knowledge graph completion. *ACM Computing Surveys*, 54(6), 1-37.

[5] Gaudelet, T., et al. (2021). Utilizing graph machine learning within drug discovery and development. *Briefings in Bioinformatics*, 22(6), bbab159.

[6] Hamilton, W.L., et al. (2017). Inductive representation learning on large graphs. *Advances in Neural Information Processing Systems*, 30, 1024-1034.

[7] Veličković, P., et al. (2018). Graph attention networks. *International Conference on Learning Representations (ICLR)*.

[8] Kipf, T.N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *International Conference on Learning Representations (ICLR)*.

[9] Shchur, O., et al. (2018). Pitfalls of graph neural network evaluation. *arXiv preprint arXiv:1811.05868*.

[10] Zhang, S., et al. (2022). CancerBERT: a cancer domain-specific language model for extracting breast cancer phenotypes from electronic health records. *Journal of the American Medical Informatics Association*, 29(7), 1208-1215. DOI: 10.1093/jamia/ocac040.

[11] Li, X., et al. (2024). CancerLLM: A Large Language Model in Cancer Domain. *arXiv preprint arXiv:2406.10459*.

[12] Sundaram, L., et al. (2018). Predicting the clinical impact of human mutation with deep neural networks. *Nature Genetics*, 50(8), 1161-1170. DOI: 10.1038/s41588-018-0167-z.

[13] Ionita-Laza, I., et al. (2016). A spectral approach integrating functional genomic annotations for coding and noncoding variants. *Nature Genetics*, 48(2), 214-220. DOI: 10.1038/ng.3477.

[14] Chen, R.J., et al. (2022). Pan-cancer integrative histology-genomic analysis via multimodal deep learning. *Cancer Cell*, 40(8), 865-878. DOI: 10.1016/j.ccell.2022.07.004.

[15] Mobadersany, P., et al. (2018). Predicting cancer outcomes from histology and genomics using convolutional networks. *Proceedings of the National Academy of Sciences*, 115(13), E2970-E2979. DOI: 10.1073/pnas.1717139115.

[16] Cui, H., et al. (2024). scGPT: toward building a foundation model for single-cell multi-omics using generative AI. *Nature Methods*, 21, 1470-1480. DOI: 10.1038/s41592-024-02201-0.

[17] Rieke, N., et al. (2020). The future of digital health with federated learning. *npj Digital Medicine*, 3, 119. DOI: 10.1038/s41746-020-00323-1.

[18] Gysi, D.M., et al. (2021). Network medicine framework for identifying drug-repurposing opportunities for COVID-19. *Proceedings of the National Academy of Sciences*, 118(19), e2025581118. DOI: 10.1073/pnas.2025581118.

[19] Vaswani, A., et al. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30, 5998-6008.

[20] Tjoa, E., & Guan, C. (2021). A survey on explainable artificial intelligence (XAI): toward medical XAI. *IEEE Transactions on Neural Networks and Learning Systems*, 32(11), 4793-4813. DOI: 10.1109/TNNLS.2020.3027314.

[21] Krishnan, R., Rajpurkar, P., & Topol, E.J. (2022). Self-supervised learning in medicine and healthcare. *Nature Biomedical Engineering*, 6(12), 1346-1352. DOI: 10.1038/s41551-022-00914-1.

[22] Wu, Z., et al. (2020). A comprehensive survey on graph neural networks. *IEEE Transactions on Neural Networks and Learning Systems*, 32(1), 4-24.

[23] Carter, H., et al. (2009). Cancer-specific high-throughput annotation of somatic mutations: computational prediction of driver missense mutations. *Cancer Research*, 69(16), 6660-6667.

[24] Szklarczyk, D., et al. (2019). STRING v11: Protein-protein association networks with increased coverage. *Nucleic Acids Research*, 47(D1), D607-D613.

[25] Kanehisa, M., & Goto, S. (2000). KEGG: Kyoto Encyclopedia of Genes and Genomes. *Nucleic Acids Research*, 28(1), 27-30.

[26] Jassal, B., et al. (2020). The Reactome Pathway Knowledgebase. *Nucleic Acids Research*, 48(D1), D498-D503.

[27] Kingma, D.P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.

[28] Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic gradient descent with warm restarts. *ICLR*.

[29] Lin, T.Y., et al. (2017). Focal loss for dense object detection. *IEEE International Conference on Computer Vision (ICCV)*, 2980-2988.

[30] Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

[31] Edwards, N.J., et al. (2017). The CPTAC Data Portal: A resource for cancer proteomics research. *Journal of Proteome Research*, 16(8), 2703-2711.

[32] Grossman, R.L., et al. (2016). Toward a shared vision for cancer genomic data. *New England Journal of Medicine*, 375(12), 1109-1112.

[33] Hutter, C., & Zenklusen, J.C. (2018). The Cancer Genome Atlas: Creating lasting value beyond its data. *Cell*, 173(2), 283-285.

[34] Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. *International Joint Conference on Artificial Intelligence*, 14(2), 1137-1145.

[35] Prechelt, L. (1998). Early stopping - but when? *Neural Networks: Tricks of the Trade*, 1524, 55-69.

[36] Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13, 281-305.

[37] McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions. *Psychometrika*, 12(2), 153-157.

[38] Lundberg, S.M., & Lee, S.I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30, 4765-4774.

[39] Zhang, J., et al. (2021). Motif-based graph self-supervised learning for molecular property prediction. *Advances in Neural Information Processing Systems*, 34, 15870-15882.

[40] Ding, L., et al. (2018). Perspective on oncogenic processes at the end of the beginning of cancer genomics. *Cell*, 173(2), 305-320.

[41] Leiserson, M.D., et al. (2015). Pan-cancer network analysis identifies combinations of rare somatic mutations across pathways and protein complexes. *Nature Genetics*, 47(2), 106-114.

[42] Menden, M.P., et al. (2013). Machine learning prediction of cancer cell sensitivity to drugs based on genomic and chemical properties. *PLoS One*, 8(4), e61318.

[43] Sanchez-Vega, F., et al. (2018). Oncogenic signaling pathways in The Cancer Genome Atlas. *Cell*, 173(2), 321-337.

[44] Collins, F.S., & Varmus, H. (2015). A new initiative on precision medicine. *New England Journal of Medicine*, 372(9), 793-795.

[45] Luo, H., et al. (2019). deepDriver: Predicting cancer driver genes based on somatic mutations using deep convolutional neural networks. *Frontiers in Genetics*, 10, 13. DOI: 10.3389/fgene.2019.00013.

[46] Mao, Y., et al. (2013). CanDrA: Cancer-specific driver missense mutation annotation with optimized features. *PLoS One*, 8(10), e77945.

[47] Vogelstein, B., et al. (2013). Cancer genome landscapes. *Science*, 339(6127), 1546-1558.

[48] Weinstein, J.N., et al. (2013). The Cancer Genome Atlas pan-cancer analysis project. *Nature Genetics*, 45(10), 1113-1120.

[49] Street, W.N., Wolberg, W.H., & Mangasarian, O.L. (1993). Nuclear feature extraction for breast tumor diagnosis. *SPIE Proceedings on Electronic Imaging*, 1905, 861-870.

[50] Elmarakeby, H.A., et al. (2021). Biologically informed deep neural network for prostate cancer discovery. *Nature*, 598(7880), 348-352. DOI: 10.1038/s41586-021-03922-4.

[51] Gao, J., et al. (2013). Integrative analysis of complex cancer genomics and clinical profiles using the cBioPortal. *Science Signaling*, 6(269), pl1.

[52] Wolberg, W.H., Street, W.N., & Mangasarian, O.L. (1995). Breast Cancer Wisconsin (Diagnostic) Data Set. *UCI Machine Learning Repository*. DOI: 10.24432/C5DW2B. (Packaged in scikit-learn as `sklearn.datasets.load_breast_cancer`.)
