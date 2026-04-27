"""
Cancer type definitions and mappings for various data sources.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class DataSource(Enum):
    TCGA = "TCGA"
    ICGC = "ICGC"
    EGA = "EGA"
    COSMIC = "COSMIC"
    CCLE = "CCLE"
    GDSC = "GDSC"
    NCI60 = "NCI60"
    # sklearn UCI Wisconsion breast-cancer (public benchmark); kNN graph in train.py
    BENCHMARK = "BENCHMARK"

@dataclass
class CancerType:
    code: str
    name: str
    description: str
    data_sources: List[DataSource]
    parent_type: Optional[str] = None
    aliases: List[str] = None

# TCGA Cancer Types
TCGA_CANCER_TYPES = {
    "BRCA": CancerType(
        code="BRCA",
        name="Breast Invasive Carcinoma",
        description="Breast cancer that has spread into surrounding breast tissue",
        data_sources=[DataSource.TCGA, DataSource.BENCHMARK, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Breast Cancer", "Breast Carcinoma"]
    ),
    "LUAD": CancerType(
        code="LUAD",
        name="Lung Adenocarcinoma",
        description="A type of non-small cell lung cancer that forms in mucus-producing cells",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Lung Adenocarcinoma", "Lung Cancer"]
    ),
    "LUSC": CancerType(
        code="LUSC",
        name="Lung Squamous Cell Carcinoma",
        description="A type of non-small cell lung cancer that forms in squamous cells",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Lung Squamous Cell Carcinoma", "Lung Cancer"]
    ),
    "COAD": CancerType(
        code="COAD",
        name="Colon Adenocarcinoma",
        description="Cancer that forms in the colon",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Colon Cancer", "Colorectal Cancer"],
        parent_type="COADREAD"
    ),
    "READ": CancerType(
        code="READ",
        name="Rectum Adenocarcinoma",
        description="Cancer that forms in the rectum",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Rectal Cancer", "Colorectal Cancer"],
        parent_type="COADREAD"
    ),
    "GBM": CancerType(
        code="GBM",
        name="Glioblastoma Multiforme",
        description="An aggressive type of brain cancer",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Glioblastoma", "Brain Cancer"],
        parent_type="GBMLGG"
    ),
    "LGG": CancerType(
        code="LGG",
        name="Brain Lower Grade Glioma",
        description="A type of brain tumor that grows slowly",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Low-Grade Glioma", "Brain Cancer"],
        parent_type="GBMLGG"
    ),
    "OV": CancerType(
        code="OV",
        name="Ovarian Serous Cystadenocarcinoma",
        description="A type of ovarian cancer that forms in the cells lining the ovaries",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Ovarian Cancer"]
    ),
    "UCEC": CancerType(
        code="UCEC",
        name="Uterine Corpus Endometrial Carcinoma",
        description="Cancer that forms in the lining of the uterus",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Endometrial Cancer", "Uterine Cancer"]
    ),
    "KIRC": CancerType(
        code="KIRC",
        name="Kidney Renal Clear Cell Carcinoma",
        description="A type of kidney cancer that forms in the cells lining the small tubes",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Kidney Cancer", "Renal Cell Carcinoma"],
        parent_type="KIPAN"
    ),
    "KIRP": CancerType(
        code="KIRP",
        name="Kidney Renal Papillary Cell Carcinoma",
        description="A type of kidney cancer that forms in the cells lining the small tubes",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Kidney Cancer", "Renal Cell Carcinoma"],
        parent_type="KIPAN"
    ),
    "THCA": CancerType(
        code="THCA",
        name="Thyroid Carcinoma",
        description="Cancer that forms in the thyroid gland",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Thyroid Cancer"]
    ),
    "PRAD": CancerType(
        code="PRAD",
        name="Prostate Adenocarcinoma",
        description="Cancer that forms in the prostate gland",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Prostate Cancer"]
    ),
    "STAD": CancerType(
        code="STAD",
        name="Stomach Adenocarcinoma",
        description="Cancer that forms in the stomach",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Stomach Cancer", "Gastric Cancer"],
        parent_type="STES"
    ),
    "SKCM": CancerType(
        code="SKCM",
        name="Skin Cutaneous Melanoma",
        description="Cancer that forms in the cells that color the skin",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Melanoma", "Skin Cancer"]
    ),
    "BLCA": CancerType(
        code="BLCA",
        name="Bladder Urothelial Carcinoma",
        description="Cancer that forms in the bladder",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Bladder Cancer"]
    ),
    "HNSC": CancerType(
        code="HNSC",
        name="Head and Neck Squamous Cell Carcinoma",
        description="Cancer that forms in the head and neck region",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Head and Neck Cancer"]
    ),
    "LIHC": CancerType(
        code="LIHC",
        name="Liver Hepatocellular Carcinoma",
        description="Cancer that forms in the liver",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Liver Cancer", "Hepatocellular Carcinoma"]
    ),
    "CESC": CancerType(
        code="CESC",
        name="Cervical Squamous Cell Carcinoma and Endocervical Adenocarcinoma",
        description="Cancer that forms in the cervix",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Cervical Cancer"]
    ),
    "SARC": CancerType(
        code="SARC",
        name="Sarcoma",
        description="Cancer that forms in the bones and soft tissues",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Soft Tissue Sarcoma", "Bone Cancer"]
    ),
    "LAML": CancerType(
        code="LAML",
        name="Acute Myeloid Leukemia",
        description="A type of blood cancer that affects the bone marrow",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["AML", "Leukemia"]
    ),
    "PAAD": CancerType(
        code="PAAD",
        name="Pancreatic Adenocarcinoma",
        description="Cancer that forms in the pancreas",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Pancreatic Cancer"]
    ),
    "ESCA": CancerType(
        code="ESCA",
        name="Esophageal Carcinoma",
        description="Cancer that forms in the esophagus",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Esophageal Cancer"],
        parent_type="STES"
    ),
    "PCPG": CancerType(
        code="PCPG",
        name="Pheochromocytoma and Paraganglioma",
        description="Tumors that form in the adrenal glands",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Pheochromocytoma", "Paraganglioma"]
    ),
    "TGCT": CancerType(
        code="TGCT",
        name="Testicular Germ Cell Tumors",
        description="Cancer that forms in the testicles",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Testicular Cancer"]
    ),
    "THYM": CancerType(
        code="THYM",
        name="Thymoma",
        description="Cancer that forms in the thymus",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Thymus Cancer"]
    ),
    "ACC": CancerType(
        code="ACC",
        name="Adrenocortical Carcinoma",
        description="Cancer that forms in the adrenal cortex",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Adrenal Cancer"]
    ),
    "MESO": CancerType(
        code="MESO",
        name="Mesothelioma",
        description="Cancer that forms in the mesothelium",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Mesothelioma"]
    ),
    "UVM": CancerType(
        code="UVM",
        name="Uveal Melanoma",
        description="Cancer that forms in the eye",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Eye Cancer", "Ocular Melanoma"]
    ),
    "DLBC": CancerType(
        code="DLBC",
        name="Lymphoid Neoplasm Diffuse Large B-cell Lymphoma",
        description="A type of non-Hodgkin lymphoma",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["DLBCL", "Lymphoma"]
    ),
    "UCS": CancerType(
        code="UCS",
        name="Uterine Carcinosarcoma",
        description="A rare type of uterine cancer",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Uterine Cancer"]
    ),
    "CHOL": CancerType(
        code="CHOL",
        name="Cholangiocarcinoma",
        description="Cancer that forms in the bile ducts",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Bile Duct Cancer"]
    )
}

# Additional cancer type groupings
CANCER_TYPE_GROUPS = {
    "COADREAD": CancerType(
        code="COADREAD",
        name="Colorectal Adenocarcinoma",
        description="Combined colon and rectal cancer",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Colorectal Cancer"]
    ),
    "GBMLGG": CancerType(
        code="GBMLGG",
        name="Glioma",
        description="Combined glioblastoma and lower grade glioma",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Brain Cancer"]
    ),
    "KIPAN": CancerType(
        code="KIPAN",
        name="Kidney Chromophobe",
        description="Combined kidney cancer types",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Kidney Cancer"]
    ),
    "STES": CancerType(
        code="STES",
        name="Stomach and Esophageal Carcinoma",
        description="Combined stomach and esophageal cancer",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["Gastrointestinal Cancer"]
    ),
    "PANCAN": CancerType(
        code="PANCAN",
        name="Pan-Cancer",
        description="All cancer types combined",
        data_sources=[DataSource.TCGA, DataSource.ICGC, DataSource.COSMIC],
        aliases=["All Cancers"]
    )
}

def get_cancer_type(code: str) -> Optional[CancerType]:
    """Get cancer type information by code."""
    return TCGA_CANCER_TYPES.get(code) or CANCER_TYPE_GROUPS.get(code)

def get_all_cancer_types() -> Dict[str, CancerType]:
    """Get all available cancer types."""
    return {**TCGA_CANCER_TYPES, **CANCER_TYPE_GROUPS}

def get_cancer_types_by_source(source: DataSource) -> List[CancerType]:
    """Get cancer types available in a specific data source."""
    return [ct for ct in get_all_cancer_types().values() if source in ct.data_sources]

def get_cancer_type_aliases(code: str) -> List[str]:
    """Get all aliases for a cancer type."""
    cancer_type = get_cancer_type(code)
    if not cancer_type:
        return []
    return cancer_type.aliases or []

def get_parent_cancer_type(code: str) -> Optional[CancerType]:
    """Get the parent cancer type if it exists."""
    cancer_type = get_cancer_type(code)
    if not cancer_type or not cancer_type.parent_type:
        return None
    return get_cancer_type(cancer_type.parent_type) 