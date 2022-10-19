## Deep Transfer Learning Enables Robust Prediction of Antimicrobial Resistance for Novel Antibiotics

Antimicrobial resistance (AMR) has become one of the serious global health problems, threatening the effective treatment of a growing number of infections. Machine learning and deep learning show a great potential application in rapid and accurate AMR prediction. However, the limited sample numbers and imbalanced class hinder the accuracy and generalization performance of the model. We proposed a deep transfer learning model that can improve model performance for AMR prediction on imbalanced minority groups, and thus provides an effective approach for model training on imbalanced data among minority groups. 

### Step1: Data preprocessing

- Variants Calling

  - Here, we called variants using `bcftools` software. You can also use other tools for variants calling.

- SNP-matrix 

  - We then extracted SNPs variants, reference alleles, and their positions, and merged all isolates based on the positions of reference alleles.

  - The final format of SNP-matrix (N replaces a locus without variation):

    | Sample_name | Position_1 | Position_2 | Position_3 | ...  | Position_n |
    | ----------- | ---------- | ---------- | ---------- | ---- | ---------- |
    | Ref_allele  | A          | T          | G          | ...  | C          |
    | Sample_1    | G          | A          | A          | ...  | T          |
    | Sample_2    | G          | N          | A          | ...  | T          |
    | Sample_3    | G.         | A.         | C.         | ...  | G          |
    | ...         | ...        | ...        | ...        | ...  | ...        |
    | Sample_m    | T          | A          | A          | ...  | T          |

- Encoding SNP-natrix

  - Label encoding
    - The A, G, C, T, N in the SNP_matrix were converted to 1, 2, 3, 4, and 0.
  - One-hot encoding
    - Each allele is encoded into a bianry matrix.

### Step2: Basic CNN model and save pre-trained model

- Our basic CNN model: `01_basic_CNN.py`
- Our pre-trained model: `AMR_pre_trained_model.h5`

### Step3: Fine-tuning

- Preparing your own SNP_matrix data
- Loading our pre-trained model to re-train on your own data: `02_TL.py`

### Step4: Plot evaluation results

- Plot evaluation results: `03_plot.R`



