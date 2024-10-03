
# Drug-Nanoparticle-Synergy

**Abstract**

Antibiotic resistance is a critical global public health challenge, driven by the limited discovery of novel antibiotics, the rapid evolution of resistance mechanisms, and persistent infections that compromise treatment efficacy. Combination therapies using antibiotics and nanoparticles (NPs) offer a promising solution, particularly against multidrug-resistant (MDR) bacteria. This study introduces an innovative approach to identifying synergistic drug-NP combinations with enhanced antimicrobial activity. We compiled two groups of datasets to predict the minimal concentration (MC) and zone of inhibition (ZOI) for various drug-NP combinations. CatBoost regression models achieved the best 10-fold cross-validation R² scores of 0.86 and 0.77, respectively. We then adopted a machine learning (ML)-reinforced genetic algorithm (GA) to identify synergistic antimicrobial NPs. The proposed approach was first validated by reproducing previous experimental results. As a proof of concept for discovering new drug-NP combinations, Au NPs were identified as highly synergistic when paired with chloramphenicol, achieving a minimum bactericidal concentration (MBC) of 71.74 ng/ml against *Salmonella typhimurium*, with a fractional inhibitory concentration index of 6.23 × 10⁻³. These findings present a novel and effective strategy for identifying synergistic drug-NP combinations, providing a promising approach to combating drug-resistant pathogens and advancing targeted antimicrobial therapies.
![](/Figure.png)

**Guidelines**

This repository contains all the necessary files for screening synergistic antimicrobial NP and drug combinations. The repository includes folders for `data`, `model selection`, `model optimization`, `validation`, and `genetic algorithms`.

**Data**

The `data` folder contains raw and preprocessed data each separated into two groups of datasets. The target parameter of first group is minimal concentration (MC), which include parameters such as MIC, MBC, MTC, MFC, MBEC, and MBIC. This group is divided into three subsets for predicting MC of nanoparticles (MC_NP), drugs (MC_Drug), and their combinations (MC_Drug_NP). The second group focuses on the zone of inhibition (ZOI), measured by disk or well diffusion methods, and is similarly divided into three subsets for predicting ZOI of nanoparticles (ZOI_NP), drugs (ZOI_Drug), and their combinations (ZOI_Drug_NP). Additionally, this folder contains datasets for validation.

**Machine Learning**

**Model Selection:**  
We evaluated the performance of 43 different regression models on raw and processed datasets by predicting MC of NPs, drugs, and drug-NPs cocktail, as well as to predict ZOI of NPs, drugs, and drug-NPs cocktail. The Python scripts are present in the `Model Selection` folder, and results are stored as CSV files in their respective folders.

**Model Optimization:**  
Top models obtained from model selection were optimized by hyperparameter tuning to identify the best parameters. CatBoost and XGB regressor models showed the best performance after optimization and were used for predicting MC of NPs, drugs, and drug-NP cocktails, as well as for predicting ZOI of NPs, drugs, and drug-NP cocktails. Optimized models are stored in the `Model Optimization` folder.

**Model Validation:**  
10-fold cross-validation was carried out before evaluating the model's performance on external test datasets. The script for model validation on the test set is provided in the `Validation` folder.

**Genetic Algorithm**

To identify synergistic antimicrobial NPs, the script for the genetic algorithm is stored in the `Genetic Algorithm` folder. It consists of subfolders `MC_DNP` and `ZOI_DNP`, each containing necessary datasets and Python scripts for unique compound generation, crossover and mutation, and a main script (`Genetic_Algorithm_selective.py`) for evolution up to user-defined population size and generation number. The optimized models were used to predict synergistic activity in these generated unique compounds. The best drug-NP combinations were selected by identifying compounds with the highest fitness scores, and an example of screening candidates is stored in the `Results` folder.
