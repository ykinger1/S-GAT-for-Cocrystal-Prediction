# S-GAT: Dual-Branch Weight-Sharing Graph Attention Network for Organic Cocrystal Prediction
Official implementation of **S-GAT** (Weight-Sharing Graph Attention Network) for end-to-end organic cocrystal prediction and atom-level interpretability analysis.

## Abstract
To tackle the industrial bottlenecks including low screening efficiency of traditional organic cocrystal screening, dependence of existing machine learning prediction models on subjective prior features, poor adaptability to bimolecular pairing properties, and unsatisfactory interpretability, this paper proposes an end-to-end cocrystal prediction method based on two-branch weight-sharing graph attention, and develops an S-GAT prediction model. A graph attention encoding architecture with two-branch weight sharing is elaborately designed. Adopting molecular graphs as the only input, the model requires no manually screened prior physicochemical features. The weight-sharing mechanism guarantees the consistency of the bimolecular feature space and enables precise capture of the complementary matching principles between molecules. Furthermore, an atom-level interpretable analysis framework compatible with unfixed-length molecular graphs is established based on the GraphSHAP method, which realizes the interpretation of the internal mechanism behind cocrystal prediction results and the localization of key action sites. Systematic experiments are conducted to comprehensively validate the performance and practicality of the proposed model. The experimental results show that the S-GAT model achieves an AUC of 0.9906 on the test set. Its overall prediction capability and generalization stability are remarkably superior to those of mainstream graph deep learning models. The model can accurately identify the key atomic sites governing cocrystal formation, and its prediction mechanism is highly consistent with classical chemical theories. This method provides an efficient and reliable new strategy for large-scale virtual screening of organic cocrystals, and also delivers explicit theoretical guidance for subsequent experimental synthesis and molecular optimization of cocrystals.

## Environment Requirements
### Hardware Environment
- Operating System: Windows 10
- CPU: Intel Core i5-10200H (2.40GHz, 4 cores 8 threads)
- GPU: NVIDIA GeForce RTX 2060 (6GB dedicated memory, for model training & inference acceleration)

### Software Environment
Core Python packages used in this project:
```txt
python==3.10
torch==2.0.1+cu117
torch-geometric==2.7.0
rdkit==2025.3.5
numpy==1.26.4
pandas==2.3.2
scikit-learn==1.6.1
shap==0.49.1
matplotlib==3.10.6
```

## Usage
### 1. Model Training & Prediction
Run the standard training and inference pipeline:
```bash
python train_and_predict.py
```

### 2. K-Fold Cross-Validation
Evaluate model stability and generalization:
```bash
python cross_validation.py
```

### 3. Atom-Level Interpretability Analysis (GraphSHAP)
Compute atomic contribution scores for prediction results:
```bash
python SHAP.py
```

### 4. Molecular Visualization
Visualize key atomic sites identified by GraphSHAP:
```bash
python SHAP_visualizations.py
```


## Key Results
- **Test Set AUC**: 0.9906
- **Input**: Only molecular SMILES (no manual feature engineering)
- **Interpretability**: Atom-level key site localization via GraphSHAP
- **Advantage**: Outperforms mainstream graph neural network models; consistent with classical chemical theories


## Contact
For questions or issues, please open an issue or contact [wd_kun@126.com].
