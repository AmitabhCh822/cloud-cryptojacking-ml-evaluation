# Machine Learning for Cloud Cryptojacking Detection: Preprocessing and Evaluation Effects

Code for the paper:

> **Empirical Evaluation of Machine Learning for Cloud Cryptojacking Detection: Effects of Preprocessing and Evaluation Protocols**
> Amitabh Chakravorty and Nelly Elsayed
> School of Information Technology, University of Cincinnati

Six classical models (Random Forest, XGBoost, LightGBM, Gradient Boosting, Decision Tree and KNN) are evaluated on two public proxy datasets, DS2OS and NSL-KDD. No public cloud-native cryptojacking dataset exists, so both serve as behavioral stand-ins. The experiments show how much of a reported score comes from preprocessing and evaluation choices rather than from the model itself.

## What each script produces

| Script | Paper output |
|---|---|
| `src/1_download_data.py` | Downloads both datasets from Kaggle |
| `src/2_preprocess.py` | Dataset statistics in Section 4 (class counts, SMOTE samples, unseen attack types) |
| `src/3_train_evaluate.py` | Table 2 (overall performance) and Table 3 (attack-class metrics and MCC) |
| `src/4_leakage_analysis.py` | Table 4 (leakage scores), Fig. 1 (leakage ablation) and the destinationServiceAddress check |
| `src/5_generalization.py` | Table 5 (seen versus novel attacks) and Fig. 2 (per-class recall) |
| `src/6_robustness.py` | Ten-seed stability and the SMOTE-N comparison |

`src/common.py` holds the shared pipeline and the model configurations from Table 1. `src/plots.py` draws both figures. Results land in `results/metrics/` as CSV files and in `results/figures/` as PNG files. Every script also writes `results/environment_*.json` with the Python and library versions of that run.

## Getting started

### Google Colab

```python
!git clone https://github.com/AmitabhCh822/cloud-cryptojacking-ml-evaluation.git
%cd cloud-cryptojacking-ml-evaluation
!pip install -q -r requirements.txt

# Kaggle token: upload kaggle.json to ~/.kaggle/ or set these two variables
import os
os.environ["KAGGLE_USERNAME"] = "your_username"
os.environ["KAGGLE_KEY"] = "your_key"

!python src/1_download_data.py
!python src/2_preprocess.py
!python src/3_train_evaluate.py
!python src/4_leakage_analysis.py
!python src/5_generalization.py
!python src/6_robustness.py
```

### Local machine

```bash
git clone https://github.com/AmitabhCh822/cloud-cryptojacking-ml-evaluation.git
cd cloud-cryptojacking-ml-evaluation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/1_download_data.py
python src/3_train_evaluate.py
```

Run the remaining scripts in numeric order. Each one is independent once the data is in `data/raw/`. `6_robustness.py` takes the longest because it retrains every model ten times.

If you would rather skip Kaggle, place `DS2OS.csv` under `data/raw/ds2os/` and `KDDTrain+.txt` with `KDDTest+.txt` under `data/raw/nsl_kdd/`. The scripts search `data/raw/` for those names.

## Datasets

**DS2OS** ([Kaggle mirror](https://www.kaggle.com/datasets/libamariyam/ds2os-dataset), [original](https://www.kaggle.com/datasets/francoisxa/ds2ostraffictraces))
- 357,952 IoT service records with 12 raw features
- Seven attack types (DoS, scan, malicious control, malicious operation, spying, data probing, wrong setup), collapsed to normal versus attack
- 347,935 normal and 10,017 attack records

**NSL-KDD** ([Kaggle mirror](https://www.kaggle.com/datasets/hassan06/nslkdd), [UNB CIC](https://www.unb.ca/cic/datasets/nsl.html))
- KDDTrain+ with 125,973 records and KDDTest+ with 22,544 records, 41 features each
- KDDTest+ includes attack types never seen in KDDTrain+, such as mscan, saint, apache2 and processtable

## Pipeline

**DS2OS**
1. Remove seven leakage-prone columns: timestamp, value, sourceID, sourceAddress, accessedNodeAddress, accessedNodeType, operation
2. Label encode the five remaining categorical features
3. Stratified 70/30 split with seed 42
4. SMOTE on the training part only (236,542 synthetic attack records)
5. Standardize with training statistics; the test set keeps its natural imbalance (104,381 normal, 3,005 attack)

**NSL-KDD**
1. Official KDDTrain+ and KDDTest+ partitions, untouched
2. Label encode protocol_type, service and flag
3. No resampling, since KDDTrain+ is already about 53 percent normal and 47 percent attack
4. Standardize with training statistics

## Model configurations (Table 1)

| Model | Configuration |
|---|---|
| Random Forest | 100 trees, max_depth=20, min_samples_split=5 |
| Decision Tree | max_depth=15, min_samples_split=5 |
| Gradient Boosting | 100 trees, learning_rate=0.1, max_depth=5 |
| XGBoost | 100 trees, learning_rate=0.1, max_depth=10 |
| LightGBM | 100 trees, max_depth=10 |
| KNN | k=5 |

## Values reported in the paper

### Table 2: overall performance

| Dataset | Model | Accuracy | Weighted F1 | Weighted precision | Training time |
|---|---|---|---|---|---|
| DS2OS | Random Forest | 96.26% | 0.9695 | 0.9830 | 21.92 s |
| DS2OS | XGBoost | 96.26% | 0.9695 | 0.9830 | 3.16 s |
| DS2OS | LightGBM | 96.26% | 0.9695 | 0.9830 | 3.78 s |
| DS2OS | Gradient Boosting | 96.23% | 0.9693 | 0.9829 | 45.31 s |
| DS2OS | Decision Tree | 96.26% | 0.9695 | 0.9830 | 0.55 s |
| DS2OS | KNN | 99.21% | 0.9915 | 0.9921 | 1.19 s |
| NSL-KDD | Random Forest | 77.17% | 0.7686 | 0.8345 | 8.61 s |
| NSL-KDD | XGBoost | 80.82% | 0.8073 | 0.8527 | 3.67 s |
| NSL-KDD | LightGBM | 80.35% | 0.8023 | 0.8503 | 3.44 s |
| NSL-KDD | Gradient Boosting | 78.25% | 0.7802 | 0.8391 | 47.81 s |
| NSL-KDD | Decision Tree | 77.66% | 0.7740 | 0.8354 | 1.01 s |
| NSL-KDD | KNN | 76.76% | 0.7639 | 0.8347 | 0.12 s |

### Table 3: attack-class metrics

| Dataset | Model | Precision | Recall | F1 | FPR | MCC |
|---|---|---|---|---|---|---|
| DS2OS | RF, XGBoost, LightGBM, DT | 0.426 | 0.966 | 0.591 | 0.038 | 0.628 |
| DS2OS | Gradient Boosting | 0.424 | 0.966 | 0.589 | 0.038 | 0.627 |
| DS2OS | KNN | 0.992 | 0.724 | 0.837 | 0.000 | 0.844 |
| NSL-KDD | Random Forest | 0.967 | 0.620 | 0.756 | 0.028 | 0.609 |
| NSL-KDD | XGBoost | 0.968 | 0.686 | 0.803 | 0.030 | 0.662 |
| NSL-KDD | LightGBM | 0.968 | 0.677 | 0.797 | 0.029 | 0.655 |
| NSL-KDD | Gradient Boosting | 0.966 | 0.640 | 0.770 | 0.030 | 0.624 |
| NSL-KDD | Decision Tree | 0.964 | 0.631 | 0.763 | 0.031 | 0.615 |
| NSL-KDD | KNN | 0.971 | 0.610 | 0.749 | 0.024 | 0.605 |

### Table 5: seen versus novel attacks on NSL-KDD

| Model | Accuracy, random split | Accuracy, KDDTest+ | Gap G |
|---|---|---|---|
| Random Forest | 0.996 | 0.773 | 0.223 |
| Decision Tree | 0.994 | 0.772 | 0.222 |
| KNN | 0.990 | 0.768 | 0.222 |
| Gradient Boosting | 0.993 | 0.782 | 0.211 |
| LightGBM | 0.995 | 0.803 | 0.192 |
| XGBoost | 0.996 | 0.808 | 0.188 |

## Reading the results

- Weighted scores hide the attack class on DS2OS. Tree models reach 0.97 weighted F1 but only 0.43 attack precision, so more than half of their alerts are false alarms.
- KNN flips that failure. Its precision is near perfect while it misses more than one attack in four.
- With all twelve DS2OS columns, XGBoost scores 0.99995 accuracy. The timestamp column alone predicts the label almost perfectly, which is why the leakage-aware pipeline removes it.
- On NSL-KDD every model loses 19 to 22 accuracy points when the test set contains attack types absent from training.

## Notes on reproducing the numbers

Table 2 and Table 3 come from the original Google Colab run. Training times depend on hardware, so compare them between models rather than against other studies. Library updates can move a metric in the third decimal place; `results/environment_*.json` records the versions behind each run.

## Citation

```bibtex
@misc{chakravorty2026code,
  author       = {Chakravorty, Amitabh and Elsayed, Nelly},
  title        = {Code for ``Empirical Evaluation of Machine Learning for Cloud Cryptojacking Detection''},
  howpublished = {GitHub repository, \url{https://github.com/AmitabhCh822/cloud-cryptojacking-ml-evaluation}},
  year         = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE).

## Contact

- Amitabh Chakravorty: [chakraa4@mail.uc.edu](mailto:chakraa4@mail.uc.edu)
- Nelly Elsayed: [elsayeny@ucmail.uc.edu](mailto:elsayeny@ucmail.uc.edu)
