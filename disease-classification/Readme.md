# Disease Classification System

A machine learning project for classifying respiratory conditions (FLU, COVID, ALLERGY, COLD) using supervised and unsupervised learning techniques with emphasis on handling imbalanced datasets.

## 📋 Overview

This project tackles the challenge of multi-class classification of respiratory illnesses using binary symptom data. The system employs advanced data balancing techniques and compares multiple ML algorithms to achieve optimal classification accuracy.

**Course**: Principles of Machine Learning  
**Author**: Vaibhav Tamta

## 🎯 Problem Statement

Classify patients into four respiratory condition categories based on 20 binary symptom features:
- **FLU** (56.2% of data)
- **ALLERGY** (36.8% of data)
- **COVID** (4.6% of data)
- **COLD** (2.3% of data)

**Challenge**: Severe class imbalance requiring sophisticated sampling strategies.

## 🔑 Key Features

- **Hybrid Data Sampling**: Combination of undersampling (majority) and SMOTE (minority classes)
- **Multiple Algorithms**: Random Forest, KNN, SVC with hyperparameter optimization
- **Clustering Analysis**: DBSCAN and Agglomerative Clustering with Hamming distance
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrices
- **Feature Analysis**: Cosine similarity analysis between classes

## 🛠️ Technology Stack

### Core Libraries
- **Scikit-learn** - ML algorithms and evaluation metrics
- **imbalanced-learn** - SMOTE and sampling techniques
- **Pandas & NumPy** - Data manipulation
- **Matplotlib & Seaborn** - Visualization
- **SciPy** - Distance metrics (Hamming distance)

### Algorithms Implemented
1. **Random Forest Classifier** (Best Performance)
2. **K-Nearest Neighbors (KNN)**
3. **Support Vector Classifier (SVC)**
4. **DBSCAN Clustering**
5. **Agglomerative Clustering**

## 📊 Dataset Characteristics

- **Shape**: 40,007 samples × 20 binary features
- **Features**: All binary (0/1) symptom indicators
- **Target**: 4 classes (FLU, ALLERGY, COVID, COLD)
- **Imbalance**: Significant - majority classes 10-20x larger than minority
- **Missing Values**: None
- **Outliers**: Handled appropriately

## 🔄 Data Preprocessing

### Hybrid Sampling Strategy

To address severe class imbalance:

1. **Undersampling Majority Classes**:
   - FLU: Reduced to 10,000 samples
   - ALLERGY: Reduced to 10,000 samples

2. **SMOTE for Minority Classes**:
   - COVID: Increased to 5,000 samples
   - COLD: Increased to 5,000 samples

**Final Balanced Dataset**: 30,000 samples (10K, 10K, 5K, 5K)

### Why No Feature Scaling?

Binary features (0/1) are already in consistent range. Scaling would:
- Not enhance feature relationships
- Not improve algorithm performance
- Potentially introduce inconsistencies
- Be meaningless for categorical binary data

## 🚀 Model Performance

### Random Forest Classifier (Best Model)

```
Accuracy:  92.6%
Precision: 95.7%
Recall:    73.1%
F1-Score:  79.6%
```

**Best Hyperparameters**:
- `max_depth`: 10
- `min_samples_leaf`: 4
- `min_samples_split`: 5
- `n_estimators`: 500
- **CV Accuracy**: 95.4%

### K-Nearest Neighbors (KNN)

```
Accuracy:  86.6%
Precision: 65.5%
Recall:    91.2%
F1-Score:  69.7%
```

**Best Hyperparameters**:
- `n_neighbors`: 11
- `weights`: 'uniform'
- `p`: 2
- `metric`: 'euclidean'
- **CV Accuracy**: 90.9%

### Clustering Results

**DBSCAN**: Single cluster (high feature similarity across classes)

**Cosine Similarity Analysis**:
- COVID vs ALLERGY: 0.994
- COVID vs FLU: 0.983
- COVID vs COLD: 0.993

**Metrics**:
- Silhouette Score: 0.03 (poorly defined clusters)
- Adjusted Rand Index: 0.58 (moderate agreement)

## 📈 Key Findings

### Supervised Learning Success
- Random Forest achieved 92.6% accuracy with high precision (95.7%)
- KNN achieved 86.6% accuracy with high recall (91.2%)
- Both models significantly outperformed clustering approaches

### Clustering Challenges
- High feature similarity between classes (cosine similarity > 0.98)
- Binary symptom overlap makes unsupervised clustering difficult
- DBSCAN grouped all samples into single cluster
- Demonstrates need for labeled data in medical diagnosis

### Model Comparison
- **Random Forest**: Better precision, fewer false positives
- **KNN**: Better recall, identifies more true positives
- **Choice**: Depends on whether minimizing false positives or false negatives is more critical

## 💡 Implementation Highlights

### Data Handling
- Designed and implemented hybrid sampling strategy (SMOTE + undersampling)
- Analyzed feature correlations and class distributions
- Applied appropriate distance metrics for binary data

### Algorithm Development
- Implemented and optimized Random Forest classifier (92.6% accuracy)
- Developed KNN classification with custom hyperparameters
- Experimented with SVC for comparison
- Applied hyperparameter tuning using RandomizedSearchCV

### Clustering Analysis
- Implemented DBSCAN with Hamming distance
- Developed Agglomerative Clustering approach
- Calculated cosine similarity for class analysis
- Evaluated clustering performance with multiple metrics

### Evaluation & Analysis
- Created comprehensive confusion matrices
- Performed comparative analysis across all methods
- Analyzed why clustering struggled with binary symptom data
- Documented findings and methodology

## 🎓 Use Cases

- **Healthcare Screening**: Initial symptom-based triage
- **Epidemiological Studies**: Pattern recognition in disease spread
- **Clinical Decision Support**: Assisting healthcare providers
- **Research**: Understanding symptom overlap in respiratory illnesses

## 🔍 Key Learnings

1. **Class Imbalance**: Hybrid sampling (under + SMOTE) crucial for minority classes
2. **Binary Features**: Hamming distance appropriate for similarity measurement
3. **Model Selection**: Ensemble methods (Random Forest) excel with complex patterns
4. **Clustering Limitations**: High feature similarity makes unsupervised learning challenging
5. **Evaluation**: Multiple metrics needed - no single metric tells full story

## 📝 Future Improvements

- [ ] Feature engineering (symptom combinations)
- [ ] Deep learning approaches (neural networks)
- [ ] Temporal analysis (symptom progression)
- [ ] Cost-sensitive learning
- [ ] External validation on different datasets
- [ ] Real-time prediction API

## 🚀 Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Running the Project

```bash
jupyter notebook disease_classification.ipynb
```

## 📁 Repository Structure

```
disease-classification/
├── README.md
├── disease_classification.ipynb
└── Assignment_3_no_team.pdf
```

## 🤝 Contributing

This is an individual academic project. Feel free to fork and experiment with your own ideas!

## 📄 License

For educational purposes. Please contact author for reuse permissions.

## 🙏 Acknowledgments

- Course instructor and teaching staff
- Scikit-learn and imbalanced-learn communities
- Healthcare professionals who provided domain insights

## 📚 References

- [Random Forest - Scikit-learn](https://scikit-learn.org/dev/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [KNN Classification - Scikit-learn](https://scikit-learn.org/dev/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [DBSCAN - Scikit-learn](https://scikit-learn.org/1.5/modules/generated/sklearn.cluster.DBSCAN.html)
- [SMOTE - imbalanced-learn](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- [Agglomerative Clustering - Scikit-learn](https://scikit-learn.org/dev/modules/generated/sklearn.cluster.AgglomerativeClustering.html)

---

*Developed as part of Principles of Machine Learning coursework*
