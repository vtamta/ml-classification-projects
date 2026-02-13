# Machine Learning Classification Projects

Two comprehensive machine learning projects demonstrating supervised learning techniques, data preprocessing, and ethical AI considerations.

## 📂 Projects

### 1. [Disease Classification System](disease-classification/)
Multi-class classification of respiratory illnesses (FLU, COVID, ALLERGY, COLD)

**Highlights:**
- Hybrid SMOTE + Undersampling for severe class imbalance
- Multiple algorithms: Random Forest, KNN, SVC
- Clustering analysis: DBSCAN, Agglomerative Clustering
- Cosine similarity analysis between disease classes
- **Best Accuracy**: 92.6% (Random Forest)

**Key Techniques:**
- Imbalanced data handling
- Hamming distance for binary features
- Hyperparameter tuning with RandomizedSearchCV
- Comprehensive evaluation metrics

### 2. [Health Risk Classification](health-risk-classification/)
Predicting patient health risk levels (High, Mid, Low) based on vital signs

**Highlights:**
- Decision Tree and Random Forest comparison
- Feature correlation analysis
- Hyperparameter optimization with GridSearchCV
- Ethical AI considerations in healthcare
- **Best Accuracy**: 84% (Random Forest)

**Key Techniques:**
- Domain knowledge for outlier removal
- Interaction effects in correlated features
- Model interpretability analysis
- Clinical deployment considerations

## 🛠️ Technologies

### Core Libraries
- **Python 3.x** - Primary programming language
- **Scikit-learn** - ML algorithms and model evaluation
- **Pandas & NumPy** - Data manipulation and analysis
- **Matplotlib & Seaborn** - Data visualization
- **imbalanced-learn** - SMOTE and sampling techniques
- **Jupyter Notebook** - Interactive development

### Algorithms Implemented
- Random Forest Classifier
- Decision Tree Classifier
- K-Nearest Neighbors (KNN)
- Support Vector Classifier (SVC)
- DBSCAN Clustering
- Agglomerative Clustering

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ml-classification-projects.git
cd ml-classification-projects
```

### Running the Projects

Navigate to either project folder and open the respective notebook:
- `disease-classification/disease_classification.ipynb`
- `health-risk-classification/health_risk_classification.ipynb`

## 📊 Project Comparison

| Aspect | Disease Classification | Health Risk Classification |
|--------|----------------------|---------------------------|
| **Classes** | 4 (FLU, COVID, ALLERGY, COLD) | 3 (High, Mid, Low Risk) |
| **Dataset Size** | ~40,000 samples | ~1,000 samples |
| **Features** | 20 binary symptoms | 6 vital signs (continuous) |
| **Main Challenge** | Severe class imbalance | Feature correlation |
| **Best Model** | Random Forest (92.6%) | Random Forest (84%) |
| **Key Focus** | Data sampling strategies | Ethical considerations |

## 💡 Key Learnings

### Technical Skills
- **Data Preprocessing**: Handling missing values, outliers, class imbalance
- **Feature Engineering**: Correlation analysis, interaction effects
- **Model Selection**: Comparing multiple algorithms
- **Hyperparameter Tuning**: GridSearchCV and RandomizedSearchCV
- **Evaluation**: Confusion matrices, precision, recall, F1-score

### Domain Expertise
- **Healthcare Context**: Understanding medical implications
- **Ethical AI**: Model interpretability, accountability, clinical deployment
- **Data Quality**: Domain knowledge for preprocessing decisions
- **Clustering Insights**: Understanding when unsupervised methods struggle

## 🎓 Academic Context

These projects were completed as part of coursework in **Principles of Machine Learning**, demonstrating:

✅ End-to-end ML pipeline development  
✅ Multiple algorithm implementation and comparison  
✅ Statistical validation with proper evaluation metrics  
✅ Ethical considerations in AI deployment  
✅ Technical writing and documentation skills  

## 📈 Results Summary

### Disease Classification
- Successfully handled 56:37:5:2 class imbalance ratio
- Achieved 92.6% accuracy with Random Forest
- Demonstrated clustering challenges with high-similarity classes
- Comprehensive comparison of supervised vs unsupervised methods

### Health Risk Classification
- 21% accuracy improvement through hyperparameter tuning (72% → 93%)
- Analyzed feature correlation vs. interaction effects
- Discussed ethical implications of medical AI
- Compared Decision Tree vs Random Forest performance

## 🔬 Methodologies

### Data Handling
1. Exploratory Data Analysis (EDA)
2. Missing value analysis
3. Outlier detection and treatment
4. Class imbalance assessment
5. Feature correlation analysis

### Model Development
1. Train-test split
2. Algorithm selection
3. Hyperparameter optimization
4. Cross-validation
5. Performance evaluation

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Silhouette Score (clustering)
- Adjusted Rand Index (clustering)

## 📝 Documentation

Each project includes:
- **Detailed README**: Problem description, methodology, results
- **Jupyter Notebook**: Complete code with explanations
- **PDF Report**: Academic assignment documentation
- **Visualizations**: Confusion matrices, performance plots

## 🤝 Contributing

These are academic projects completed for coursework. Feel free to fork and experiment with your own ideas!

## 📄 License

For educational purposes. Please contact author for reuse permissions.

## 🙏 Acknowledgments

- Course instructor and teaching staff
- Scikit-learn and imbalanced-learn communities
- Open-source machine learning community

## 📚 References

Both projects include comprehensive references to:
- Scikit-learn documentation
- Academic papers and research
- Healthcare domain resources
- Ethical AI guidelines

---

*Demonstrating practical machine learning applications with emphasis on data quality, algorithmic rigor, and ethical deployment considerations.*
