# Health Risk Classification

A machine learning project for predicting patient health risk levels (High, Mid, Low) using vital health parameters, with focus on tree-based algorithms and ethical considerations in healthcare AI.

## 📋 Overview

This project develops a classification system to assess health risks based on key vital signs. The model aims to assist healthcare professionals in identifying high-risk cases for early intervention while maintaining interpretability and ethical standards.

**Course**: Principles of Machine Learning  
**Student**: Vaibhav Tamta

## 🎯 Problem Statement

Classify patients into three risk categories based on six health parameters:
- **High Risk**: Requires immediate medical attention
- **Mid Risk**: Needs monitoring and preventive care  
- **Low Risk**: Standard care sufficient

**Goal**: Accurate, interpretable predictions to support clinical decision-making.

## 🔑 Key Features

- **Tree-Based Algorithms**: Decision Tree and Random Forest implementations
- **Hyperparameter Optimization**: GridSearchCV for optimal model performance
- **Balanced Dataset**: No class imbalance - equal distribution across risk levels
- **Comprehensive Evaluation**: Confusion matrices, accuracy, precision, recall
- **Ethical Analysis**: Healthcare-specific considerations and limitations
- **Feature Correlation Study**: Interaction effects between vital signs

## 🛠️ Technology Stack

### Core Libraries
- **Scikit-learn** - ML algorithms and model evaluation
- **Pandas & NumPy** - Data manipulation and analysis
- **Matplotlib & Seaborn** - Visualization
- **StandardScaler** - Feature scaling (demonstrated but not necessary)

### Algorithms Implemented
1. **Decision Tree Classifier**
2. **Random Forest Classifier**

## 📊 Dataset Characteristics

- **Shape**: ~1,014 samples × 6 features
- **Features**: All numerical vital signs
  - Age (years)
  - SystolicBP (mmHg)
  - DiastolicBP (mmHg)
  - BS (Blood Sugar - mg/dL)
  - BodyTemp (°F)
  - HeartRate (bpm)
- **Target**: 3 balanced classes (High Risk, Mid Risk, Low Risk)
- **Missing Values**: None
- **Class Distribution**: Balanced (~33% each)

### Feature Correlation

**Key Finding**: High correlation (0.79) between SystolicBP and DiastolicBP

**Decision**: Both features retained despite correlation
- **Reason**: Interaction effects provide valuable information
- **Evidence**: Removing either feature significantly reduced accuracy
- **Interpretation**: Combined blood pressure readings capture nuanced health status

## 🔄 Data Preprocessing

### Outlier Handling

- **Initial Analysis**: 392 rows (0.38%) contained outliers
- **Decision**: Minimal removal to preserve information
- **Action Taken**: Removed 2 rows with HeartRate < 10 (medically impossible)
- **Domain Knowledge**: Normal heart rate range is 60-100 bpm (30-40 for athletes)

### Feature Scaling

**Demonstrated but not applied for final models**

Decision Tree and Random Forest tested with:
- ✅ Raw features
- ✅ StandardScaler normalization

**Result**: Identical performance (93% train, 83% test)

**Reason**: Tree-based algorithms split on thresholds, not distances
- Scaling doesn't affect split points
- Only distance-based algorithms (KNN, SVM) require scaling
- Maintaining raw values improves interpretability

## 🚀 Model Performance

### Decision Tree Classifier

**Without Hyperparameter Tuning**:
```
Training Score: 72%
Testing Score:  64%
```

**With GridSearchCV Optimization**:
```
Training Score: 93%
Testing Score:  83%
```

**Best Hyperparameters**:
- `criterion`: 'entropy'
- `max_depth`: None
- `min_samples_split`: 2
- `min_samples_leaf`: 1

### Random Forest Classifier

**Performance**:
```
Training Score: 93%
Testing Score:  84% (slightly better)
```

**Best Hyperparameters**:
- `n_estimators`: Optimized via GridSearchCV
- `max_depth`: Optimized via GridSearchCV  
- `min_samples_split`: Optimized via GridSearchCV
- `min_samples_leaf`: Optimized via GridSearchCV

### Model Comparison

**Similarities**:
- Both achieve ~93% training accuracy
- Both reach ~83-84% testing accuracy
- Performance gap due to small dataset size (~1,014 samples)

**Why Similar Performance?**:
- Small, simple dataset
- Random Forest = ensemble of Decision Trees
- Limited data complexity reduces ensemble advantage
- Random Forest shows marginal improvement (1% better)

## 📈 Evaluation Metrics

### Confusion Matrices

Both models show similar confusion patterns:
- Strong diagonal (correct predictions)
- Some confusion between Mid Risk and adjacent categories
- Overall balanced performance across all three classes

### Key Observations

1. **High Risk Detection**: Critical for patient safety - both models perform well
2. **Mid Risk**: Moderate confusion with High and Low risk
3. **Low Risk**: Generally well-identified

## ⚠️ Ethical Considerations

### Misclassification Risks

**False Negative (High Risk → Low Risk)**:
- ❌ **Most Dangerous**: Delay in critical care
- ❌ Life-threatening consequences
- ❌ Missed opportunity for early intervention

**False Positive (Low Risk → High Risk)**:
- ⚠️ Unnecessary stress and anxiety
- ⚠️ Additional medical costs
- ⚠️ Potential unnecessary treatments
- ⚠️ Emotional strain

### Model Interpretability

**Decision Tree**: ✅ Highly interpretable
- Clear decision paths
- Explainable to healthcare professionals
- Patients can understand reasoning

**Random Forest**: ⚠️ Less interpretable
- "Black box" nature
- Complex aggregation of multiple trees
- Harder to explain individual predictions

**Importance**: Healthcare requires transparency
- Doctors need to understand "why"
- Patients deserve explanations
- Trust depends on interpretability

### Ethical Deployment

**❌ Not Recommended**:
- Sole decision-maker for patient care
- Replacement for clinical judgment
- Autonomous treatment decisions

**✅ Recommended**:
- Decision support tool
- Screening and triage assistance
- Prompting further clinical investigation
- Combined with human expertise

### Accountability Concerns

**Who is responsible if model fails?**
- Healthcare provider using the tool?
- Developer who created the model?
- Institution that deployed it?
- **Answer**: Requires clear guidelines and human oversight

## 🎓 Use Cases

- **Health Screening**: Initial risk assessment in clinics
- **Resource Allocation**: Prioritize high-risk patients
- **Rural Healthcare**: Support areas with limited specialists
- **Early Warning System**: Flag concerning vital sign patterns
- **Clinical Decision Support**: Supplement (not replace) doctor judgment

## 🚀 Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Running the Project

```bash
jupyter notebook health_risk_classification.ipynb
```

## 📁 Repository Structure

```
health-risk-classification/
├── README.md
├── health_risk_classification.ipynb
└── ML-assignment-1.pdf
```

## 💡 Key Learnings

1. **Feature Interaction**: Correlated features can still provide value through interactions
2. **Hyperparameter Tuning**: Critical - improved accuracy from 72% to 93%
3. **Model Selection**: Simple models sufficient for small datasets
4. **Healthcare AI Ethics**: Interpretability often more important than accuracy
5. **Domain Knowledge**: Essential for preprocessing decisions (outlier removal)
6. **Scaling**: Not necessary for tree-based algorithms

## 🔬 Technical Insights

### Why Decision Trees Work Well Here

1. **Numerical Features**: Continuous vital signs suit threshold-based splits
2. **Small Dataset**: Prevents overfitting common in complex models
3. **Interpretability**: Doctors can trace decision logic
4. **Non-linear Relationships**: Captures complex health interactions

### GridSearchCV Impact

**Hyperparameters Tuned**:
- `max_depth`: Controls tree depth (prevents overfitting)
- `min_samples_split`: Minimum samples to split node
- `min_samples_leaf`: Minimum samples in leaf node
- `criterion`: Split quality measure (gini vs entropy)

**Result**: 21% accuracy improvement (72% → 93%)

## 📝 Future Improvements

- [ ] Larger, more diverse dataset
- [ ] Feature engineering (BMI, age groups)
- [ ] Temporal analysis (track changes over time)
- [ ] Explainable AI (SHAP values)
- [ ] External validation on different populations
- [ ] Integration with electronic health records
- [ ] Real-time monitoring system

## 🤝 Contributing

This is an individual academic project. For questions or suggestions, please contact the author.

## 📄 License

For educational purposes. Please contact author for reuse permissions.

## 🙏 Acknowledgments

- Course instructor for guidance
- Healthcare professionals for domain insights
- Scikit-learn community
- Open-source ML community

## 📚 References

- [Decision Trees - Scikit-learn](https://scikit-learn.org/1.5/modules/tree.html)
- [Random Forest - Scikit-learn](https://scikit-learn.org/dev/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [GridSearchCV - Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- Medical domain knowledge on vital sign ranges

---

*Developed as part of Principles of Machine Learning coursework with emphasis on ethical AI in healthcare*
