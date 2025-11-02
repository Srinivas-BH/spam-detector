# Spam Detection System - Enhanced Version

## Overview
A high-performance spam email detection system with ensemble machine learning models, achieving **91%+ accuracy**. The system uses multiple algorithms and provides a web interface with visualization for comparing different model predictions.

## Features
- ✅ **Ensemble Model**: Combines multiple ML algorithms for best accuracy
- ✅ **Multiple Algorithms**: Random Forest, Naive Bayes, Logistic Regression, Gradient Boosting
- ✅ **Advanced Feature Engineering**: 13+ engineered features including URL detection, phishing patterns, urgency indicators
- ✅ **Web Interface**: Beautiful GUI with scatter plot visualization
- ✅ **API Endpoints**: RESTful API for integration
- ✅ **Multi-Algorithm Comparison**: Compare predictions from all models simultaneously

## Current Performance
- **Ensemble Accuracy**: ~91.27%
- **Training Dataset**: 30K samples (balanced)
- **Test Accuracy**: 91.27%

## Files Structure

```
spam-detector/
├── app.py                          # Main Flask application
├── train_ensemble_model.py         # Fast training script (30K samples, ~91% accuracy)
├── train_high_accuracy_model.py   # High-accuracy training (80K samples, targets 99%)
├── create_master_dataset.py       # Merges all datasets
├── master_spam_dataset.csv         # Combined dataset (291K messages)
├── spam_model_ensemble.pkl        # Trained ensemble model
├── vectorizer.pkl                 # TF-IDF vectorizer
├── feature_info.pkl               # Feature metadata
├── model_rf.pkl                   # Random Forest model
├── model_nb.pkl                   # Naive Bayes model
├── model_lr.pkl                   # Logistic Regression model
├── model_gb.pkl                    # Gradient Boosting model
├── templates/
│   └── spam.html                  # Web interface with visualization
└── requirements.txt               # Python dependencies
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download NLTK Data
```bash
python download_nltk.py
```
Or run Python and download:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### 3. Create Master Dataset (if needed)
```bash
python create_master_dataset.py
```
This merges all available datasets into `master_spam_dataset.csv`.

### 4. Train the Model
**Quick Training (91% accuracy, ~5-10 minutes):**
```bash
python train_ensemble_model.py
```

**High-Accuracy Training (targets 99%, ~20-30 minutes):**
```bash
python train_high_accuracy_model.py
```

### 5. Run the Application
```bash
python app.py
```

The app will start on `http://localhost:5000`

## API Endpoints

### 1. Web Interface
- **GET** `/` - Main web interface
- **POST** `/predict` - Form-based prediction

### 2. JSON API
- **GET** `/hello` - Health check
- **POST** `/api/predict` - Single prediction
  ```json
  {
    "message": "Your email message here"
  }
  ```
  Response:
  ```json
  {
    "prediction": "spam",
    "spam_probability": 0.95
  }
  ```

- **POST** `/predict-all` - Multi-algorithm comparison
  ```json
  {
    "message": "Your email message here"
  }
  ```
  Response:
  ```json
  {
    "ok": true,
    "algorithms": {
      "random_forest": {
        "label": "spam",
        "ham_score": 0.05,
        "spam_score": 0.95
      },
      "naive_bayes": {
        "label": "spam",
        "ham_score": 0.10,
        "spam_score": 0.90
      }
    },
    "consensus": {
      "label": "spam",
      "ham_score": 0.075,
      "spam_score": 0.925
    }
  }
  ```

## Features Used in Classification

1. **URL Detection**: Checks for http/https links
2. **Short URL Detection**: Identifies bit.ly, t.co, etc.
3. **Phishing Words**: Detects suspicious keywords
4. **Urgency Indicators**: Flags urgent language
5. **Message Statistics**: Length, word count, character ratios
6. **Punctuation Analysis**: Exclamation/question mark counts
7. **Money References**: Detects financial mentions
8. **Winner/Prize Language**: Identifies promotional content
9. **TF-IDF Vectorization**: 2000-5000 most important words/phrases
10. **N-gram Analysis**: 1-3 word combinations

## Model Details

### Ensemble Model
Uses VotingClassifier with soft voting, combining:
- Random Forest (91% accuracy)
- Gradient Boosting (90.88% accuracy)
- Naive Bayes (81.23% accuracy)

### Individual Models
All models are saved and can be used independently:
- `model_rf.pkl` - Random Forest
- `model_nb.pkl` - Naive Bayes
- `model_lr.pkl` - Logistic Regression
- `model_gb.pkl` - Gradient Boosting

## Improving Accuracy to 99%

To achieve 99% accuracy:

1. **Use more data**: Run `train_high_accuracy_model.py` which uses 80K samples
2. **Hyperparameter tuning**: Adjust model parameters in the training script
3. **Feature engineering**: Add more domain-specific features
4. **Data quality**: Ensure clean, balanced dataset
5. **Model stacking**: Use advanced ensemble techniques

## Testing

Test the API:
```bash
python test_api.py
```

Test multiple algorithms:
```bash
python test_enhanced_api.py
```

## Troubleshooting

### Model Not Found Error
If you see "Model not loaded", run:
```bash
python train_ensemble_model.py
```

### NLTK Data Error
Download required data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Memory Issues
Reduce dataset size in training scripts or use a machine with more RAM.

## Performance Notes

- **Current Training Time**: 5-10 minutes (30K samples)
- **Prediction Speed**: <100ms per message
- **Model Size**: ~50-100 MB
- **Memory Usage**: ~500 MB - 1 GB during training

## Next Steps

1. Deploy to production (Heroku, AWS, etc.)
2. Add model versioning
3. Implement active learning
4. Add real-time model retraining
5. Expand to other languages
6. Add email-specific features (headers, sender info)

## License

This project is for educational purposes.

