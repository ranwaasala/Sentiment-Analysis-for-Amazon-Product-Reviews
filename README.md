# Amazon Reviews'23 Data Analytics & Sentiment Analysis Project

## Project Overview
This project presents a complete data science workflow for analyzing the Amazon Reviews'23 dataset collected by the McAuley Lab. The analysis spans from raw review ingestion to exploratory analysis, sentiment labeling, and visual insight generation.

The primary objectives are to:
- Explore user review behavior and sentiment trends
- Analyze the role of product metadata in shaping consumer trust
- Evaluate feedback quality and perceived helpfulness
- Surface patterns for fairness-aware product recommendations
- Identify potential biases across user interactions and product categories

## About the Dataset
The Amazon Reviews'23 dataset is a large-scale benchmark featuring:
- 571M+ user reviews from May 1996 to September 2023
- 48M+ unique products with detailed metadata (title, price, features, images, etc.)
- Fine-grained review-level fields: rating, review text, verified_purchase, helpful_votes, and timestamp
- Rich item metadata: product descriptions, categories, store info, bundles, and image links

**Dataset Citation:** Hou et al. (2024). Bridging Language and Items for Retrieval and Recommendation. arXiv preprint arXiv:2403.03952

## Repository Structure
```
├── Final_Graduation_Project_Phase2_DEPI.ipynb   # Main Jupyter notebook with full analysis
├── Grad Proj - Sentiment Analysis.pdf           # Project documentation/presentation
├── LLM guide.pdf                                # Guide for LLM implementation
├── README.md                                    # This file
├── app.py                                       # Deployment application
├── label_encoder.pkl                            # Serialized label encoder
├── logistic_regression_model.pkl                # Trained logistic regression model
├── tfidf_vectorizer.pkl                         # Serialized TF-IDF vectorizer
```

## Project Pipeline
The analysis is structured into the following key stages:

### 1. Data Preparation & Exploration
- Data loading and database integration
- Data cleaning and preprocessing
- Handling missing values
- Data type correction
- Outlier removal
- Basic statistical analysis and distributions

### 2. QA & Visualization
- Formulation of key analytical questions
- Creation of insightful visualizations
- Exploration of variable relationships through charts and graphs
- Trend analysis across time periods and product categories

### 3. Data Preparation
- Text Cleaning & Preparation
   Lowercasing, removal of URLs/HTML, punctuation, special characters, and numbers
   Tokenization, stop words removal, contraction expansion
- Feature Extraction
   Splitting data into train, validation, and test sets
   Conversion to numerical features using TF-IDF and CountVectorizer
- Label Encoding & Scaling
   Mapping sentiment labels (negative/neutral/positive) to numeric codes
   Feature scaling with MaxAbsScaler on TF-IDF matrix
- Data Balancing
   Undersampling to ensure equal representation for all sentiment classes

### 4. Predictive Modeling
Multiple models were trained and evaluated:
- Logistic Regression
- Random Forest Classifier
- Optimized Random Forest (with hyperparameter tuning via RandomizedSearchCV)
- XGBoost (tuned)
- Support Vector Machines (SVM)
- Naive Bayes
- Model comparison using ROC-AUC curves

### 5. Application Deployment
- Built an interactive web app in Streamlit for real-time sentiment classification and visualization
- Integrated the trained model and LLM for advanced review analysis (summarization, bias detection, emotion tagging)

### 6. Final Presentation & Interpretation
- Summary of data cleaning methodology
- Key insights and findings
- Model performance assessment
- Business value explanation and recommendations

## Deployment
The `app.py` file provides a deployment solution for the sentiment analysis model. The trained model, vectorizer, and label encoder are saved as pickle files for use in the application.

## Getting Started

### Prerequisites
- Python 3.8+
- Required packages (install via `pip install -r requirements.txt`):
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - xgboost

### Running the Analysis
1. Clone this repository
2. Install the required dependencies
3. Run the Jupyter notebook `Final_Graduation_Project_Phase2_DEPI.ipynb`

### Running the Application
To deploy the sentiment analysis model and LLM locally:
```bash
python app.py
```

## Results
The project provides insights into user review behavior on Amazon's platform and delivers a robust sentiment analysis model that can be deployed for real-time analysis of product reviews.
