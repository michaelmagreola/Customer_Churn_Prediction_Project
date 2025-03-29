📊 Customer Churn Prediction with Machine Learning

Predicting customer churn using various classification algorithms.

📝 Overview

This project aims to build a predictive model to identify customers likely to churn from a subscription-based telecom service. By analyzing customer behavior and service data, businesses can proactively retain customers and implement better retention strategies.
🎯 Objectives
• Analyze and visualize customer churn data.
• Build predictive machine learning models (Logistic Regression, Decision Tree, Random Forest, XGBoost).
• Compare and optimize models using performance metrics.
• Provide actionable insights into customer churn factors.

📂 Dataset
https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Source: Kaggle Telco Customer Churn
Description: Dataset containing customer information, subscription plans, tenure, and churn status.

How to Run the Project

1. Clone the Repository
   git clone <your_repository_url>
   cd customer-churn-prediction
2. Create Virtual Environment & Install Dependencies
   python -m venv venv
   source venv/bin/activate # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
3. Open the Notebook
   Navigate to the notebooks/ folder and open churn_prediction.ipynb in Jupyter Notebook.
   cd notebooks
   jupyter notebook

⚙️ Project Structure
customer-churn-prediction/
│
├── data/
│ └── Telco-Customer-Churn.csv # Raw data file
│
├── notebooks/
│ └── churn_prediction.ipynb # Jupyter Notebook with end-to-end analysis
│
├── models/
│ └── best_model.pkl # Serialized trained model
│
├── src/
│ ├── preprocessing.py # Data preprocessing functions
│ └── model.py # ML model training and evaluation
│
├── app/ (optional)
│ └── streamlit_app.py # Streamlit web app for deployment
│
├── requirements.txt # Project dependencies
└── .gitignore # Files and folders ignored by git

🛠️ Tech Stack
• Python
• Pandas, NumPy
• Matplotlib, Seaborn
• Scikit-learn
• XGBoost (Optional)
• Streamlit (Optional deployment)

📈 Results & Insights
Model Accuracy Precision Recall F1-score ROC-AUC
Logistic Regression 79% 0.65 0.55 0.60 0.84
Decision Tree 77% 0.61 0.57 0.59 0.75
Random Forest 80% 0.67 0.56 0.61 0.85
XGBoost 81% 0.69 0.58 0.63 0.86

📌 Conclusion
This project successfully demonstrates the use of machine learning techniques to predict customer churn, providing businesses valuable insights for retention strategies.

Key insights:
• Contract type and tenure significantly impact churn.
• Monthly charges directly correlate with churn rate.
• Customer support and engagement levels influence customer retention.

🧠 Future Improvements
• Implement advanced feature engineering techniques.
• Include more comprehensive model explainability (SHAP/LIME).
• Deploy and monitor the model in a production environment.

📬 Contact: Feel free to reach out for any questions or collaboration opportunities
• GitHub: https://github.com/michaelmagreola
• Email: michaelmagreola@gmail.com
