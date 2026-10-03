=== Streamlit Deployment Package ===

Files included:
1. app.py            - The main Streamlit dashboard application.
2. requirements.txt  - Dependencies list (Streamlit, scikit-learn, joblib, pandas, numpy).
3. scaler.joblib     - Fitted standard scaler for model feature normalization.
4. cluster_model.joblib - Trained hierarchical clustering model configuration.

Instructions to run locally:
1. Install dependencies using your terminal:
   pip install -r requirements.txt

2. Run the Streamlit application:
   streamlit run app.py

Instructions for Cloud Deployment (Streamlit Community Cloud):
1. Upload all 4 files (app.py, requirements.txt, scaler.joblib, cluster_model.joblib) to a GitHub repository.
2. Log in to https://share.streamlit.io/ and connect your GitHub account.
3. Deploy the repository using 'app.py' as the main entry point file.
