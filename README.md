🎯 Problem Statement

Online reviews strongly influence customer decisions in e-commerce and service platforms. 
However, fake or deceptive reviews reduce customer trust and mislead businesses. 
A reliable solution is needed to automatically detect and filter such reviews.

🛠 Tech Stack

Programming Language: Python
Libraries/Frameworks: scikit-learn, pandas, numpy, Streamlit, joblib
Techniques: Natural Language Processing (TF-IDF, text preprocessing), Logistic Regression, Handcrafted Features (exclamation count, uppercase ratio, review length)
Tools: Jupyter Notebook, GitHub, Streamlit

🚀 Solution / Approach

Data Preprocessing:
Cleaned and tokenized review text.
Applied TF-IDF vectorization for feature extraction.
Designed handcrafted features (exclamation count, uppercase ratio, review length) to improve model accuracy.

Model Training:
Built a binary classifier using Logistic Regression.
Optimized performance with train/test split and evaluation metrics.
Deployment:

Developed a Streamlit dashboard for real-time predictions.
Users can paste any review and instantly see whether it is Fake or Genuine, along with confidence score. 
