# nlp-evaluator

# 📰 Politifact NLP Phase Evaluator

A **Streamlit web application** to evaluate political statements using NLP techniques.  
The app can **scrape statements from PolitiFact**, allow CSV uploads of pre-collected data, and perform **machine learning-based analysis** with multiple models.  

---

## Features

- **Data Sourcing**
  - Scrape statements from [PolitiFact](https://www.politifact.com/) by **date range**.
  - Upload your own CSV file with statements and labels.

- **Data Filtering**
  - Filter scraped/uploaded data by **date range** or **speaker**.

- **NLP Analysis**
  - Convert statements into **TF-IDF vectors**.
  - Encode labels automatically.
  - Run multiple ML models:
    - **Naive Bayes**
    - **Decision Tree**
    - **Logistic Regression**
    - **Support Vector Machine (SVM)**
    - **K-Nearest Neighbors (KNN)**

- **Performance Evaluation**
  - Displays **accuracy**, **F1-score**, and **training time**.
  - Visual **bar chart comparison** of models.
  - **Scatter plot trade-off**: time vs accuracy.

- **Caching**
  - Scraped data is cached to reduce repeated requests and speed up the app.
 
  - 
---

