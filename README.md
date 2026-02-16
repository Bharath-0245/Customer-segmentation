# 🛍️ Customer Segmentation Using Unsupervised Learning

> A Machine Learning project that segments customers into meaningful groups using clustering techniques.

---

## 📌 Project Overview

Customer Segmentation is a powerful technique used to understand customer behavior and design targeted marketing strategies.

This project applies **Unsupervised Learning algorithms** to segment mall customers based on their:

- Age  
- Gender  
- Annual Income  
- Spending Score  

Since the dataset does not contain predefined labels, clustering algorithms are used to discover hidden patterns within the data.

---
## 👨‍💻 Authors

- Polisolla Bharath
- Jayanth 


## 🎯 Problem Statement

To group customers into meaningful segments based on demographic and behavioral attributes without predefined labels, enabling businesses to:

- Improve marketing strategies  
- Offer personalized services  
- Make data-driven business decisions  

---

## 📂 Dataset

**Dataset Used:** `Mall_Customers.csv`

The dataset simulates real-world business scenarios where customer labels are not predefined, making it ideal for clustering analysis.

---

## 🧠 Methodology

### 1️⃣ Data Preprocessing

- Label Encoding for categorical variables (e.g., Gender)
- Feature Scaling using StandardScaler
- Data normalization for better clustering performance

---

### 2️⃣ Clustering Techniques

#### 🔹 K-Means Clustering
- Primary clustering algorithm used
- Optimal number of clusters (K) determined using:
  - Elbow Method
  - Silhouette Score
- Groups customers into distinct clusters

#### 🔹 Hierarchical Clustering
- Used as a secondary validation technique
- Provides structural insight into cluster formation
- Dendrogram used to visualize cluster merging
- Does not require predefined number of clusters

---

### 3️⃣ Dimensionality Reduction

#### 🔹 Principal Component Analysis (PCA)
- Reduces dimensionality of dataset
- Helps visualize clusters in 2D space
- Improves interpretability

---

## 📊 Results

The final model successfully identified distinct customer segments categorized by:

- Income levels  
- Spending behavior  
- Demographic patterns  

These insights can help businesses in:

- Targeted marketing  
- Customer retention strategies  
- Business optimization  

---

## 🛠️ Technologies Used

- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---


# Run the project
python ML.py
