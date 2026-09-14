# amazon-reviews-data-mining
# Amazon Product Reviews: End-to-End Data Mining & Machine Learning Pipeline

An end-to-end data mining and machine learning project built on the **Amazon Reviews 2023** dataset (McAuley Lab). The project implements a full analytical pipeline: from large-scale streaming data extraction (ETL) and exploratory analysis to product clustering, dual recommendation engines, sentiment classification, and market basket analysis.

---

##  Project Architecture & Highlights

### 1. Data Pipeline & ETL (`datasets`, `pandas`)
* **Multi-Category Ingestion:** Streamed and processed raw reviews and metadata across 5 distinct Amazon categories:
  * *Subscription Boxes* (~16k records)
  * *Magazine Subscriptions* (~71k records)
  * *Gift Cards* (~152k records)
  * *Digital Music* (~130k records)
  * *Handmade Products* (~664k records)
* **Pre-processing:** Handled missing values, normalized metadata structures (prices, rating counts, descriptions), text cleaning (regex punctuation & lowercase normalization), and inner joining on parent ASIN identifiers.

### 2. Exploratory Data Analysis (EDA) & Trend Analysis
* **Rating Distributions & Category Comparison:** Visualized global score distributions and comparative mean ratings using `seaborn` and `matplotlib`.
* **Disappointment Analysis:** Extracted high-volume, low-rated products ($N \ge 50$ reviews, $\bar{r} < 3.5$) to uncover frequent pain points via corpus frequency extraction (`broken`, `cheap`, `disappointed`, `shipping`, `charged`).
* **Best-Sellers & Temporal Trends:** Identified top-performing products by volume and tracked seasonal rating variations over time.

### 3. Machine Learning & Modeling Tasks (Focused on *Digital Music*)

####  Task 1: Product Clustering & Text Embedding
* Combined normalized numerical features (`price`, `average_rating`) with textual representations using **TF-IDF Vectorization** (300 max features) and Porter Stemming on product descriptions.
* Determined optimal clusters using the **Elbow Method** ($k=4$) with a **Silhouette Score of 0.552**.
* Applied **PCA (Principal Component Analysis)** for 2D visualization of product clusters.

####  Task 2: Recommendation Systems
Implemented three complementary recommendation strategies:
1. **User-Based Collaborative Filtering:** Computes user similarity using **Cosine Similarity** over sparse user-item interaction matrices to predict ratings for unreviewed items.
2. **Item-Based Collaborative Filtering:** Calculates item-item similarity vectors to suggest relevant products based on past consumption history.
3. **Content-Based Filtering:** Employs TF-IDF and Cosine Similarity over product descriptions to generate immediate, cold-start resistant recommendations.

####  Task 3: Sentiment Analysis & NLP
* Preprocessed customer review text and transformed it via TF-IDF (3,000 features).
* Formulated a 3-class classification problem (`positive`, `neutral`, `negative`).
* Trained and evaluated multiple classifiers via **10-Fold Stratified Cross-Validation**:
  * **Multinomial Naive Bayes**
  * **K-Nearest Neighbors (KNN)**
  * **Random Forest Classifier** (Top performer: Macro $F_1 \approx 0.369$, Overall Accuracy $\approx 0.89$).
* Analyzed class imbalance impacts and documented precision/recall tradeoffs on underrepresented negative/neutral classes.

####  Task 4: Frequent Pattern & Association Rule Mining
* Extracted multi-item customer baskets using transaction encoding.
* Mined frequent itemsets using the **Apriori Algorithm** (`min_support = 0.001`).
* Generated association rules evaluated on **Lift** (`min_threshold = 0.8`), uncovering cross-selling patterns (lift values $> 200$ between related music products).

---

##  Repository Structure

.
├── 01_data_extraction_and_eda.ipynb      # ETL, data cleaning & exploratory analysis
├── 02_machine_learning_and_modeling.ipynb # Clustering, Recommenders, NLP & Apriori
├── requirements.txt                      # Project dependencies
└── README.md                             # Documentation
