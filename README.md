# ITAI-1371 — ML Midterm: Adult Income EDA & Classification

**Course:** ITAI-1371 — Intro to Machine Learning
**Date:** March 2026
**Author:** Alfredo Garza
**Dataset:** [UCI / Kaggle Adult Income Dataset](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset)

---

## 🎯 Problem Statement

Build a binary classification pipeline that predicts whether a person earns **more or less than $50,000/year** based on demographic and employment features from the UCI Adult Income dataset.

This project focuses on the full data preparation pipeline — the essential foundation before any model can be trained — including EDA, cleaning, feature engineering, encoding, and scaling.

---

## 📊 Dataset Overview

| Property | Value |
|----------|-------|
| Source | UCI Machine Learning Repository / Kaggle |
| Rows | 32,561 |
| Features | 15 (6 numeric, 9 categorical) |
| Target Variable | `income` (<=50K or >50K) |
| Class Distribution | ~75% <=50K, ~25% >50K (imbalanced) |

**Key Features:** age, workclass, education, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country

---

## 🔬 Approach & Methodology

### 1. Dataset Overview & Train/Test Split
- Loaded raw dataset (32,561 rows × 15 columns)
- - Applied a **70/30 stratified split** before any analysis
  -   - Training set: ~22,792 rows
      -   - Test set: ~9,769 rows
          - - EDA and all preprocessing performed **only on training data** to prevent data leakage
           
            - ### 2. Exploratory Data Analysis (EDA)
            - - Identified class imbalance: ~17,285 earning <=50K vs ~5,507 earning >50K (3:1 ratio)
              - - Analyzed age distribution (peaks 35–45, right-skewed toward older ages)
                - - Identified disguised missing values stored as `?` in `workclass`, `occupation`, `native_country`
                  - - Produced income distribution bar charts and age histograms (`eda_distributions.png`)
                    - - Created 3×3 grid of categorical feature bar charts (`eda_categorical.png`)
                     
                      - ### 3. Data Cleaning
                      - - Replaced `?` placeholder values with `NaN`
                        - - Imputed missing values using **mode** (most frequent value) per column
                          - - Verified: all three columns reduced to **0 nulls** after cleaning
                           
                            - ### 4. Scaling & Normalization
                            - - Applied **Min-Max Scaling** to all numeric columns (age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week)
                              - - Scaled range: [0, 1] — essential for distance-based algorithms (KNN, SVM)
                               
                                - ### 5. Encoding
                                - - **Label Encoding** applied to binary/ordinal categorical columns
                                  - - **One-Hot Encoding (OHE)** applied to remaining categorical features
                                    - - Final encoded dataset shape: 22,792 rows × 103 columns (15 → 103 after encoding)
                                     
                                      - ### 6. Feature Engineering
                                      - Three new features were engineered from existing data:
                                      - - `capital_net` — capital gain minus capital loss (net financial position)
                                        - - `age_group` — age binned into life stages: Young / EarlyCareer / MidCareer / LateCareer / Senior
                                          - - `is_high_hours` — binary flag (0/1) for working more than 45 hours/week
                                           
                                            - ### 7. Before & After Summary
                                            - Full transformation tracking table showing row count, column count, and missing value count at every pipeline stage:
                                           
                                            - | Stage | Rows | Columns | Missing Values |
                                            - |-------|------|---------|----------------|
                                            - | Raw Dataset | 32,561 | 15 | ~2,983 |
                                            - | After Train Split | 22,792 | 15 | ~2,000 |
                                            - | After Cleaning | 22,792 | 15 | 0 |
                                            - | After Encoding | 22,792 | 103 | 0 |
                                            - | After Feature Engineering | 22,792 | 103 | 0 |
                                           
                                            - ### 8. Export
                                            - - Clean training dataset exported as `adult_income_clean_train.csv`
                                              - - Raw test set preserved as `adult_income_test_raw.csv` (untouched until model evaluation)
                                               
                                                - ---

                                                ## 📁 Repository Contents

                                                | File | Description |
                                                |------|-------------|
                                                | `adult_income_midterm.ipynb` | Main Jupyter notebook — full pipeline implementation |
                                                | `adult_income.csv` | Original raw dataset |
                                                | `adult_income_clean_train.csv` | Cleaned and processed training dataset |
                                                | `adult_income_test_raw.csv` | Raw test set (held out for model evaluation) |
                                                | `midterm_proposal.pdf` | Project proposal document |
                                                | `dataset_url.docx` | Dataset source reference |
                                                | `files.zip` | Archive of all cleaned datasets and notebook |

                                                ---

                                                ## ⚙️ Dependencies

                                                ```
                                                pandas
                                                numpy
                                                matplotlib
                                                seaborn
                                                scikit-learn
                                                jupyter
                                                ```

                                                Install with:
                                                ```bash
                                                pip install pandas numpy matplotlib seaborn scikit-learn jupyter
                                                ```

                                                ---

                                                ## ▶️ How to Run

                                                ```bash
                                                # Clone the repository
                                                git clone https://github.com/phredogee/ml-midterm-eda.git
                                                cd ml-midterm-eda

                                                # Install dependencies
                                                pip install pandas numpy matplotlib seaborn scikit-learn jupyter

                                                # Launch the notebook
                                                jupyter notebook adult_income_midterm.ipynb
                                                ```

                                                Run all cells in order. The notebook is fully documented with explanations for each step.

                                                ---

                                                ## 💡 Learning Outcomes

                                                - Applied proper train/test splitting **before** EDA to prevent data leakage
                                                - - Identified and handled disguised missing values (`?` placeholders)
                                                  - - Practiced Min-Max scaling and understood its importance for algorithm fairness
                                                    - - Applied both Label Encoding and One-Hot Encoding for categorical data
                                                      - - Created meaningful engineered features from domain knowledge
                                                        - - Documented a full before/after data transformation pipeline
                                                         
                                                          - ---

                                                          ## 📬 Contact

                                                          **Alfredo Garza** — [LinkedIn](https://www.linkedin.com/in/alfredo-c-garza/) | phredogee71@gmail.com
                                                          
