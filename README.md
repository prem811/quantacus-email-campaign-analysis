# Email Campaign Optimization - Case Study

## Overview:
In this case study, we aim to analyze an email marketing campaign and build a machine learning model to predict which users are most likely to click on a link inside the email. The goal is to optimize the email sending strategy to maximize the **Click-Through Rate (CTR)** based on historical data.

### Key Objectives:
1. **Data Preprocessing**: Handling missing values and encoding categorical features.
2. **Modeling**: Building and training a machine learning model (Random Forest) to predict whether a user will click on the email link.
3. **CTR Simulation**: Simulating how the model improves CTR compared to random email selection.
4. **Segment Analysis**: Analyzing how different segments of users (e.g., user country, email version) performed in the campaign.

## Steps Taken:
### 1. Data Preprocessing:
- **Loaded the data**: The datasets containing email information, user interactions (opens and clicks) were loaded into pandas DataFrames.
- **Handled missing values**: We checked for any missing or null values in the dataset and handled them accordingly.
- **Created new columns**: Added two new columns to track if an email was opened or clicked by a user:
    ```python
    email_df['opened'] = email_df['email_id'].isin(opened_df['email_id']).astype(int)
    email_df['clicked'] = email_df['email_id'].isin(clicked_df['email_id']).astype(int)
    ```
- **One-Hot Encoding**: Categorical variables like `email_text`, `email_version`, and `user_country` were converted into numerical features using **One-Hot Encoding**.

### 2. Modeling:
- **Built a Random Forest Classifier**: 
    - A Random Forest model was trained to predict whether a user will click on the email link based on features like `email_text`, `email_version`, and `user_country`.
    - We handled class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique) to ensure the model can better predict the minority class (clicked).
    ```python
    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)
    ```
- **Evaluation**: The model was evaluated using **precision**, **recall**, **F1-score**, and **confusion matrix**.
    ```python
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    ```

### 3. CTR Simulation:
- **Simulated Click-Through Rate (CTR)** improvement:
    - We predicted the probability of users clicking the email link using the trained model.
    - We selected the top **20%** of users predicted to click on the email and calculated the CTR for those users.
    - This simulated **CTR improvement** was compared against a random email selection approach.
    ```python
    y_probs = rf_model.predict_proba(X_test)[:, 1]
    threshold = np.percentile(y_probs, 100 * (1 - top_n_percent))
    ```

### 4. Segment Analysis:
- **Analyzed different segments**: We analyzed how different user segments performed in terms of open and click rates, focusing on:
    - **User country**: Which countries had higher click rates?
    - **Email version**: Did personalized emails perform better than generic ones?
    ```python
    country_group = email_df.groupby('user_country')[['opened', 'clicked']].mean() * 100
    sns.barplot(x='user_country', y='opened', data=country_group, label='Open Rate', color='skyblue')
    sns.barplot(x='user_country', y='clicked', data=country_group, label='Click Rate', color='orange')
    ```

## Results:
- The **Random Forest model** successfully predicted which users are more likely to click on the link in an email, improving the overall **CTR** compared to random email selection.
- **CTR Simulation**: The model's prediction of user clicks led to a higher CTR compared to randomly selecting users.
- **Segment Analysis**: Some user countries and email versions showed better engagement, which could help optimize future email campaigns.

## Conclusion:
- By using machine learning models like **Random Forest** and targeting users predicted to click based on the model, email campaigns can achieve a higher CTR.
- Further improvements could involve **hyperparameter tuning** or exploring more sophisticated models like **XGBoost**.

## Files in this Repository:
- `Quantacus_Email_Campaign_Analysis.ipynb`: The notebook containing the full analysis, including data preprocessing, modeling, and evaluation.

## Installation:
To run the notebook, install the required libraries using the following:

```bash
pip install -r requirements.txt



Requirements:
imbalanced-learn

scikit-learn

pandas

matplotlib

seaborn

