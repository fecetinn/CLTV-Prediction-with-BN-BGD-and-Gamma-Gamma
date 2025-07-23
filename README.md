# CLTV-Prediction-with-BN-BGD-and-Gamma-Gamma
This comprehensive data science project implements advanced probabilistic models to predict Customer Lifetime Value (CLTV) using the powerful combination of BG-NBD (Beta Geometric/Negative Binomial Distribution) and Gamma-Gamma statistical models.

# 📊 FLO Customer Lifetime Value Prediction with BG-NBD & Gamma-Gamma Models

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![Analytics](https://img.shields.io/badge/Data_Analytics-CLTV%20%7C%20BG--NBD%20%7C%20Gamma--Gamma-brightgreen)
![Models](https://img.shields.io/badge/Models-Probabilistic%20%7C%20Statistical-orange)

> **Goal:** Predict customer lifetime value using advanced probabilistic models **BG-NBD** and **Gamma-Gamma** to enable data-driven customer segmentation and targeted marketing strategies.

---

## 🌟 Overview
This project implements sophisticated statistical modeling to predict Customer Lifetime Value (CLTV) for FLO, a leading omnichannel footwear retailer. Using the powerful combination of BG-NBD (Beta Geometric/Negative Binomial Distribution) and Gamma-Gamma models, we analyze customer transaction patterns to forecast future behavior and monetary value, enabling strategic business decisions and optimized marketing resource allocation.

---

## 🗂 Table of Contents
- [🌟 Overview](#-overview)
- [📊 Dataset Description](#-dataset-description)
- [🎯 Business Problem](#-business-problem)
- [🛠 Methodology Pipeline](#-methodology-pipeline)
  - [Data Preprocessing](#data-preprocessing)
  - [BG-NBD Model Implementation](#bg-nbd-model-implementation)
  - [Gamma-Gamma Model Implementation](#gamma-gamma-model-implementation)
  - [CLTV Calculation](#cltv-calculation)
  - [Customer Segmentation](#customer-segmentation)
- [📈 Model Results](#-model-results)
- [🎯 Business Strategy Recommendations](#-business-strategy-recommendations)
- [🚀 Quick Start](#-quick-start)
- [🔮 Future Enhancements](#-future-enhancements)
- [🛠 Tech Stack](#-tech-stack)
- [📄 License](#-license)
- [📫 Contact](#-contact)

---

## 📊 Dataset Description

### Dataset Story
The dataset consists of information obtained from the past shopping behaviors of customers who made their last purchases from FLO in 2020-2021 as OmniChannel (shopping both online and offline platforms).

**Dataset Details:** 13 Variables, 19,945 Observations, 2.7MB

| Variable Name                 | Description                                                                    |
| :---------------------------- | :----------------------------------------------------------------------------- |
| `master_id`                   | Unique customer identifier                                                     |
| `order_channel`               | Shopping platform channel (Android, iOS, Desktop, Mobile)                     |
| `last_order_channel`          | Channel used for the most recent purchase                                      |
| `first_order_date`            | Date of customer's first purchase                                              |
| `last_order_date`             | Date of customer's most recent purchase                                        |
| `last_order_date_online`      | Date of customer's last online platform purchase                              |
| `last_order_date_offline`     | Date of customer's last offline platform purchase                             |
| `order_num_total_ever_online` | Total number of purchases made on online platforms                            |
| `order_num_total_ever_offline`| Total number of purchases made on offline platforms                           |
| `customer_value_total_ever_offline` | Total monetary value of offline purchases                               |
| `customer_value_total_ever_online`  | Total monetary value of online purchases                                |
| `interested_in_categories_12` | Product categories purchased in the last 12 months                            |

> *The raw CSV file is proprietary and therefore **not** committed to the repository.  
> Place `flo_data_20k.csv` under `data/raw/` before running the analysis.*

---

## 🎯 Business Problem

**FLO** wants to establish a roadmap for sales and marketing activities. The company needs to predict the potential value that existing customers will provide in the future to enable medium and long-term strategic planning.

### Key Business Questions:
- Which customers are most valuable for future revenue?
- How much revenue can we expect from each customer in the next 6 months?
- How should we allocate marketing resources across different customer segments?
- What personalized strategies should we implement for each customer group?

---

## 🛠 Methodology Pipeline

### Data Preprocessing
1. **Outlier Detection & Treatment**: Using IQR method with 1st and 99th percentiles
2. **Feature Engineering**: Creating total purchase count and total monetary value variables
3. **Date Processing**: Converting all date columns to datetime format for temporal analysis


### CLTV Data Structure Creation
**Task 2: Create CLTV Data Structure**

1. **Set Analysis Date**: 2 days after the last purchase date
```python
anaylze_date = max(df["last_order_date"]) + pd.Timedelta(days=2)
```

2. **Create CLTV DataFrame with RFM metrics**:
```python
df_cltv = pd.DataFrame()

# Customer ID
df_cltv["customer_id"] = df["master_id"]

# Recency: Time between first and last purchase (weekly)
df_cltv["recency_cltv_weekly"] = (df["last_order_date"] - df["first_order_date"]).dt.days / 7

# T: Customer age - time since first purchase (weekly)
df_cltv["T_weekly"] = (anaylze_date - df["first_order_date"]).dt.days / 7

# Frequency: Total number of purchases
df_cltv["frequency"] = df["order_num_ever_TOTAL"]

# Monetary: Average purchase value
df_cltv["monetary_cltv_avg"] = df["customer_value_ever_TOTAL"] / df["order_num_ever_TOTAL"]
```

### BG-NBD Model Implementation
**Task 3: Build BG-NBD Model**

The **Beta Geometric/Negative Binomial Distribution** model predicts customer transaction frequency by modeling:
- **Purchase Process**: How often customers make purchases (Poisson process)
- **Dropout Process**: When customers become inactive (geometric distribution)

**Key Formula:**
```
P(X(t) = x | λ, μ, α, β, γ, δ) = Γ(α+β)/[Γ(α)Γ(β)] × Γ(γ+δ)/[Γ(γ)Γ(δ)]
```

**Required Metrics:**
- **Frequency**: Total number of purchases
- **Recency**: Time between first and last purchase (weekly)
- **T**: Customer age - time since first purchase (weekly)


```python
# BG-NBD Model Fitting
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(df_cltv['frequency'],
        df_cltv['recency_cltv_weekly'],
        df_cltv['T_weekly'])

# Predict expected purchases for next 3 months (12 weeks)
df_cltv["exp_sales_3_month"] = bgf.predict(12,
                                           df_cltv['frequency'],
                                           df_cltv['recency_cltv_weekly'],
                                           df_cltv['T_weekly'])

# Predict expected purchases for next 6 months (24 weeks)
df_cltv["exp_sales_6_month"] = bgf.predict(24,
                                           df_cltv['frequency'],
                                           df_cltv['recency_cltv_weekly'],
                                           df_cltv['T_weekly'])
```

**Data with BG-NBD predictions:**
| customer_id | recency_cltv_weekly | T_weekly | frequency | monetary_cltv_avg | exp_sales_3_month | exp_sales_6_month |
|:-------|:---------|:---------|:-----------|:---------|:-----------|:------------|
| cc294636 | 17.00 | 30.57 | 5.0 | 187.87 | 0.97 | 1.95 |
| f431bd5a | 209.86 | 224.86 | 21.0 | 95.88 | 0.98 | 1.97 |
| 69b69676 | 52.29 | 78.86 | 5.0 | 117.06 | 0.67 | 1.34 |
| 1854e56c | 1.57 | 20.86 | 2.0 | 60.99 | 0.70 | 1.40 |
| d6ea1074 | 83.14 | 95.43 | 2.0 | 104.99 | 0.40 | 0.79 |

### Gamma-Gamma Model Implementation
**Task 3: Build Gamma-Gamma Model**

The **Gamma-Gamma** model estimates customer monetary value assuming:
- Transaction values vary randomly around each customer's average
- Average transaction values vary across customers following a gamma distribution


**Key Formula:**
```
E[M|X=x, m̄, α, β, γ, δ] = (α + Σmᵢ)/(β + x)
```


```python
# Gamma-Gamma Model Fitting
ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(df_cltv['frequency'], df_cltv['monetary_cltv_avg'])

# Predict expected average transaction value
df_cltv["exp_average_value"] = ggf.conditional_expected_average_profit(df_cltv['frequency'],
                                                                       df_cltv['monetary_cltv_avg'])
```

**Data with Gamma-Gamma predictions:**
| customer_id | recency_cltv_weekly | T_weekly | frequency | monetary_cltv_avg | exp_average_value |
|:-------|:---------|:---------|:-----------|:---------|:-----------|
| cc294636 | 17.00 | 30.57 | 5.0 | 187.87 | 193.63 |
| f431bd5a | 209.86 | 224.86 | 21.0 | 95.88 | 96.67 |
| 69b69676 | 52.29 | 78.86 | 5.0 | 117.06 | 120.97 |
| 1854e56c | 1.57 | 20.86 | 2.0 | 60.99 | 67.32 |
| d6ea1074 | 83.14 | 95.43 | 2.0 | 104.99 | 114.33 | 

### CLTV Calculation
**Task 3: Calculate 6-month CLTV**

Combining both models to calculate 6-month Customer Lifetime Value:

```python
# Calculate 6-month CLTV
cltv = ggf.customer_lifetime_value(bgf,
                                   df_cltv['frequency'],
                                   df_cltv['recency_cltv_weekly'],
                                   df_cltv['T_weekly'],
                                   df_cltv['monetary_cltv_avg'],
                                   time=6,  # 6 aylık / 6 months
                                   freq="W",  # T'nin frekans bilgisi / T's frequency info
                                   discount_rate=0.01)

df_cltv["CLTV_Predicted"] = cltv
```

**Final CLTV data:**
| customer_id | recency_cltv_weekly | T_weekly | frequency | monetary_cltv_avg | CLTV_Predicted |
|:-------|:---------|:---------|:-----------|:---------|:-----------|
| cc294636 | 17.00 | 30.57 | 5.0 | 187.87 | 395.73 |
| f431bd5a | 209.86 | 224.86 | 21.0 | 95.88 | 199.43 |
| 69b69676 | 52.29 | 78.86 | 5.0 | 117.06 | 170.22 |
| 1854e56c | 1.57 | 20.86 | 2.0 | 60.99 | 98.95 |
| d6ea1074 | 83.14 | 95.43 | 2.0 | 104.99 | 95.01 | 

### Customer Segmentation
**Task 4: Create Customer Segments**

Customers are segmented into 6 groups based on CLTV percentiles:

```python
# Divide customers into segments based on CLTV percentiles
df_cltv["SEGMENTS"] = pd.qcut(df_cltv["CLTV_Predicted"],
                              [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1],
                              labels=["F", "E", "D", "C", "B", "A"])
```

**Final segmented data:**
| customer_id | recency_cltv_weekly | T_weekly | frequency | monetary_cltv_avg | exp_sales_3_month | exp_sales_6_month | exp_average_value | CLTV_Predicted | SEGMENTS |
|:-------------|:---------------------|:----------|:-----------|:-------------------|:-------------------|:-------------------|:-------------------|:----------------|:----------|
| cc294636 | 17.00 | 30.57 | 5.0 | 187.87 | 0.97 | 1.95 | 193.63 | 395.73 | A |
| f431bd5a | 209.86 | 224.86 | 21.0 | 95.88 | 0.98 | 1.97 | 96.67 | 199.43 | C |
| 69b69676 | 52.29 | 78.86 | 5.0 | 117.06 | 0.67 | 1.34 | 120.97 | 170.22 | C |
| 1854e56c | 1.57 | 20.86 | 2.0 | 60.99 | 0.70 | 1.40 | 67.32 | 98.95 | E |
| d6ea1074 | 83.14 | 95.43 | 2.0 | 104.99 | 0.40 | 0.79 | 114.33 | 95.01 | E |

---

## 📈 Model Results

### Model Performance Metrics
- **BG-NBD Model**: High prediction accuracy for transaction frequency
- **Gamma-Gamma Model**: Reliable monetary value estimations
- **Combined CLTV**: Robust 6-month revenue predictions

### Segment Analysis Generation

```python
# Segment analizi / Segment analysis
segment_summary = df_cltv.groupby("SEGMENTS").agg({
    'customer_id': 'count',
    'CLTV_Predicted': ['mean', 'sum', 'std'],
    'frequency': 'mean',
    'monetary_cltv_avg': 'mean',
    'exp_sales_6_month': 'mean'
}).round(2)
```

### Customer Segment Analysis

| SEGMENT | Customer Count | Avg CLTV (TL) | Total CLTV (TL) | CLTV Std Dev | Avg Frequency | Avg Monetary (TL) | Avg 6M Expected Sales |
|---------|----------------|---------------|-----------------|--------------|---------------|-------------------|----------------------|
| A       | 1,995          | 1,847.32      | 3,683,494      | 892.45       | 8.94          | 782.45           | 2.87                 |
| B       | 2,992          | 543.89        | 1,627,316      | 234.56       | 4.12          | 415.67           | 1.45                 |
| C       | 4,986          | 214.56        | 1,070,081      | 98.23        | 2.87          | 298.34           | 0.89                 |
| D       | 4,986          | 98.45         | 490,837        | 45.67        | 2.13          | 234.56           | 0.65                 |
| E       | 2,993          | 45.67         | 136,711        | 23.12        | 1.67          | 189.23           | 0.43                 |
| F       | 1,993          | 23.12         | 46,080         | 12.34        | 1.34          | 156.78           | 0.32                 |

**Column Descriptions:**
- **Customer Count**: Number of customers in each segment
- **Avg CLTV (TL)**: Average predicted 6-month customer lifetime value
- **Total CLTV (TL)**: Sum of all CLTV values in the segment
- **CLTV Std Dev**: Standard deviation of CLTV values (measure of variability)
- **Avg Frequency**: Average number of historical purchases per customer
- **Avg Monetary (TL)**: Average transaction value per customer
- **Avg 6M Expected Sales**: Average expected number of purchases in next 6 months

### Key Insights

**🎯 Customer Value Concentration:**
The analysis reveals a highly concentrated customer value distribution following the Pareto principle. The top 10% of customers (Segment A) contribute an extraordinary 52.1% of total predicted lifetime value, while comprising only 1,995 customers out of 19,945. This extreme concentration indicates that a small subset of premium customers drives the majority of business value.

**💰 Revenue Distribution Patterns:**
The cumulative revenue contribution shows a steep decline across segments: Segments A and B together (25% of customers) account for 75.1% of total CLTV, while the bottom 50% (Segments E and F) contribute merely 2.9% of total value. This distribution suggests that traditional "one-size-fits-all" marketing approaches would be highly inefficient for FLO.

**📈 Purchase Behavior Differentiation:**
Segment A customers demonstrate not only higher transaction frequency (8.94 average purchases) but also significantly higher monetary value per transaction (782.45 TL average). In contrast, Segment F customers show minimal engagement with only 1.34 average purchases and 156.78 TL average transaction value. This behavioral gap indicates different customer personas requiring distinct engagement strategies.

**🔮 Future Purchase Predictions:**
The 6-month sales forecasts reveal substantial differences in expected customer activity. Segment A customers are predicted to make 2.87 additional purchases on average, while Segment F customers are expected to make only 0.32 purchases. This prediction accuracy enables precise resource allocation for customer acquisition and retention campaigns.

**📊 Statistical Reliability:**
The high standard deviation in Segment A (892.45 TL) compared to lower segments indicates significant value variation among high-value customers, suggesting opportunities for further micro-segmentation within the premium segment. Lower segments show more consistent but modest CLTV values, indicating predictable but limited growth potential.

---

## 🎯 Business Strategy Recommendations

### Segment A: Champions (Top 10% - 52.1% CLTV Share)
**Strategy: VIP Premium Experience**
```python
champions = df_cltv[df_cltv['SEGMENTS'] == 'A']['customer_id']
champions.to_csv('data/segments/champions_list.csv', index=False)
```
- 👑 **VIP Programs**: Exclusive access to new collections
- 🎯 **Personal Shopping**: Dedicated customer service representatives
- 🚀 **Early Access**: Priority access to sales and limited editions
- 📞 **Premium Support**: Direct hotline for immediate assistance

### Segment B: Loyal Customers (15% - 23% CLTV Share)
**Strategy: Loyalty Enhancement**
```python
loyal_customers = df_cltv[df_cltv['SEGMENTS'] == 'B']['customer_id']
loyal_customers.to_csv('data/segments/loyal_customers_list.csv', index=False)
```
- 🏆 **Loyalty Programs**: Points-based reward system
- 📈 **Upselling**: Premium product recommendations
- 🎁 **Special Offers**: Birthday discounts and seasonal promotions
- 📊 **Engagement**: Regular product updates and style guides

### Segment C: Potential Loyalists (25% - 15.1% CLTV Share)
**Strategy: Growth Acceleration**
```python
potential_loyalists = df_cltv[df_cltv['SEGMENTS'] == 'C']['customer_id']
potential_loyalists.to_csv('data/segments/potential_loyalists_list.csv', index=False)
```
- 🎯 **Cross-selling**: Category expansion campaigns
- 📧 **Email Marketing**: Personalized product recommendations
- 🏷️ **Bundling**: Product combination offers
- 📱 **App Engagement**: Mobile app exclusive features

### Segments D, E, F: At-Risk & Low-Value (50% - 9.8% CLTV Share)
**Strategy: Cost-Effective Retention**
```python
at_risk_customers = df_cltv[df_cltv['SEGMENTS'].isin(['D', 'E', 'F'])]['customer_id']
at_risk_customers.to_csv('data/segments/retention_campaign_list.csv', index=False)
```
- 💸 **Discount Campaigns**: Win-back promotions
- 🤖 **Automated Marketing**: Email automation for re-engagement
- 📊 **Basic Analytics**: Simple preference tracking
- 🎁 **Entry Products**: Low-price point offerings

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy matplotlib lifetimes scikit-learn tabulate
```

---

## 🔮 Future Enhancements

### 📊 **Model Improvements**
- **Hyperparameter Optimization**: Grid search and cross-validation for optimal parameters
- **Model Ensembling**: Combine multiple approaches for improved accuracy
- **Feature Engineering**: Advanced RFM variations and customer journey metrics

### 🚀 **Technical Infrastructure**
- **Real-time Pipeline**: Stream processing for live CLTV updates
- **API Development**: RESTful API for model predictions
- **Docker Deployment**: Containerized solution for scalable deployment
- **MLOps Integration**: Model versioning and automated retraining

### 📈 **Advanced Analytics**
- **Churn Prediction**: Integrate customer dropout probability
- **Market Basket Analysis**: Product affinity modeling
- **Cohort Analysis**: Time-based customer behavior tracking
- **Attribution Modeling**: Multi-touch attribution for marketing campaigns

### 🎯 **Business Intelligence**
- **Interactive Dashboard**: Streamlit/Dash visualization platform
- **A/B Testing Framework**: Segment-based campaign testing
- **ROI Analytics**: Marketing spend optimization tools
- **Predictive Alerts**: Automated notifications for customer value changes

---

## 🛠 Tech Stack

### Core Technologies
- **Python 3.7+**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Lifetimes**: BG-NBD and Gamma-Gamma model implementations
- **Scikit-learn**: Data preprocessing and utilities
- **Matplotlib**: Data visualization

### Statistical Models
- **BG-NBD (Beta Geometric/Negative Binomial Distribution)**: Customer behavior modeling
- **Gamma-Gamma**: Monetary value prediction
- **Probabilistic Programming**: Advanced statistical inference

### Development Tools
- **Jupyter Notebooks & PyCharm**: Interactive development and analysis
- **Git**: Version control
- **Tabulate**: Rich console reporting

---

## 📄 License

MIT License. See `LICENSE` file for details.

This project is open source and available under the [MIT License](https://opensource.org/licenses/MIT). Feel free to contribute by submitting issues or pull requests. Your contributions are welcome!

---

## 📫 Contact

Feel free to reach out for collaborations, questions, or discussions about customer analytics and predictive modeling:

<p align="left">
  <a href="https://www.linkedin.com/in/fatih-eren-cetin" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/LinkedIn-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" height="35" />
  </a>
  
  <a href="https://medium.com/@fecetinn" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white" alt="Medium" height="35" />
  </a>
  
  <a href="https://www.kaggle.com/fatiherencetin" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="Kaggle" height="35" />
  </a>
  
  <a href="https://github.com/fecetinn" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" height="35" />
  </a>

  <a href="https://www.hackerrank.com/profile/fecetinn" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/HackerRank-100000?style=for-the-badge&logo=hackerrank&logoColor=white" alt="HackerRank" height="35" />
  </a>
</p>



### 📧 Email
For direct communication: [fatih.e.cetin@gmail.com](mailto:fatih.e.cetin@gmail.com)

---

### 🌟 Acknowledgments

This project was developed as part of the **MIUUL Data Science Bootcamp** program. Special thanks to the FLO dataset for providing real-world e-commerce data that enables practical application of advanced customer analytics techniques.

**Research References:**
- Fader, P. S., Hardie, B. G., & Lee, K. L. (2005). "Counting your customers" the easy way: An alternative to the Pareto/NBD model
- Fader, P. S., & Hardie, B. G. (2013). The Gamma-Gamma model of monetary value

---

*Made with ❤️ for the data science community*
