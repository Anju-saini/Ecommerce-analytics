
# E-Commerce Sales & Customer Analytics

> **A full-pipeline Data Analytics & Machine Learning project on E-Commerce data.**  
> Covers Data Generation → Cleaning → EDA → 19 Visualizations → Churn Prediction → Revenue Forecasting

---

## Project Overview

This project simulates a real-world Indian e-commerce business (think Flipkart / Meesho scale) and performs end-to-end analytics across three datasets — **Customers**, **Products**, and **Orders** — spanning 3 years (2022–2024).

### Business Questions Answered

| # | Question | Method |
|---|---|---|
| 1 | Which product categories drive the most revenue? | EDA + Bar Charts |
| 2 | Who are our most valuable customers? | RFM Segmentation |
| 3 | When do customers buy the most? | Time-Series Analysis |
| 4 | Does discounting hurt profitability? | Scatter + Groupby |
| 5 | Which customers are likely to churn? | Random Forest Classifier |
| 6 | What will next month's revenue look like? | Gradient Boosting Regressor |

---

## Project Structure

```
ecommerce-analytics/
│
├── data/
│   ├── generate_data.py          # Generates synthetic realistic dataset
│   ├── customers.csv             # 2,000 customers (raw)
│   ├── products.csv              # 80 products across 8 categories (raw)
│   ├── orders.csv                # 8,000 orders 2022-2024 (raw)
│   ├── customers_clean.csv       # Cleaned + feature-engineered
│   ├── products_clean.csv        # Cleaned + margin added
│   ├── orders_clean.csv          # Cleaned + time features added
│   └── rfm.csv                   # RFM scores per customer
│
├── src/
│   ├── 01_data_cleaning.py       # Missing values, outliers, new features
│   ├── 02_eda.py                 # Full exploratory analysis
│   ├── 03_visualizations.py      # 14 EDA charts
│   ├── 04_ml_model.py            # Churn Prediction + Revenue Forecast (5 ML charts)
│   └── 05_insights_report.py     # Business insights summary
│
├── visualizations/               # 19 auto-generated PNG charts
│
├── run_all.py                    # Run full pipeline in one command
├── requirements.txt
└── README.md
```

---

## Datasets

| Dataset | Rows | Columns | Description |
|---|---|---|---|
| `customers.csv` | 2,000 | 10 | Customer demographics, location, segment |
| `products.csv` | 80 | 9 | Products across 8 categories with pricing |
| `orders.csv` | 8,000 | 13 | Transactions with status, payment, ratings |

**Categories:** Electronics, Clothing, Home & Kitchen, Books, Sports, Beauty, Toys, Grocery

**Customer Segments:** Premium, Regular, Budget, New

---

## Setup & Usage

### 1. Clone
```bash
git clone https://github.com/YOUR_USERNAME/ecommerce-analytics.git
cd ecommerce-analytics
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run everything
```bash
python run_all.py
```

Or step by step:
```bash
python data/generate_data.py        # Generate raw CSVs
python src/01_data_cleaning.py      # Clean & engineer features
python src/02_eda.py                # EDA insights in terminal
python src/03_visualizations.py     # Save all 14 EDA charts
python src/04_ml_model.py           # Train ML models + 5 charts
python src/05_insights_report.py    # Business summary report
```

---

## Data Cleaning (`01_data_cleaning.py`)

### Customers
| Problem | Fix |
|---|---|
| Missing `age` (60 rows) | Filled with median age |
| Missing `gender` (40 rows) | Filled with mode |
| Age outliers | Clipped to `[18, 75]` |
| New features | `tenure_days`, `age_group` |

### Products
| New Feature | Description |
|---|---|
| `margin_pct` | `(price - cost) / price × 100` |
| `price_tier` | Budget / Mid-range / Premium / Luxury |

### Orders
| Problem | Fix |
|---|---|
| Missing `delivery_days` | Filled with median (delivered orders only) |
| Missing `rating` | Filled with category median |
| New features | `year`, `month`, `quarter`, `weekday`, `discount_band`, `is_loss` |

---

## EDA Highlights (`02_eda.py`)

### Key KPIs
| Metric | Value |
|---|---|
| Total Revenue | ₹9.1M+ |
| Overall Gross Margin | ~40% |
| Delivery Success Rate | 72% |
| Return Rate | 12% |
| Cancellation Rate | 10% |

### Top Findings
- **UPI** is the #1 payment method (30% of orders)
- **Q4 (Oct-Dec)** contributes ~25% of annual revenue — festival season effect
- **Premium segment** customers have the highest Average Order Value
- **High discounts (>20%)** drag average profit down significantly

---

## Visualizations (`03_visualizations.py`)

14 charts covering sales, customers, products, and geography:

| # | Chart | Insight Type |
|---|---|---|
| 01 | Monthly Revenue Trend | Time-series with Q4 highlights |
| 02 | Revenue by Category | Best/worst performing categories |
| 03 | Order Status Distribution | Delivery health |
| 04 | Segment Revenue by Status | Who returns/cancels more |
| 05 | Month × Category Heatmap | Seasonal demand by category |
| 06 | Top 10 Cities Revenue | Geographic hotspots |
| 07 | Payment Method Analysis | Orders + Revenue split |
| 08 | Discount vs Profit Scatter | Profitability at risk |
| 09 | Age Group × Gender Revenue | Demographic targeting |
| 10 | Day-of-Week Patterns | When do customers shop? |
| 11 | RFM Segment Distribution | Customer quality breakdown |
| 12 | Delivery Days Analysis | SLA performance |
| 13 | Rating vs Margin Bubble | Product portfolio health |
| 14 | YoY Revenue Growth | Business trajectory |

---

## Machine Learning (`04_ml_model.py`)

### Task A — Customer Churn Prediction

**Definition:** A customer who has not placed an order in the last 180 days is labelled as churned.

**Features used:**
```
age, tenure_days, total_orders, total_revenue, avg_order_value,
total_discount, cancelled_orders, returned_orders, avg_rating,
unique_categories, recency, frequency, monetary, rfm_score,
is_subscribed, gender_enc, segment_enc
```

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | ~62% | ~0.67 |
| Random Forest | ~75% | ~0.82 |

**Key churn drivers (feature importance):** `recency`, `rfm_score`, `total_orders`, `tenure_days`

**ML Charts:** Confusion Matrix, ROC Curve, Feature Importance

---

### Task B — Monthly Revenue Forecasting

**Approach:** Time-series regression using lag features and rolling statistics.

**Features:** `t` (time index), `month`, `quarter`, `is_q4`, `lag1`, `lag2`, `lag3`, `roll3_mean`

| Model | MAE | R² |
|---|---|---|
| Linear Regression | ~₹35,000 | ~0.70 |
| Gradient Boosting | ~₹22,000 | ~0.87 |

**ML Charts:** Forecast vs Actual plot, Residual analysis

---

## RFM Segmentation

Customers are scored on:
- **R**ecency — How recently did they buy?
- **F**requency — How often do they buy?
- **M**onetary — How much do they spend?

| Segment | Description | Action |
|---|---|---|
| Champions | High R+F+M | Reward & upsell |
| Loyal | Medium-high scores | Nurture with offers |
| At Risk | Declining frequency | Re-engagement campaign |
| Lost | Low on all dimensions | Win-back or let go |

---

## Business Recommendations

1. **Electronics & Home & Kitchen** are top revenue drivers — prioritise inventory and marketing here
2. **Re-engage At Risk & Lost customers** via personalised email/push campaigns using RFM targeting
3. **Cap discounts at 10%** — deep discounts erode margin without proportional volume gains
4. **Invest in Q4 capacity** — festival season drives outsized revenue; stock-outs are costly
5. **Fix delivery SLA** — 45%+ orders take more than 7 days; this is a key churn driver
6. **UPI is dominant** — ensure gateway reliability; even 30-min downtime impacts thousands of orders
7. **Premium segment loyalty programme** — highest AOV customers are most profitable to retain

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, aggregation |
| `numpy` | Numerical operations |
| `matplotlib` | Base plotting |
| `seaborn` | Statistical visualisations |
| `scikit-learn` | ML models, preprocessing, evaluation |

---

## Output Files After Running

```
data/
  customers_clean.csv
  products_clean.csv
  orders_clean.csv
  rfm.csv

visualizations/        ← 19 charts
  01_monthly_revenue_trend.png
  02_revenue_by_category.png
  03_order_status_pie.png
  04_segment_revenue.png
  05_month_category_heatmap.png
  06_top_cities_revenue.png
  07_payment_methods.png
  08_discount_vs_profit.png
  09_age_gender_revenue.png
  10_weekly_order_pattern.png
  11_rfm_segments.png
  12_delivery_analysis.png
  13_margin_vs_rating.png
  14_yoy_growth.png
  15_churn_confusion_matrix.png
  16_churn_roc_curve.png
  17_churn_feature_importance.png
  18_revenue_forecast.png
  19_residual_analysis.png
```

---

## Future Improvements

- [ ] Connect to a real dataset (e.g., Brazilian E-Commerce on Kaggle)
- [ ] Build an interactive Streamlit dashboard
- [ ] Add XGBoost / LightGBM for better churn prediction
- [ ] Add customer lifetime value (CLV) modelling
- [ ] Product recommendation engine (collaborative filtering)
- [ ] A/B test analysis module for discount strategies

---

## Author

  
Data Analyst | Python · Pandas · Scikit-learn · Seaborn  
[LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)

---

## License

Open source under the [MIT License](LICENSE).
