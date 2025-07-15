# NYC Airbnb Data Analytics Project

## Overview
This project presents an in-depth exploratory data analysis (EDA) of over **48,000 Airbnb listings in New York City**, sourced from the `AB_NYC_2019.csv` dataset. The objective is to uncover meaningful insights into **pricing behavior, occupancy trends, listing popularity, and geographic distribution** using robust data cleaning, visualization, and statistical techniques. All work was conducted in a fully reproducible **Google Colab** environment.

---

## 1. Dataset Summary
- **Total listings**: 48,895 → **Post-cleaning**: 48,881
- **Key fields**: `price`, `number_of_reviews`, `reviews_per_month`, `availability_365`, `minimum_nights`, `neighbourhood_group`, `room_type`, `host_id`, `name`, `latitude`, `longitude`

---

## 2. Data Cleaning & Preprocessing
- Replaced missing values in `name` and `host_name` with `"Unnamed Listing"` and `"Unknown Host"`
- Filled missing `reviews_per_month` with 0
- Parsed `last_review` to datetime
- Removed listings with:
  - `price` > $10,000 or < $0
  - `minimum_nights` > 365
- Final dataset: **48,881 rows × 16 columns**

---

## 3. Price Insights by Neighborhood
- **Top 5 most expensive neighborhoods**:
  - Tribeca: $490.64  
  - Sea Gate: $487.86  
  - Riverdale: $442.09  
  - Battery Park City: $367.09  
  - Flatiron District: $341.92
- **Bottom 5 cheapest**:
  - Bull's Head: $47.33  
  - Hunts Point: $50.50  
  - Tremont: $51.55  
  - Soundview: $53.47  
  - Bronxdale: $57.11
- Combined bar charts revealed **price skewness** and **sample size sensitivity**

---

## 4. Price Distribution by Borough
- Highest avg price: **Manhattan ($196.88)**  
- Lowest avg price: **Bronx ($87.50)**
- Boxplots showed:
  - Manhattan: **widest variance**
  - Staten Island: **tight clustering**

---

## 5. Correlation Analysis
- Pearson matrix showed:
  - **Strongest positive**: `reviews_per_month` vs `number_of_reviews` → 0.59  
  - **Strongest negative**: `minimum_nights` vs `reviews_per_month` → -0.15
- Scatter plots and pair plots confirmed trends

---

## 6. Geographic Listing Visualization
- Generated a borough-colored **scatter map** with lat/lon data
- Pie chart of listing counts per borough
- Key findings:
  - **Manhattan & Brooklyn dominate** density
  - **Staten Island is sparsely listed**, mostly northern tip

---

## 7. Price Heatmap & Distribution (< $1000)
- Filtered outliers > $1000 (kept 99.4% of data)
- Price-color scatter map shows spatial pricing patterns
- Overlaid borough means/medians on violin and box plots
- Histograms with KDEs highlight borough-specific distributions

---

## 8. Word Cloud of Listing Titles
- Tokenized and cleaned 277,845 words from listing `name`
- Removed stopwords and punctuation
- **Top words**: "room", "private", "cozy", "studio", "manhattan", "brooklyn"
- Word cloud captures **host branding themes**

---

## 9. Busiest Hosts in NYC
- Aggregated by `host_id` and `host_name`
- **Top host**: Sonder (NYC) with 327 listings
- Trends among top 10 hosts:
  - **Manhattan-heavy** listings
  - Higher availability (~240 days/year)
  - Lower reviews/month → suggests **longer-term or professional rentals**
- Visuals:
  - Multi-metric dashboard
  - Correlation heatmaps
  - Bar & scatter plots by host

---

## 10. Custom Visual Insights

### Insight 1: Price vs. Review Frequency
- Heatmap shows avg total reviews across:
  - **Price buckets**
  - **Monthly review frequency**
- Sweet spot: **$51–100 listings with 1–3 reviews/month** had the most total reviews

### Insight 2: Borough Performance Radar Matrix
- Normalized radar comparison on:
  - Avg Price  
  - Reviews/Month  
  - Reviews/Listing  
  - Estimated Occupancy (via availability_365)
- Borough Profiles:
  - **Manhattan**: High price, low occupancy
  - **Brooklyn**: Balanced price & highest occupancy
  - **Bronx**: Lowest price, strong occupancy
  - **Staten Island**: High reviews/month, low occupancy
  - **Queens**: Balanced across all metrics

---

## Tools & Technologies
- **Python**: pandas, numpy, seaborn, matplotlib, re, wordcloud
- **Google Colab**: Jupyter Notebook-based EDA
- **Visualization**: colorblind-safe palettes, annotated charts, statistical overlays

---

## Outcome
This project presents a **comprehensive, data-driven overview** of Airbnb activity in NYC using real data and rigorous visual storytelling. It is valuable to:

- **Hosts**: optimize pricing, identify demand patterns
- **Guests**: spot value areas
- **Analysts & Policymakers**: understand market saturation, affordability, and geographic trends

The work demonstrates **professional-level data cleaning, transformation, visualization, and statistical interpretation**, fully reproducible through the linked notebook.

