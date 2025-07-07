# Data-Driven-Menu-Analysis
This repository presents a data-driven approach to optimizing fast-food menu performance by balancing item popularity with profitability. Using a dataset of 10,000 transactions, the project applies data mining techniques to uncover patterns, estimate profitability, and derive actionable recommendations.

# Methods & Techniques
- Data Cleaning & Feature Engineering: Preprocessing, temporal feature extraction, and standardization.
- Cost Estimation: Regression-based model using price tiers and volume tiers to estimate item costs.
- Menu Engineering: Categorization of menu items into 'Stars', 'Plowhorses', 'Puzzles', and 'Dogs' using Kasavana and Smith’s model.
- Clustering: K-means clustering based on revenue, margin, quantity, and price.
- Predictive Modeling: Linear, Ridge, Lasso, and Random Forest models for sales and profit prediction.
- Association Rule Mining: Apriori algorithm to detect frequent itemsets and bundling opportunities.
- Data Visualization: Clear visuals illustrating key insights and item positioning.

# Prerequisites
- Python 3.x
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, mlxtend, networkx

# Usage
Place your dataset (e.g., Balaji Fast Food Sales.csv) in the root directory and run: python main_script.py

# Output Files
- cleaned_menu_sales.csv: Cleaned transactional data
- complete_menu_analysis.csv: Final item-level summary
- menu_performance_matrix.png: Popularity vs. profitability chart
- sales_forecast.png: Forecasting visualization
- association_rules.png: Co-purchase patterns
- menu_optimization_strategy.png: Executive summary of recommendations

A full list of outputs and visualizations is available in the /output folder.
