import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# For regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# For clustering
from sklearn.cluster import KMeans

# For association rules
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from itertools import combinations

# For visualizations
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

# 1. Data Inspection
def inspect_dataset(file_path):
    print("Loading dataset...")
    try:
        df = pd.read_csv(file_path, delimiter=",", encoding="utf-8", on_bad_lines='skip', low_memory=False)
        print("Successfully loaded dataset")
    except Exception as e:
        print(f"Error: {e}")
        try:
            df = pd.read_csv(file_path, encoding='utf-8', engine='python', on_bad_lines='skip')
        except:
            try:
                df = pd.read_excel(file_path)
            except Exception as final_error:
                raise Exception(f"Could not read file: {final_error}")

    print("\n--- Dataset Overview ---")
    print(f"Total records: {df.shape[0]}")
    print(f"Total columns: {df.shape[1]}")
    print(df.head())
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("\n--- Missing Values ---")
        print(missing_values[missing_values > 0])
    
    # Check for data types
    print("\n--- Data Types ---")
    print(df.dtypes)
    
    return df

# 2. Data Cleaning
def clean_data(df):
    df = df.copy()
    print("\n--- Cleaning Data ---")
    
    # Fill missing values
    df['transaction_type'] = df['transaction_type'].fillna('Unknown')
    df['time_of_sale'] = df['time_of_sale'].fillna('Unknown')
    df['received_by'] = df['received_by'].fillna('Unknown')
    
    # Calculate transaction amount where missing
    df['transaction_amount'] = df.apply(lambda row: row['item_price'] * row['quantity'] 
                                       if pd.isna(row['transaction_amount']) else row['transaction_amount'], axis=1)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if df['date'].isna().sum() > 0:
        df['date'] = df['date'].fillna(df['date'].dropna().median())
    
    # Convert numeric columns and handle missing values
    for col in ['item_price', 'quantity', 'transaction_amount']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # Standardize categorical values
    df['time_of_sale'] = df['time_of_sale'].str.title()
    df['transaction_type'] = df['transaction_type'].str.title()
    
    # Create additional features
    df['day_of_week'] = df['date'].dt.day_name()
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['is_weekend'] = df['date'].dt.dayofweek >= 5
    
    print(f"Data cleaned. Shape: {df.shape}")
    return df

# 3. Define Popularity & Revenue Analysis
def define_menu_analysis_variables(df):
    print("\n--- Calculating Menu Analysis Variables ---")

    # Create food vs drink tag
    df['category_group'] = df['item_type'].apply(lambda x: 'Drink' if str(x).strip().lower() == 'beverages' else 'Food')

    item_popularity = df.groupby(['item_name', 'item_type', 'category_group']).agg(
        order_count=('order_id', 'count'),
        total_quantity=('quantity', 'sum'),
        total_revenue=('transaction_amount', 'sum')
    ).reset_index()

    total_orders = len(df['order_id'].unique())
    total_revenue = df['transaction_amount'].sum()

    item_popularity['pct_of_orders'] = (item_popularity['order_count'] / total_orders) * 100
    item_popularity['pct_of_revenue'] = (item_popularity['total_revenue'] / total_revenue) * 100

    avg_price = df.groupby('item_name')['item_price'].mean().reset_index().rename(columns={'item_price': 'avg_price'})
    analysis_data = pd.merge(item_popularity, avg_price, on='item_name')

    # Time of day
    time_of_day_sales = df.groupby(['item_name', 'time_of_sale']).size().unstack(fill_value=0)
    if not time_of_day_sales.empty:
        time_of_day_pct = time_of_day_sales.div(time_of_day_sales.sum(axis=1), axis=0) * 100
        time_of_day_pct = time_of_day_pct.add_prefix('pct_sold_')
        analysis_data = pd.merge(analysis_data, time_of_day_pct.reset_index(), on='item_name', how='left')

    # Day of week
    if 'day_of_week' in df.columns:
        day_sales = df.groupby(['item_name', 'day_of_week']).size().unstack(fill_value=0)
        if not day_sales.empty:
            day_pct = day_sales.div(day_sales.sum(axis=1), axis=0) * 100
            day_pct = day_pct.add_prefix('pct_sold_')
            analysis_data = pd.merge(analysis_data, day_pct.reset_index(), on='item_name', how='left')

    print(f"Menu analysis data created with {len(analysis_data)} items")
    return analysis_data

# 4. Model-based Cost Estimation & Profit Calculation
def model_estimate_item_cost(df, analysis_data):
    print("\n--- Modeling Item Costs with Regression ---")
    
    # Create more comprehensive data for cost estimation
    cost_data = df[['item_name', 'item_type', 'item_price', 'time_of_sale', 'quantity', 
                    'transaction_amount']].copy().dropna()
    
    # Aggregate and create relevant features for cost modeling
    cost_data = cost_data.groupby(['item_name', 'item_type', 'time_of_sale']).agg({
        'item_price': 'mean',
        'quantity': 'sum',
        'transaction_amount': 'sum'
    }).reset_index()
    
    # Create additional features for better cost estimation
    cost_data['volume_tier'] = pd.qcut(cost_data['quantity'], 4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
    cost_data['price_tier'] = pd.qcut(
    cost_data['item_price'],
    q=4,
    labels=['Budget', 'Value', 'Premium', 'Luxury'][:len(pd.qcut(cost_data['item_price'], 4, duplicates='drop').unique())],
    duplicates='drop'
)
    
    # We'll use a data-driven approach based on pricing structure and item characteristics
    # This models the relationship between price, volume, and costs better than a fixed percentage
    
    # For high-priced items (e.g., premium cuts), food cost % is typically lower
    # For low-priced, high-volume items, food cost % is typically higher
    # We'll create an estimated cost model that reflects these industry patterns
    
    # Estimate baseline costs using a variable percentage based on price tier
    cost_tiers = {
        'Budget': 0.45,    # Higher food cost % for budget items
        'Value': 0.35,     # Medium-high food cost %
        'Premium': 0.28,   # Medium-low food cost %
        'Luxury': 0.22     # Lower food cost % for luxury items
    }
    
    # Apply the dynamic cost estimation based on pricing tier
    cost_data['estimated_base_cost'] = cost_data.apply(
    lambda row: row['item_price'] * cost_tiers.get(row['price_tier'], 0.35), axis=1  # use 0.35 as fallback
)

    
    # Add adjustment factors based on volume (economies of scale)
    volume_adjustment = {
        'Low': 1.05,       # Small batches cost slightly more per unit
        'Medium-Low': 1.02,
        'Medium-High': 0.98,
        'High': 0.95       # Large batches cost slightly less per unit
    }
    
    cost_data['historical_cost'] = cost_data.apply(
    lambda row: row['estimated_base_cost'] * volume_adjustment.get(row['volume_tier'], 1.0), axis=1
)

    
    # Prepare data for regression
    X = cost_data[['item_price', 'item_type', 'time_of_sale', 'quantity', 'price_tier', 'volume_tier']]
    y = cost_data['historical_cost']
    
    # Process categorical variables
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['item_type', 'time_of_sale', 'price_tier', 'volume_tier'])
    ], remainder='passthrough')
    
    # Test different regression models to find the best fit
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    }
    
    best_model_name = None
    best_model = None
    best_score = float('-inf')
    
    print("\nEvaluating cost estimation models:")
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        
        # Evaluate using cross-validation
        scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
        mean_score = np.mean(scores)
        print(f"{name}: R² = {mean_score:.4f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_model_name = name
            best_model = model
    
    print(f"\nSelected {best_model_name} for cost estimation (R² = {best_score:.4f})")
    
    # Build and train final model
    final_model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', best_model)])
    final_model.fit(X, y)
    
    # Prepare full dataset for prediction
    X_full = analysis_data[['item_name', 'item_type', 'avg_price']].copy()
    
    # Join with additional features needed for prediction
    item_time = df.groupby('item_name')['time_of_sale'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown').reset_index()
    item_quantity = df.groupby('item_name')['quantity'].sum().reset_index()
    
    X_full = pd.merge(X_full, item_time, on='item_name', how='left')
    X_full = pd.merge(X_full, item_quantity, on='item_name', how='left')
    X_full = X_full.rename(columns={'avg_price': 'item_price'})
    
    # Create the same price and volume tiers as in the training data
    # Use quantile cutoffs from the original cost data
    # Create safe price and volume bins
    def get_safe_bins_and_labels(series, n_quantiles=4, label_names=None):
        try:
            # Create quantile-based bins
            _, bins = pd.qcut(series, n_quantiles, retbins=True, duplicates='drop')
            bins = sorted(set(bins))  # Ensure bins are unique and strictly increasing
            
            # Dynamically assign labels
            if label_names and len(bins) - 1 == len(label_names):
                labels = label_names
            else:
                labels = [f"Bin{i+1}" for i in range(len(bins)-1)]
            
            return bins, labels
        except Exception as e:
            print(f"Could not create safe bins for {series.name}: {e}")
            return None, None

    # Get price bins and labels
    price_bins, price_labels = get_safe_bins_and_labels(
        cost_data['item_price'], 
        n_quantiles=4, 
        label_names=['Budget', 'Value', 'Premium', 'Luxury']
    )

    # Get volume bins and labels
    volume_bins, volume_labels = get_safe_bins_and_labels(
        cost_data['quantity'], 
        n_quantiles=4, 
        label_names=['Low', 'Medium-Low', 'Medium-High', 'High']
    )

    # Assign price tiers
    try:
        if price_bins and len(price_bins) - 1 == len(price_labels):
            X_full['price_tier'] = pd.cut(
                X_full['item_price'],
                bins=price_bins,
                labels=price_labels,
                include_lowest=True
            )
        else:
            raise ValueError("Invalid price bins for pd.cut()")
    except Exception as e:
        print(f"Fallback to qcut for price_tier: {e}")
        X_full['price_tier'] = pd.qcut(
            X_full['item_price'],
            q=4,
            labels=['Budget', 'Value', 'Premium', 'Luxury'],
            duplicates='drop'
        )

    # Assign volume tiers
    try:
        if volume_bins and len(volume_bins) - 1 == len(volume_labels):
            X_full['volume_tier'] = pd.cut(
                X_full['quantity'],
                bins=volume_bins,
                labels=volume_labels,
                include_lowest=True
            )
        else:
            raise ValueError("Invalid volume bins for pd.cut()")
    except Exception as e:
        print(f"Fallback to qcut for volume_tier: {e}")
        X_full['volume_tier'] = pd.qcut(
            X_full['quantity'],
            q=4,
            labels=['Low', 'Medium-Low', 'Medium-High', 'High'],
            duplicates='drop'
        )

    # Predict costs and calculate profit metrics
    analysis_data['estimated_cost'] = final_model.predict(X_full[['item_price', 'item_type', 'time_of_sale', 
                                                                 'quantity', 'price_tier', 'volume_tier']])
    
    # Make sure costs are reasonable (not negative or higher than price)
    analysis_data['estimated_cost'] = np.clip(analysis_data['estimated_cost'], 
                                            analysis_data['avg_price'] * 0.10,  # Minimum 10% of price
                                            analysis_data['avg_price'] * 0.80)  # Maximum 80% of price
    
    analysis_data['estimated_profit_per_item'] = analysis_data['avg_price'] - analysis_data['estimated_cost']
    analysis_data['profit_margin'] = (analysis_data['estimated_profit_per_item'] / analysis_data['avg_price']) * 100
    analysis_data['total_profit'] = analysis_data['total_quantity'] * analysis_data['estimated_profit_per_item']
    
    # Sort by total profit for ranking
    analysis_data = analysis_data.sort_values(['total_profit', 'total_quantity'], ascending=False)
    
    print("\nTop 5 most profitable items (regression-based):")
    print(analysis_data[['item_name', 'item_type', 'total_quantity', 'total_revenue', 'total_profit', 'profit_margin']].head())
    
    # Plot the relationship between estimated costs and prices
    plt.figure(figsize=(10, 6))
    plt.scatter(analysis_data['avg_price'], analysis_data['estimated_cost'], alpha=0.6)
    plt.xlabel('Average Price')
    plt.ylabel('Estimated Cost')
    plt.title('Relationship Between Item Price and Estimated Cost')
    
    # Add regression line
    z = np.polyfit(analysis_data['avg_price'], analysis_data['estimated_cost'], 1)
    p = np.poly1d(z)
    price_range = np.linspace(analysis_data['avg_price'].min(), analysis_data['avg_price'].max(), 100)
    plt.plot(price_range, p(price_range), "r--", alpha=0.8)
    
    # Add cost percentage reference lines
    for cost_pct in [0.2, 0.4, 0.6, 0.8]:
        plt.plot(price_range, price_range * cost_pct, ":", alpha=0.5, 
                 label=f"{int(cost_pct * 100)}% of price")
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cost_price_relationship.png", dpi=300, bbox_inches='tight')
    print("Saved 'cost_price_relationship.png'")
    
    return analysis_data

# 5. Menu Performance Matrix Visualization
def generate_menu_performance_visualization(analysis_data):
    print("\n--- Generating Enhanced Menu Performance Matrix ---")

    plt.figure(figsize=(12, 8))
    x = analysis_data['pct_of_revenue']
    y = analysis_data['profit_margin']
    quantity_norm = analysis_data['total_quantity'] / analysis_data['total_quantity'].max()
    bubble_size = quantity_norm * 1500  # scaled size for better visibility

    scatter = plt.scatter(
        x, y,
        s=bubble_size,
        c=quantity_norm,
        cmap='viridis',
        alpha=0.8,
        edgecolors='w',
        linewidth=0.5
    )

    # Median lines
    x_median = np.median(x)
    y_median = np.median(y)
    plt.axvline(x=x_median, color='gray', linestyle='--', alpha=0.7)
    plt.axhline(y=y_median, color='gray', linestyle='--', alpha=0.7)

    # Quadrant Labels
    offset = 2
    plt.text(x_median + offset, y_median + offset, 'STARS', fontsize=12, weight='bold', color='green')
    plt.text(x.min() + offset, y_median + offset, 'PUZZLES', fontsize=12, weight='bold', color='blue')
    plt.text(x_median + offset, y.min() + offset, 'WORKHORSES', fontsize=12, weight='bold', color='orange')
    plt.text(x.min() + offset, y.min() + offset, 'DOGS', fontsize=12, weight='bold', color='red')

    # Labels (optional): Top 5 items
    for _, row in analysis_data.nlargest(5, 'total_quantity').iterrows():
        plt.annotate(row['item_name'], (row['total_quantity'], row['profit_margin']),
                     textcoords="offset points", xytext=(5, 5), ha='left', fontsize=9)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Relative Sales Volume')

    plt.xlabel('Sales Volume')
    plt.ylabel('Profit Margin (%)')
    plt.title('Menu Performance Matrix (Color & Size = Popularity)')
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig("menu_performance_matrix.png", dpi=300, bbox_inches='tight')
    print("Saved 'menu_performance_matrix.png'")



# 6. Decision Tree Analysis for Menu Item Classification
def decision_tree_analysis(analysis_data):
    print("\n--- Performing Decision Tree Analysis ---")
    
    # Define features and target
    X = analysis_data[['pct_of_revenue', 'profit_margin', 'total_quantity']]
    
    # Create custom labels based on quadrants (Stars, Puzzles, Workhorses, Dogs)
    median_revenue = analysis_data['pct_of_revenue'].median()
    median_profit = analysis_data['profit_margin'].median()
    
    def assign_quadrant(row):
        if row['pct_of_revenue'] >= median_revenue and row['profit_margin'] >= median_profit:
            return 'Star'
        elif row['pct_of_revenue'] < median_revenue and row['profit_margin'] >= median_profit:
            return 'Puzzle'
        elif row['pct_of_revenue'] >= median_revenue and row['profit_margin'] < median_profit:
            return 'Workhorse'
        else:
            return 'Dog'
    
    analysis_data['quadrant'] = analysis_data.apply(assign_quadrant, axis=1)
    y = analysis_data['quadrant']
    
    # Encode categorical target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42)
    
    # Create and train decision tree
    dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_classifier.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = dt_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Decision Tree Classification Accuracy: {accuracy:.4f}")
    
    # Visualize decision tree
    plt.figure(figsize=(15, 10))
    plot_tree(dt_classifier, feature_names=X.columns, class_names=le.classes_, filled=True, rounded=True)
    plt.title("Decision Tree for Menu Item Classification")
    plt.savefig("menu_decision_tree.png", dpi=300, bbox_inches='tight')
    print("Saved 'menu_decision_tree.png'")
    
    # Add predictions to analysis data
    analysis_data['predicted_quadrant'] = le.inverse_transform(dt_classifier.predict(X))
    
    # Generate actionable insights based on quadrants
    print("\n--- Menu Item Classification Results ---")
    for quadrant in ['Star', 'Puzzle', 'Workhorse', 'Dog']:
        items = analysis_data[analysis_data['quadrant'] == quadrant]['item_name'].tolist()
        print(f"{quadrant} items ({len(items)}): {', '.join(items[:5])}{'...' if len(items) > 5 else ''}")
    
    recommendations = {
        'Star': "PROMOTE: These items should be highlighted on the menu and in marketing campaigns.",
        'Puzzle': "REPOSITION: Consider featuring these items more prominently or bundling with popular items.",
        'Workhorse': "OPTIMIZE: Look for ways to improve profit margins while maintaining popularity.",
        'Dog': "REVISE OR REMOVE: Consider reformulating, repricing, or removing these items from the menu."
    }
    
    print("\n--- Actionable Recommendations ---")
    for quadrant, recommendation in recommendations.items():
        print(f"{quadrant}: {recommendation}")
    
    return analysis_data

# 7. K-means Clustering for Menu Segmentation
def kmeans_clustering(analysis_data):
    print("\n--- Performing K-means Clustering Analysis ---")
    
    # Select features for clustering
    features = ['pct_of_revenue', 'profit_margin', 'total_quantity', 'avg_price']
    X = analysis_data[features]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal number of clusters using elbow method
    inertia = []
    max_k = min(10, len(X))
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.savefig("kmeans_elbow_curve.png", dpi=300, bbox_inches='tight')
    print("Saved 'kmeans_elbow_curve.png'")
    
    # Choose optimal number of clusters (could be automated but manual for clarity)
    optimal_k = 2  # Change if elbow plot suggests different value
    
    # Apply K-means with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to analysis data
    analysis_data['cluster'] = clusters
    
    # Summarize clusters
    cluster_summary = analysis_data.groupby('cluster').agg({
        'item_name': 'count',
        'pct_of_revenue': 'mean',
        'profit_margin': 'mean',
        'total_quantity': 'mean',
        'avg_price': 'mean',
        'total_profit': 'mean'
    }).rename(columns={'item_name': 'count'})
    
    print("\n--- Cluster Summary ---")
    print(cluster_summary)
    
    # Visualize clusters (2D projection)
    plt.figure(figsize=(12, 10))
    
    # Create a colormap for the clusters
    cmap = ListedColormap(sns.color_palette("husl", optimal_k).as_hex())
    
    plt.scatter(
        analysis_data['pct_of_revenue'], 
        analysis_data['profit_margin'],
        c=analysis_data['cluster'],
        cmap=cmap,
        s=analysis_data['total_quantity'] / analysis_data['total_quantity'].max() * 500,
        alpha=0.7,
        edgecolors='w'
    )
    
    # Add item labels
    for i, row in analysis_data.iterrows():
        plt.annotate(row['item_name'], 
                    (row['pct_of_revenue'], row['profit_margin']),
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=9)
    
    # Add centroids
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    for i in range(optimal_k):
        plt.scatter(
            centroids[i, features.index('pct_of_revenue')],
            centroids[i, features.index('profit_margin')],
            marker='X',
            s=200,
            c=f'C{i}',
            edgecolors='k',
            linewidth=2
        )
    
    plt.xlabel('Percentage of Total Revenue (%)')
    plt.ylabel('Profit Margin (%)')
    plt.title('K-means Clustering of Menu Items', fontsize=14)
    plt.colorbar(label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("kmeans_clusters.png", dpi=300, bbox_inches='tight')
    print("Saved 'kmeans_clusters.png'")
    
    # Generate cluster-based recommendations
    print("\n--- Cluster-Based Menu Optimization Recommendations ---")
    cluster_strategies = []
    
    for cluster in range(optimal_k):
        cluster_items = analysis_data[analysis_data['cluster'] == cluster]
        avg_profit = cluster_items['profit_margin'].mean()
        avg_revenue = cluster_items['pct_of_revenue'].mean()
        avg_price = cluster_items['avg_price'].mean()
        
        cluster_description = f"Cluster {cluster} ({len(cluster_items)} items)"
        cluster_details = f"Avg Profit Margin: {avg_profit:.2f}%, Avg Revenue: {avg_revenue:.2f}%, Avg Price: ${avg_price:.2f}"
        
        if avg_profit > cluster_summary['profit_margin'].median() and avg_revenue > cluster_summary['pct_of_revenue'].median():
            strategy = "STAR PERFORMERS: Feature prominently, consider price increases"
        elif avg_profit > cluster_summary['profit_margin'].median() and avg_revenue <= cluster_summary['pct_of_revenue'].median():
            strategy = "HIDDEN GEMS: Increase visibility, consider promotions to boost volume"
        elif avg_profit <= cluster_summary['profit_margin'].median() and avg_revenue > cluster_summary['pct_of_revenue'].median():
            strategy = "VOLUME DRIVERS: Optimize costs, consider slight price increases"
        else:
            strategy = "UNDERPERFORMERS: Revise recipes, adjust pricing, or consider removal"
        
        sample_items = ", ".join(cluster_items['item_name'].head(3).tolist())
        cluster_strategies.append((cluster_description, cluster_details, strategy, sample_items))
    
    for desc, details, strategy, items in cluster_strategies:
        print(f"{desc}\n{details}\nStrategy: {strategy}\nExample items: {items}\n")
    
    return analysis_data

# 8. Regression models for sales/profit prediction
def regression_analysis(df, analysis_data):
    print("\n--- Performing Regression Analysis for Sales/Profit Prediction ---")
    
    # Prepare time series data for analysis
    time_series = df.copy()
    time_series['date'] = pd.to_datetime(time_series['date'])
    time_series['year_month'] = time_series['date'].dt.strftime('%Y-%m')
    
    # Monthly sales by item
    monthly_sales = time_series.groupby(['year_month', 'item_name'])['quantity'].sum().reset_index()
    pivot_sales = monthly_sales.pivot(index='year_month', columns='item_name', values='quantity').fillna(0)
    
    # Select top items by sales volume for prediction
    top_items = analysis_data.sort_values('total_quantity', ascending=False)['item_name'].head(5).tolist()
    
    # For each top item, create a regression model to predict future sales
    plt.figure(figsize=(15, 10))
    
    for i, item in enumerate(top_items):
        if item in pivot_sales.columns:
            item_sales = pivot_sales[item].reset_index()
            item_sales['month_num'] = range(len(item_sales))
            
            X = item_sales[['month_num']]
            y = item_sales[item]
            
            # Split data
            if len(X) > 3:  # Ensure we have enough data
                X_train, X_test = X.iloc[:-2], X.iloc[-2:]
                y_train, y_test = y.iloc[:-2], y.iloc[-2:]
                
                # Try multiple regression models
                models = {
                    'Linear': LinearRegression(),
                    'Ridge': Ridge(alpha=1.0),
                    'Decision Tree': DecisionTreeRegressor(max_depth=3),
                    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=3)
                }
                
                best_model = None
                best_score = -float('inf')
                
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    score = model.score(X_test, y_test)
                    print(f"{item} - {name} R² Score: {score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                
                # Predict future months
                future_months = np.array(range(len(X), len(X) + 3)).reshape(-1, 1)
                predictions = best_model.predict(future_months)
                
                # Plot actual and predicted
                plt.subplot(len(top_items), 1, i+1)
                plt.plot(item_sales['month_num'], item_sales[item], 'o-', label='Actual')
                
                # Plot the predictions
                all_months = np.concatenate([X.values.flatten(), future_months.flatten()])
                all_values = np.concatenate([y.values, predictions])
                plt.plot(all_months, best_model.predict(all_months.reshape(-1, 1)), 'r--', label='Model')
                plt.plot(future_months, predictions, 'g*', markersize=10, label='Forecast')
                
                plt.title(f"Sales Forecast for {item}")
                plt.xlabel('Month')
                plt.ylabel('Quantity Sold')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Add prediction to analysis data
                item_idx = analysis_data.index[analysis_data['item_name'] == item].tolist()
                if item_idx:
                    forecast_growth = (predictions[0] / y.iloc[-1] - 1) * 100
                    analysis_data.loc[item_idx, 'forecast_growth_pct'] = forecast_growth
    
    plt.tight_layout()
    plt.savefig("sales_forecast.png", dpi=300, bbox_inches='tight')
    print("Saved 'sales_forecast.png'")
    
    # Profit prediction model
    print("\n--- Building Profit Prediction Model ---")
    
    # Prepare features for profit prediction
    profit_features = analysis_data[['total_quantity', 'avg_price', 'pct_of_orders', 'pct_of_revenue']]
    profit_target = analysis_data['total_profit']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(profit_features, profit_target, test_size=0.25, random_state=42)
    
    # Create and train random forest regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = rf_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"Profit Prediction Model - R² Score: {r2:.4f}, RMSE: {rmse:.2f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': profit_features.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n--- Feature Importance for Profit Prediction ---")
    print(feature_importance)
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance for Profit Prediction')
    plt.tight_layout()
    plt.savefig("profit_feature_importance.png", dpi=300, bbox_inches='tight')
    print("Saved 'profit_feature_importance.png'")
    
    return analysis_data

# 9. Association Rule Mining with Apriori Algorithm
def association_rule_mining(df):
    print("\n--- Performing Association Rule Mining for Bundling Strategies ---")
    
    # Prepare transaction data
    print("Preparing transaction data...")
    df_copy = df.copy()
    
    # Group items by order_id to get transactions
    transactions = df_copy.groupby('order_id')['item_name'].apply(list).tolist()
    
    # Convert to one-hot encoded format
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Apply Apriori algorithm
    print("Running Apriori algorithm...")
    # Start with a higher min_support and decrease if needed
    min_support = 0.01
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    
    if frequent_itemsets.empty:
        min_support = 0.005  # Try with lower support
        print(f"No frequent itemsets found, trying lower min_support={min_support}")
        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    
    if not frequent_itemsets.empty:
        # Generate association rules
        print(f"Found {len(frequent_itemsets)} frequent itemsets, generating rules...")
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        #Show how many rules were generated
        print(f"{len(rules)} rules generated before filtering.")
        # Sort by lift to keep meaningful ones
        rules = rules.sort_values('lift', ascending=False)

        if not rules.empty:
            # Sort rules by lift
            rules = rules.sort_values('lift', ascending=False)
            
            # Display top association rules
            print("\n--- Top Association Rules for Bundling ---")
            pd.set_option('display.max_colwidth', None)
            
            # Filter for rules with single antecedent and consequent for clearer bundling strategies
            simple_rules = rules[(rules['antecedents'].apply(lambda x: len(x) == 1)) & 
                                (rules['consequents'].apply(lambda x: len(x) == 1))]
            
            if not simple_rules.empty:
                # Convert frozensets to strings for better display
                simple_rules['antecedent'] = simple_rules['antecedents'].apply(lambda x: list(x)[0])
                simple_rules['consequent'] = simple_rules['consequents'].apply(lambda x: list(x)[0])
                
                # Display top bundling opportunities
                top_rules = simple_rules[['antecedent', 'consequent', 'support', 'confidence', 'lift']].head(10)
                print(top_rules)
                
                # Create bundling recommendations
                print("\n--- Bundling Strategy Recommendations ---")
                for _, rule in top_rules.iterrows():
                    antecedent = rule['antecedent']
                    consequent = rule['consequent']
                    support = rule['support']
                    confidence = rule['confidence']
                    lift = rule['lift']
                    
                    # Find profit margins for both items
                    ant_profit = analysis_data.loc[analysis_data['item_name'] == antecedent, 'profit_margin'].values
                    cons_profit = analysis_data.loc[analysis_data['item_name'] == consequent, 'profit_margin'].values
                    
                    ant_profit_val = ant_profit[0] if len(ant_profit) > 0 else None
                    cons_profit_val = cons_profit[0] if len(cons_profit) > 0 else None
                    
                    # Formulate recommendation
                    if ant_profit_val is not None and cons_profit_val is not None:
                        if ant_profit_val < cons_profit_val:
                            print(f"Bundle '{antecedent}' (lower margin: {ant_profit_val:.1f}%) with '{consequent}' "
                                  f"(higher margin: {cons_profit_val:.1f}%)")
                            print(f"   {confidence:.1%} of customers who buy {antecedent} also buy {consequent} (lift: {lift:.2f})")
                            if lift > 2:
                                print(f"   STRONG ASSOCIATION - Consider a formal bundle with slight discount")
                            else:
                                print(f"   MODERATE ASSOCIATION - Consider promoting items together")
                        else:
                            print(f"Bundle '{consequent}' (lower margin: {cons_profit_val:.1f}%) with '{antecedent}' "
                                  f"(higher margin: {ant_profit_val:.1f}%)")
                            print(f"   {confidence:.1%} of customers who buy {antecedent} also buy {consequent} (lift: {lift:.2f})")
                            if lift > 2:
                                print(f"   STRONG ASSOCIATION - Consider a formal bundle with slight discount")
                            else:
                                print(f"   MODERATE ASSOCIATION - Consider promoting items together")
                    else:
                        print(f"Consider bundling '{antecedent}' with '{consequent}'")
                        print(f"   {confidence:.1%} of customers who buy {antecedent} also buy {consequent} (lift: {lift:.2f})")
                
                # Visualize association rules
                plt.figure(figsize=(10, 8))
                plt.scatter(simple_rules['support'], simple_rules['confidence'], alpha=0.5, 
                           s=simple_rules['lift']*20)
                
                for i, row in simple_rules.head(10).iterrows():
                    plt.annotate(f"{row['antecedent']} → {row['consequent']}", 
                                (row['support'], row['confidence']),
                                xytext=(5, 5), textcoords='offset points', fontsize=8)
                
                plt.xlabel('Support')
                plt.ylabel('Confidence')
                plt.title('Association Rules - Support vs Confidence')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig("association_rules.png", dpi=300, bbox_inches='tight')
                print("Saved 'association_rules.png'")
                
                # Create network visualization of item relationships
                try:
                    import networkx as nx
                    
                    # Create graph
                    G = nx.Graph()
                    
                    # Add edges from association rules
                    for _, rule in simple_rules.head(15).iterrows():
                        G.add_edge(rule['antecedent'], rule['consequent'], 
                                  weight=rule['lift'], confidence=rule['confidence'])
                    
                    # Calculate node sizes based on item frequency
                    item_counts = df['item_name'].value_counts()
                    node_sizes = [item_counts.get(item, 1) * 5 for item in G.nodes()]
                    
                    # Plot graph
                    plt.figure(figsize=(12, 10))
                    pos = nx.spring_layout(G, seed=42)
                    
                    # Draw nodes
                    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                                         node_color='skyblue', alpha=0.7)
                    
                    # Draw edges with width based on lift
                    edge_widths = [G[u][v]['weight'] for u, v in G.edges()]
                    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4)
                    
                    # Draw labels
                    nx.draw_networkx_labels(G, pos, font_size=8)
                    
                    plt.title('Network Graph of Item Associations')
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig("item_association_network.png", dpi=300, bbox_inches='tight')
                    print("Saved 'item_association_network.png'")
                except ImportError:
                    print("NetworkX not installed, skipping network visualization")
            else:
                print("No simple rules found with sufficient confidence")
        else:
            print("No association rules found with the current parameters")
    else:
        print("No frequent itemsets found, consider lowering the minimum support threshold")
    
    return rules if 'rules' in locals() else None

def generate_final_recommendations(analysis_data, association_rules=None):
    print("\n=== FINAL MENU OPTIMIZATION RECOMMENDATIONS ===")
    
    # Classify all items into actionable categories
    median_profit = analysis_data['profit_margin'].median()
    median_revenue = analysis_data['pct_of_revenue'].median()
    
    # Create recommendation categories
    categories = {
        'promote': [],
        'reposition': [],
        'optimize': [],
        'revise': [],
        'bundle': [],
        'price_increase': [],
        'price_decrease': []
    }
    
    # Generate item-specific recommendations
    for _, item in analysis_data.iterrows():
        recommendations = []
        
        # Basic classification
        if item['profit_margin'] >= median_profit and item['pct_of_revenue'] >= median_revenue:
            categories['promote'].append(item['item_name'])
            recommendations.append("PROMOTE: Feature prominently on menu and marketing")
            
            # If very high profit margin, consider price increase
            if item['profit_margin'] > median_profit * 1.5:
                categories['price_increase'].append(item['item_name'])
                recommendations.append("PRICE INCREASE: Consider 5-10% price increase")
                
        elif item['profit_margin'] >= median_profit and item['pct_of_revenue'] < median_revenue:
            categories['reposition'].append(item['item_name'])
            recommendations.append("REPOSITION: Increase visibility on menu and marketing")
            
        elif item['profit_margin'] < median_profit and item['pct_of_revenue'] >= median_revenue:
            categories['optimize'].append(item['item_name'])
            recommendations.append("OPTIMIZE: Review recipe cost or increase price")
            
            # If very low profit margin but high revenue, definitely adjust pricing
            if item['profit_margin'] < median_profit * 0.5:
                categories['price_increase'].append(item['item_name'])
                recommendations.append("PRICE INCREASE: Consider 10-15% price increase")
                
        else:  # Low profit, low revenue
            categories['revise'].append(item['item_name'])
            recommendations.append("REVISE: Consider recipe modification or removal")
            
            # If profit margin is extremely low, consider steeper price increase or removal
            if item['profit_margin'] < 10:
                recommendations.append("WARNING: Very low profit margin, consider removing")
            
        # Add item-specific recommendations to analysis data
        analysis_data.loc[analysis_data['item_name'] == item['item_name'], 'recommendations'] = '; '.join(recommendations)
    
    # Add bundling recommendations if association rules are available
    if association_rules is not None and not association_rules.empty:
        # Filter for high-lift associations
        strong_associations = association_rules[association_rules['lift'] > 1.5]
        if not strong_associations.empty:
            for _, rule in strong_associations.head(10).iterrows():
                antecedent = list(rule['antecedents'])[0]  # Convert from frozenset
                consequent = list(rule['consequents'])[0]
                
                # Add bundling recommendation
                bundle_msg = f"Bundle with {consequent} (lift: {rule['lift']:.2f})"
                
                # Add to recommendations column if it exists
                if 'recommendations' in analysis_data.columns:
                    current_rec = analysis_data.loc[analysis_data['item_name'] == antecedent, 'recommendations'].iloc[0]
                    if pd.notna(current_rec):
                        analysis_data.loc[analysis_data['item_name'] == antecedent, 'recommendations'] = current_rec + '; ' + bundle_msg
                    else:
                        analysis_data.loc[analysis_data['item_name'] == antecedent, 'recommendations'] = bundle_msg
                
                # Add to bundle category
                if antecedent not in categories['bundle']:
                    categories['bundle'].append(antecedent + " + " + consequent)
    
    # Print summary of recommendations by category
    print("\n--- Menu Optimization Strategy Summary ---")
    for category, items in categories.items():
        if items:
            print(f"\n{category.upper()} ({len(items)} items):")
            for item in items[:5]:  # Show just the top 5 to avoid overwhelming
                print(f"  - {item}")
            if len(items) > 5:
                print(f"  - ... and {len(items) - 5} more")
    
    # Generate and save comprehensive recommendations table
    if 'recommendations' in analysis_data.columns:
        recommendations_table = analysis_data[['item_name', 'item_type', 'avg_price', 
                                             'profit_margin', 'pct_of_revenue', 'recommendations']]
        recommendations_table = recommendations_table.sort_values('profit_margin', ascending=False)
        recommendations_table.to_csv("menu_optimization_recommendations.csv", index=False)
        print("\nSaved comprehensive recommendations to 'menu_optimization_recommendations.csv'")
    
    # Create final executive summary visualization
    plt.figure(figsize=(12, 10))
    
    # Create a scatter plot with quadrants
    scatter = plt.scatter(
        analysis_data['pct_of_revenue'], 
        analysis_data['profit_margin'],
        s=analysis_data['total_quantity'] / analysis_data['total_quantity'].max() * 500,
        c=analysis_data['cluster'] if 'cluster' in analysis_data.columns else None,
        alpha=0.7,
        edgecolors='w',
        linewidth=0.5
    )
    
    # Add quadrant lines
    plt.axvline(x=median_revenue, color='gray', linestyle='--', alpha=0.7)
    plt.axhline(y=median_profit, color='gray', linestyle='--', alpha=0.7)
    

    # Add quadrant labels
    plt.text(median_revenue * 1.5, median_profit * 1.5, 'PROMOTE', 
            fontsize=14, ha='center', va='center', bbox=dict(facecolor='lightgreen', alpha=0.3))
    plt.text(median_revenue * 0.5, median_profit * 1.5, 'REPOSITION', 
            fontsize=14, ha='center', va='center', bbox=dict(facecolor='lightyellow', alpha=0.3))
    plt.text(median_revenue * 1.5, median_profit * 0.5, 'OPTIMIZE', 
            fontsize=14, ha='center', va='center', bbox=dict(facecolor='lightblue', alpha=0.3))
    plt.text(median_revenue * 0.5, median_profit * 0.5, 'REVISE', 
            fontsize=14, ha='center', va='center', bbox=dict(facecolor='lightcoral', alpha=0.3))
    
    # Add chart labels
    plt.xlabel('Percentage of Total Revenue (%)')
    plt.ylabel('Profit Margin (%)')
    plt.title('Menu Optimization Strategy Matrix', fontsize=16)
    
    # Add legend for bubble size
    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, num=4, 
                                             func=lambda s: s/500*analysis_data['total_quantity'].max())
    plt.legend(handles, labels, loc="upper left", title="Total Quantity Sold")
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("menu_optimization_strategy_2.png", dpi=300, bbox_inches='tight')
    print("Saved 'menu_optimization_strategy_2.png'")
    
    return analysis_data

def generate_final_strategy_visualization(analysis_data):
    print("\n--- Generating Enhanced Menu Optimization Strategy Visualization ---")

    median_profit = analysis_data['profit_margin'].median()
    median_revenue = analysis_data['pct_of_revenue'].median()

    plt.figure(figsize=(12, 8))
    x = analysis_data['pct_of_revenue']
    y = analysis_data['profit_margin']
    size = analysis_data['total_quantity'] / analysis_data['total_quantity'].max() * 1000

    scatter = plt.scatter(x, y, s=size, alpha=0.7, c=analysis_data['cluster'], cmap='tab10', edgecolors='w', linewidth=0.5)
    plt.axvline(x=median_revenue, color='gray', linestyle='--', alpha=0.7)
    plt.axhline(y=median_profit, color='gray', linestyle='--', alpha=0.7)

    # Quadrant zones
    plt.text(median_revenue + 1, median_profit + 1, 'PROMOTE', fontsize=12, weight='bold', color='green')
    plt.text(x.min() + 1, median_profit + 1, 'REPOSITION', fontsize=12, weight='bold', color='blue')
    plt.text(median_revenue + 1, y.min() + 1, 'OPTIMIZE', fontsize=12, weight='bold', color='orange')
    plt.text(x.min() + 1, y.min() + 1, 'REVISE', fontsize=12, weight='bold', color='red')

    # Key item labels (1 per quadrant)
    quadrants = {
        'promote': (x > median_revenue) & (y > median_profit),
        'reposition': (x <= median_revenue) & (y > median_profit),
        'optimize': (x > median_revenue) & (y <= median_profit),
        'revise': (x <= median_revenue) & (y <= median_profit)
    }

    for label, condition in quadrants.items():
        subset = analysis_data[condition].sort_values(by='total_quantity', ascending=False).head(1)
        for _, row in subset.iterrows():
            plt.annotate(row['item_name'], (row['pct_of_revenue'], row['profit_margin']),
                         textcoords="offset points", xytext=(5, 5), ha='left', fontsize=9, color='black')

    plt.xlabel('Percentage of Total Revenue (%)')
    plt.ylabel('Profit Margin (%)')
    plt.title('Menu Optimization Strategy Matrix')
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig("menu_optimization_strategy.png", dpi=300, bbox_inches='tight')
    print("Saved 'menu_optimization_strategy.png'")

# 10. Menu Optimization Final Recommendations
def generate_final_recommendations(analysis_data, association_rules=None):
    print("\n=== FINAL MENU OPTIMIZATION RECOMMENDATIONS ===")
    
    # Classify all items into actionable categories
    median_profit = analysis_data['profit_margin'].median()
    median_revenue = analysis_data['pct_of_revenue'].median()
    
    # Create recommendation categories
    categories = {
        'promote': [],
        'reposition': [],
        'optimize': [],
        'revise': [],
        'bundle': [],
        'price_increase': [],
        'price_decrease': []
    }
    
    # Generate item-specific recommendations
    for _, item in analysis_data.iterrows():
        recommendations = []
        
        # Basic classification
        if item['profit_margin'] >= median_profit and item['pct_of_revenue'] >= median_revenue:
            categories['promote'].append(item['item_name'])
            recommendations.append("PROMOTE: Feature prominently on menu and marketing")
            
            # If very high profit margin, consider price increase
            if item['profit_margin'] > median_profit * 1.5:
                categories['price_increase'].append(item['item_name'])
                recommendations.append("PRICE INCREASE: Consider 5-10% price increase")
                
        elif item['profit_margin'] >= median_profit and item['pct_of_revenue'] < median_revenue:
            categories['reposition'].append(item['item_name'])
            recommendations.append("REPOSITION: Increase visibility on menu and marketing")
            
        elif item['profit_margin'] < median_profit and item['pct_of_revenue'] >= median_revenue:
            categories['optimize'].append(item['item_name'])
            recommendations.append("OPTIMIZE: Review recipe cost or increase price")
            
            # If very low profit margin but high revenue, definitely adjust pricing
            if item['profit_margin'] < median_profit * 0.5:
                categories['price_increase'].append(item['item_name'])
                recommendations.append("PRICE INCREASE: Consider 10-15% price increase")
                
        else:  # Low profit, low revenue
            categories['revise'].append(item['item_name'])
            recommendations.append("REVISE: Consider recipe modification or removal")
            
            # If profit margin is extremely low, consider steeper price increase or removal
            if item['profit_margin'] < 10:
                recommendations.append("WARNING: Very low profit margin, consider removing")
            
        # Add item-specific recommendations to analysis data
        analysis_data.loc[analysis_data['item_name'] == item['item_name'], 'recommendations'] = '; '.join(recommendations)
    
    # Add bundling recommendations if association rules are available
    if association_rules is not None and not association_rules.empty:
        # Filter for high-lift associations
        strong_associations = association_rules[association_rules['lift'] > 1.5]
        if not strong_associations.empty:
            for _, rule in strong_associations.head(10).iterrows():
                antecedent = list(rule['antecedents'])[0]  # Convert from frozenset
                consequent = list(rule['consequents'])[0]
                
                # Add bundling recommendation
                bundle_msg = f"Bundle with {consequent} (lift: {rule['lift']:.2f})"
                
                # Add to recommendations column if it exists
                if 'recommendations' in analysis_data.columns:
                    current_rec = analysis_data.loc[analysis_data['item_name'] == antecedent, 'recommendations'].iloc[0]
                    if pd.notna(current_rec):
                        analysis_data.loc[analysis_data['item_name'] == antecedent, 'recommendations'] = current_rec + '; ' + bundle_msg
                    else:
                        analysis_data.loc[analysis_data['item_name'] == antecedent, 'recommendations'] = bundle_msg
                
                # Add to bundle category
                if antecedent not in categories['bundle']:
                    categories['bundle'].append(antecedent + " + " + consequent)
    
    # Print summary of recommendations by category
    print("\n--- Menu Optimization Strategy Summary ---")
    for category, items in categories.items():
        if items:
            print(f"\n{category.upper()} ({len(items)} items):")
            for item in items[:5]:  # Show just the top 5 to avoid overwhelming
                print(f"  - {item}")
            if len(items) > 5:
                print(f"  - ... and {len(items) - 5} more")
    
    # Generate and save comprehensive recommendations table
    if 'recommendations' in analysis_data.columns:
        recommendations_table = analysis_data[['item_name', 'item_type', 'avg_price', 
                                             'profit_margin', 'pct_of_revenue', 'recommendations']]
        recommendations_table = recommendations_table.sort_values('profit_margin', ascending=False)
        recommendations_table.to_csv("menu_optimization_recommendations.csv", index=False)
        print("\nSaved comprehensive recommendations to 'menu_optimization_recommendations.csv'")
    
    return analysis_data

# Additional visualization for value chain per item

def generate_value_chain_barchart(analysis_data):
    print("\n--- Generating Value Chain Bar Chart ---")

    try:
        top_items = analysis_data.sort_values("total_quantity", ascending=False).head(10)
        bar_width = 0.25
        x = np.arange(len(top_items))

        plt.figure(figsize=(14, 7))
        plt.bar(x - bar_width, top_items['avg_price'], width=bar_width, label='Price')
        plt.bar(x, top_items['estimated_cost'], width=bar_width, label='Cost')
        plt.bar(x + bar_width, top_items['estimated_profit_per_item'], width=bar_width, label='Profit per Item')
        #plt.bar(x + 1.5 * bar_width, top_items['total_quantity'], width=bar_width, label='Total Sales Qty')

        plt.xticks(x, top_items['item_name'], rotation=45, ha='right')
        plt.xlabel('Item Name')
        plt.ylabel('Value (Currency)')
        plt.title('Item-Level Value Chain: Price vs Cost vs Profit')
        plt.legend()
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        plt.savefig("item_value_chain.png", dpi=300, bbox_inches='tight')
        print("Saved 'item_value_chain.png'")
    except Exception as e:
        print(f"Error generating value chain chart: {e}")


# Main
def main(file_path=r"C:\\Users\\User\\OneDrive - Högskolan Dalarna\\Desktop\\Business Intelligence\\FinalProject\\Balaji Fast Food Sales.csv"):
    df = inspect_dataset(file_path)
    cleaned_df = clean_data(df)

    analysis_data = define_menu_analysis_variables(cleaned_df)
    analysis_data = model_estimate_item_cost(cleaned_df, analysis_data)

    generate_menu_performance_visualization(analysis_data)
    generate_value_chain_barchart(analysis_data)  # <<< added value chain chart here

    analysis_data = decision_tree_analysis(analysis_data)
    analysis_data = kmeans_clustering(analysis_data)
    analysis_data = regression_analysis(cleaned_df, analysis_data)
    association_rules_result = association_rule_mining(cleaned_df)
    analysis_data = generate_final_recommendations(analysis_data, association_rules_result)
    generate_final_strategy_visualization(analysis_data)

    cleaned_df.to_csv("cleaned_menu_sales.csv", index=False)
    analysis_data.to_csv("complete_menu_analysis.csv", index=False)

    print("\n=== Analysis Complete ===")
    print("The following files have been saved:")
    print("- cleaned_menu_sales.csv - The cleaned sales data")
    print("- complete_menu_analysis.csv - Complete item analysis with metrics and recommendations")
    print("- menu_performance_matrix.png - Revenue vs. Profit visualization")
    print("- menu_decision_tree.png - Decision tree classification visualization")
    print("- kmeans_clusters.png - K-means clustering visualization")
    print("- kmeans_elbow_curve.png - K-means optimal cluster determination")
    print("- sales_forecast.png - Sales forecast visualization")
    print("- profit_feature_importance.png - Feature importance for profit prediction")
    print("- association_rules.png - Association rules visualization")
    print("- item_value_chain.png - Bar chart showing price, cost, profit per item")
    print("- menu_optimization_recommendations.csv - Comprehensive recommendations table")
    print("- menu_optimization_strategy_2.png - Executive summary visualization")

    return cleaned_df, analysis_data, association_rules_result

if __name__ == "__main__":
    cleaned_df, analysis_data, association_rules = main()
    print("Script execution completed successfully.")