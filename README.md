# 🔥 Enhanced Firefly Algorithm: Advanced CSV Data Analysis & Visualization

**Author:** KingJr07  
**Last Updated:** 2025-06-12



## Overview

The **Enhanced Firefly Algorithm** is a powerful Python tool for **automated, robust CSV data analysis and visualization**. It combines advanced feature engineering, anomaly and pattern detection, and evolutionary feature weighting with beautiful, interactive reporting—all in a single, easy-to-use script.



## ✨ Features

- **Interactive Data Preview:** HTML dashboards showing missing values, data types, unique counts, and memory usage.
- **Advanced Visualizations:** Automated numeric/categorical distributions, box plots, and time-based pattern charts.
- **Automated Reporting:** Generates a detailed Sweetviz HTML report and a JSON file with actionable insights.
- **Pattern & Anomaly Detection:** Identifies correlations, temporal patterns, and outliers using an enhanced Firefly Algorithm.
- **Robust Preprocessing:** Handles missing data, encodes categories, extracts datetime features, and eliminates constant columns.
- **Actionable Insights:** Outputs recommendations for data quality, feature selection, and further analysis.
- **Error-Resistant:** Clear error messages, checks for constant/invalid columns, and aborts gracefully if the data is unsuitable.
- **Open Source & Extensible:** Easy to modify for your own analytics workflows.

---

## 🚀 Quick Start

### 1. **Install Requirements**

```shell
pip install pandas numpy matplotlib seaborn scikit-learn tqdm plotly sweetviz
```

### 2. **Run the Analyzer**

```shell
python firefly_data_analyzer.py
```

You'll be prompted to enter the path to your CSV file.

---

## 📁 Outputs

After processing, you'll get:

- `analysis_results/data_preview.html` — Interactive data summary dashboard
- `analysis_results/numeric_distributions.html` — Numeric feature visualizations
- `analysis_results/categorical_distributions.html` — Categorical feature visualizations
- `analysis_results/time_patterns_<column>.html` — Time-based feature visualizations (if present)
- `analysis_results/insights.json` — Machine-readable and human-readable insights & recommendations
- `analysis_results/detailed_report.html` — Full Sweetviz analysis report

---

## 🧠 How It Works

1. **Loads your CSV** and profiles its structure.
2. **Preprocesses**: Handles missing values, encodes categorical columns, extracts robust datetime features, and removes constant/all-NaN columns.
3. **Visualizes**: Generates interactive HTML dashboards.
4. **Firefly Algorithm**: Optimizes feature weights and discovers patterns via an evolutionary strategy.
5. **Finds Patterns & Anomalies**: Detects strong correlations, temporal cycles, and numeric outliers.
6. **Generates Insights**: Summarizes findings and produces recommendations for data cleaning and feature selection.
7. **Reports**: Provides you with interactive and static reports for further analysis.

---

## ⚠️ Troubleshooting

- **"All features are constant after preprocessing":**  
  Your data may not have enough variation—check for columns with only one unique value.
- **"No usable features or data after preprocessing":**  
  The script filters out features that are all-NaN, constant, or unsuitable. Check your CSV for quality and variety.
- **Other errors:**  
  The script logs detailed error messages to help you debug or improve your dataset.

---

## 📖 Example Usage

```shell
python firefly_data_analyzer.py
# Enter path: /path/to/your/data.csv
```

Sample console output:

```
Dataset Profile:
- Rows: 1,000
- Columns: 12
- Numeric Features: 8
- Categorical Features: 2
- Datetime Features: 2

Generating interactive data preview...
Preparing data for analysis...
Running Firefly Algorithm analysis...
Creating visualizations...
Generating insights...
Generating automated report...

Key Findings:
1. Most Important Features:
   - age: 0.883
   - income: 0.771
   ...
2. Detected Patterns:
   Strong Correlations:
   - income & spending_score: 0.84
   ...
3. Anomalies Detected:
   - income: 10 outliers (1.0%)
4. Top Recommendations:
   - Consider removing low-importance features to simplify the model.
```

---

## 🛠️ Customization

- The code is modular and fully commented—adapt feature engineering, algorithm configuration, or reporting as needed for your use case.
- Want to integrate with Jupyter or another UI? Import the classes and use them programmatically!

---

## 🤝 Contributing

PRs and issues are welcome! Please open an issue to discuss new features, improvements, or bugfixes.

---

## 📄 License

[MIT License](LICENSE)

---

## 👋 Connect

- **Author:** [KingJr07](https://github.com/KingJr07)
- **Project Home:** [github.com/KingJr07/firefly-data-analyzer](https://github.com/KingJr07/firefly-csv-analyzer)

---

**Transform your CSV analysis workflow—try the Enhanced Firefly Algorithm today!**
