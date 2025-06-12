"""
Enhanced Firefly Algorithm: Advanced CSV Data Analysis and Visualization
Author: KingJr07
Created: 2025-06-12 08:06:07 UTC
Last Modified: 2025-06-12 08:37:20 UTC

Features:
- Interactive data preview
- Enhanced visualizations
- Automated report generation
- Pattern discovery
- Anomaly detection
- Correlation analysis
- Time series detection
- Missing value analysis
- Distribution analysis
- Robust error handling and feature validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging
from tqdm import tqdm
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sweetviz as sv
import os
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class DataProfile:
    n_rows: int
    n_cols: int
    missing_values: Dict[str, float]
    data_types: Dict[str, str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    unique_counts: Dict[str, int]
    memory_usage: float

@dataclass
class FireflyConfig:
    n_fireflies: int = 30
    n_generations: int = 50
    alpha: float = 0.5
    beta0: float = 1.0
    gamma: float = 0.1
    dimension: Optional[int] = None
    random_state: int = 42
    min_weight: float = 0.1
    convergence_threshold: float = 1e-4
    early_stopping_rounds: int = 5

class DataPreprocessor:
    @staticmethod
    def profile_data(df: pd.DataFrame) -> DataProfile:
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # MB
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_columns = []
        for col in df.columns:
            try:
                pd.to_datetime(df[col])
                datetime_columns.append(col)
            except Exception:
                continue
        return DataProfile(
            n_rows=len(df),
            n_cols=len(df.columns),
            missing_values={
                col: (df[col].isna().sum() / len(df)) * 100 if len(df) > 0 else 0
                for col in df.columns
            },
            data_types={col: str(df[col].dtype) for col in df.columns},
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            datetime_columns=datetime_columns,
            unique_counts={col: df[col].nunique() for col in df.columns},
            memory_usage=memory_usage
        )

    @staticmethod
    def prepare_data(
        df: pd.DataFrame,
        profile: DataProfile
    ) -> Tuple[np.ndarray, List[str]]:
        df_prepared = df.copy()
        features_to_use = []
        # Numeric columns: fill and use
        for col in profile.numeric_columns:
            if profile.missing_values[col] < 30:
                df_prepared[col] = df_prepared[col].fillna(df_prepared[col].median())
                if df_prepared[col].nunique() > 1 and not df_prepared[col].isna().all():
                    features_to_use.append(col)
        # Categorical columns: one-hot if low cardinality
        for col in profile.categorical_columns:
            if profile.unique_counts[col] < 10:
                dummies = pd.get_dummies(df_prepared[col], prefix=col)
                df_prepared = pd.concat([df_prepared, dummies], axis=1)
                for dcol in dummies.columns:
                    if df_prepared[dcol].nunique() > 1 and not df_prepared[dcol].isna().all():
                        features_to_use.append(dcol)
        # Datetime columns: extract robust features
        for col in profile.datetime_columns:
            dates = pd.to_datetime(df_prepared[col], errors='coerce')
            df_prepared[f'{col}_year'] = dates.dt.year
            df_prepared[f'{col}_month'] = dates.dt.month
            df_prepared[f'{col}_day'] = dates.dt.day
            df_prepared[f'{col}_dayofweek'] = dates.dt.dayofweek
            for ncol in [f'{col}_year', f'{col}_month', f'{col}_day', f'{col}_dayofweek']:
                if df_prepared[ncol].nunique() > 1 and not df_prepared[ncol].isna().all():
                    features_to_use.append(ncol)
        # Remove all-NaN and constant columns
        features_to_use_final = [f for f in features_to_use if f in df_prepared.columns]
        df_prepared = df_prepared[features_to_use_final].dropna(axis=1, how='all')
        features_to_use_final = [c for c in features_to_use_final if c in df_prepared.columns]
        df_prepared = df_prepared[features_to_use_final].dropna(axis=0, how='any')
        if not features_to_use_final or df_prepared.shape[0] == 0:
            raise ValueError("No usable features or data after preprocessing. Please check your dataset.")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_prepared[features_to_use_final])
        # Remove constant columns
        variances = scaled_data.var(axis=0)
        non_constant_cols = [i for i, v in enumerate(variances) if v > 0]
        if not non_constant_cols:
            raise ValueError("All features are constant after preprocessing. Cannot proceed.")
        scaled_data = scaled_data[:, non_constant_cols]
        features_to_use_final = [features_to_use_final[i] for i in non_constant_cols]
        return scaled_data, features_to_use_final

class DataVisualizer:
    def __init__(self, output_dir: str = 'analysis_results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def create_data_preview(
        self,
        df: pd.DataFrame,
        profile: DataProfile
    ) -> None:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Missing Values',
                'Data Types Distribution',
                'Unique Values Distribution',
                'Memory Usage by Column'
            ),
            specs=[[{"type": "xy"}, {"type": "domain"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        missing_vals = pd.Series(profile.missing_values)
        fig.add_trace(
            go.Bar(
                x=missing_vals.index,
                y=missing_vals.values,
                name='Missing %'
            ),
            row=1, col=1
        )
        dtype_counts = pd.Series(list(profile.data_types.values())).value_counts()
        fig.add_trace(
            go.Pie(
                labels=dtype_counts.index,
                values=dtype_counts.values,
                name='Data Types'
            ),
            row=1, col=2
        )
        unique_vals = pd.Series(profile.unique_counts)
        fig.add_trace(
            go.Bar(
                x=unique_vals.index,
                y=unique_vals.values,
                name='Unique Values'
            ),
            row=2, col=1
        )
        memory_usage = df.memory_usage(deep=True) / 1024**2
        fig.add_trace(
            go.Bar(
                x=memory_usage.index.astype(str),
                y=memory_usage.values,
                name='Memory (MB)'
            ),
            row=2, col=2
        )
        fig.update_layout(height=800, title_text="Dataset Overview")
        fig.write_html(f"{self.output_dir}/data_preview.html")

    def plot_distributions(
        self,
        df: pd.DataFrame,
        profile: DataProfile
    ) -> None:
        if profile.numeric_columns:
            fig = make_subplots(
                rows=len(profile.numeric_columns),
                cols=2,
                subplot_titles=[
                    f"{col} Distribution" for col in profile.numeric_columns
                ] + [
                    f"{col} Box Plot" for col in profile.numeric_columns
                ]
            )
            for i, col in enumerate(profile.numeric_columns, 1):
                fig.add_trace(
                    go.Histogram(
                        x=df[col],
                        name=col,
                        nbinsx=30
                    ),
                    row=i, col=1
                )
                fig.add_trace(
                    go.Box(
                        y=df[col],
                        name=col
                    ),
                    row=i, col=2
                )
            fig.update_layout(
                height=300 * len(profile.numeric_columns),
                title_text="Numeric Features Distribution Analysis"
            )
            fig.write_html(f"{self.output_dir}/numeric_distributions.html")
        if profile.categorical_columns:
            fig = make_subplots(
                rows=len(profile.categorical_columns),
                cols=1,
                subplot_titles=[
                    f"{col} Value Counts" for col in profile.categorical_columns
                ]
            )
            for i, col in enumerate(profile.categorical_columns, 1):
                value_counts = df[col].value_counts()
                fig.add_trace(
                    go.Bar(
                        x=value_counts.index.astype(str),
                        y=value_counts.values,
                        name=col
                    ),
                    row=i, col=1
                )
            fig.update_layout(
                height=300 * len(profile.categorical_columns),
                title_text="Categorical Features Distribution"
            )
            fig.write_html(f"{self.output_dir}/categorical_distributions.html")

    def plot_time_patterns(
        self,
        df: pd.DataFrame,
        profile: DataProfile
    ) -> None:
        if not profile.datetime_columns:
            return
        for datetime_col in profile.datetime_columns:
            dates = pd.to_datetime(df[datetime_col], errors='coerce')
            if dates.isnull().all():
                continue
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Daily Pattern',
                    'Monthly Pattern',
                    'Yearly Pattern',
                    'Day of Week Pattern'
                ]
            )
            if hasattr(dates.dt, "hour"):
                daily_counts = dates.dt.hour.value_counts().sort_index()
                fig.add_trace(
                    go.Scatter(
                        x=daily_counts.index,
                        y=daily_counts.values,
                        mode='lines+markers',
                        name='Daily'
                    ),
                    row=1, col=1
                )
            monthly_counts = dates.dt.month.value_counts().sort_index()
            fig.add_trace(
                go.Bar(
                    x=monthly_counts.index,
                    y=monthly_counts.values,
                    name='Monthly'
                ),
                row=1, col=2
            )
            yearly_counts = dates.dt.year.value_counts().sort_index()
            fig.add_trace(
                go.Scatter(
                    x=yearly_counts.index,
                    y=yearly_counts.values,
                    mode='lines+markers',
                    name='Yearly'
                ),
                row=2, col=1
            )
            dow_counts = dates.dt.dayofweek.value_counts().sort_index()
            fig.add_trace(
                go.Bar(
                    x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                    y=[dow_counts.get(i, 0) for i in range(7)],
                    name='Day of Week'
                ),
                row=2, col=2
            )
            fig.update_layout(
                height=800,
                title_text=f"Time Patterns Analysis for {datetime_col}"
            )
            fig.write_html(
                f"{self.output_dir}/time_patterns_{datetime_col}.html"
            )

class FireflyAnalyzer:
    def __init__(self, config: FireflyConfig):
        self.config = config
        np.random.seed(config.random_state)
        self.best_solutions = []
        self.feature_importance = {}
        self.convergence_history = []

    def analyze(
        self,
        data: np.ndarray,
        feature_names: List[str]
    ) -> Dict:
        self.config.dimension = data.shape[1]
        fireflies = self._initialize_fireflies()
        best_solution = None
        best_intensity = float('-inf')
        no_improvement = 0
        pbar = tqdm(range(self.config.n_generations))
        for generation in pbar:
            intensities = []
            for f in fireflies:
                try:
                    intensity = self._calculate_intensity(f, data)
                    if np.isnan(intensity) or np.isinf(intensity):
                        intensity = -np.inf
                except Exception:
                    intensity = -np.inf
                intensities.append(intensity)
            intensities = np.array(intensities)
            max_idx = np.argmax(intensities)
            if intensities[max_idx] > best_intensity:
                best_intensity = intensities[max_idx]
                best_solution = fireflies[max_idx].copy()
                no_improvement = 0
            else:
                no_improvement += 1
            self.convergence_history.append(best_intensity)
            if (no_improvement >= self.config.early_stopping_rounds and
                generation > self.config.n_generations // 2):
                logging.info(
                    f"Early stopping at generation {generation + 1} "
                    f"due to no improvement"
                )
                break
            fireflies = self._update_fireflies(
                fireflies,
                intensities,
                generation
            )
            pbar.set_description(
                f"Generation {generation + 1}/{self.config.n_generations} "
                f"Best: {best_intensity:.4f}"
            )
        if best_solution is None or not np.isfinite(best_intensity):
            raise RuntimeError("Firefly algorithm did not converge to a valid solution. "
                               "All fireflies returned NaN/inf intensity. "
                               "Please check your data for constant or invalid columns.")
        self.feature_importance = {
            name: float(weight)
            for name, weight in zip(feature_names, best_solution)
        }
        return {
            'best_solution': best_solution,
            'best_intensity': best_intensity,
            'convergence': self.convergence_history,
            'feature_importance': self.feature_importance
        }

    def _initialize_fireflies(self) -> np.ndarray:
        return np.random.uniform(
            self.config.min_weight,
            1.0,
            size=(self.config.n_fireflies, self.config.dimension)
        )

    def _calculate_intensity(
        self,
        firefly: np.ndarray,
        data: np.ndarray
    ) -> float:
        weighted_data = data * firefly
        if weighted_data.shape[0] < 2:
            return -np.inf
        try:
            corr_matrix = np.corrcoef(weighted_data.T)
            if np.isnan(corr_matrix).any() or np.isinf(corr_matrix).any():
                return -np.inf
            intensity = np.abs(corr_matrix).mean()
        except Exception:
            return -np.inf
        sparsity_penalty = (
            np.sum(firefly > 0.5) /
            len(firefly)
        )
        smoothness_penalty = np.abs(np.diff(firefly)).mean()
        return (
            intensity -
            0.2 * sparsity_penalty -
            0.1 * smoothness_penalty
        )

    def _update_fireflies(
        self,
        fireflies: np.ndarray,
        intensities: np.ndarray,
        generation: int
    ) -> np.ndarray:
        new_fireflies = fireflies.copy()
        alpha = self.config.alpha * (0.97 ** generation)
        for i in range(len(fireflies)):
            for j in range(len(fireflies)):
                if intensities[j] > intensities[i]:
                    r = np.linalg.norm(fireflies[i] - fireflies[j])
                    beta = self.config.beta0 * np.exp(
                        -self.config.gamma * r**2
                    )
                    random_step = alpha * (
                        np.random.rand(self.config.dimension) - 0.5
                    )
                    new_fireflies[i] += (
                        beta * (fireflies[j] - fireflies[i]) +
                        random_step
                    )
        return np.clip(new_fireflies, self.config.min_weight, 1.0)

class InsightGenerator:
    def __init__(self, output_dir: str = 'analysis_results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_insights(
        self,
        df: pd.DataFrame,
        profile: DataProfile,
        analysis_results: Dict
    ) -> Dict:
        insights = {
            'summary': {},
            'features': [],
            'patterns': [],
            'anomalies': [],
            'recommendations': []
        }
        insights['summary'] = self._generate_summary(df, profile)
        insights['features'] = self._analyze_features(
            df,
            profile,
            analysis_results['feature_importance']
        )
        insights['patterns'] = self._find_patterns(
            df,
            profile,
            analysis_results['feature_importance']
        )
        insights['anomalies'] = self._detect_anomalies(
            df,
            profile,
            analysis_results['feature_importance']
        )
        insights['recommendations'] = self._generate_recommendations(
            insights,
            profile
        )
        self._save_insights(insights)
        return insights

    def _generate_summary(
        self,
        df: pd.DataFrame,
        profile: DataProfile
    ) -> Dict:
        return {
            'dataset_size': f"{profile.n_rows:,} rows × {profile.n_cols:,} columns",
            'memory_usage': f"{profile.memory_usage:.2f} MB",
            'missing_data_summary': {
                col: pct
                for col, pct in profile.missing_values.items()
                if pct > 0
            },
            'data_types_summary': {
                'numeric': len(profile.numeric_columns),
                'categorical': len(profile.categorical_columns),
                'datetime': len(profile.datetime_columns)
            }
        }

    def _analyze_features(
        self,
        df: pd.DataFrame,
        profile: DataProfile,
        feature_importance: Dict[str, float]
    ) -> List[Dict]:
        feature_insights = []
        for col in df.columns:
            insight = {
                'name': col,
                'type': profile.data_types[col],
                'importance': feature_importance.get(col, 0),
                'missing_pct': profile.missing_values[col],
                'unique_values': profile.unique_counts[col]
            }
            if col in profile.numeric_columns:
                insight.update({
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'skewness': float(df[col].skew())
                })
            feature_insights.append(insight)
        return feature_insights

    def _find_patterns(
        self,
        df: pd.DataFrame,
        profile: DataProfile,
        feature_importance: Dict[str, float]
    ) -> List[Dict]:
        patterns = []
        for col in profile.datetime_columns:
            dates = pd.to_datetime(df[col], errors='coerce')
            patterns.append({
                'type': 'temporal',
                'feature': col,
                'daily_pattern': dates.dt.hour.value_counts().to_dict()
                    if hasattr(dates.dt, "hour") else {},
                'weekly_pattern': dates.dt.dayofweek.value_counts().to_dict(),
                'monthly_pattern': dates.dt.month.value_counts().to_dict()
            })
        if profile.numeric_columns:
            corr_matrix = df[profile.numeric_columns].corr()
            strong_correlations = []
            for i in range(len(profile.numeric_columns)):
                for j in range(i+1, len(profile.numeric_columns)):
                    corr = corr_matrix.iloc[i, j]
                    if abs(corr) > 0.7:
                        strong_correlations.append({
                            'features': (
                                profile.numeric_columns[i],
                                profile.numeric_columns[j]
                            ),
                            'correlation': float(corr)
                        })
            if strong_correlations:
                patterns.append({
                    'type': 'correlation',
                    'correlations': strong_correlations
                })
        return patterns

    def _detect_anomalies(
        self,
        df: pd.DataFrame,
        profile: DataProfile,
        feature_importance: Dict[str, float]
    ) -> List[Dict]:
        anomalies = []
        for col in profile.numeric_columns:
            if feature_importance.get(col, 0) > 0.3:
                values = df[col]
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1
                outliers = values[
                    (values < q1 - 1.5 * iqr) |
                    (values > q3 + 1.5 * iqr)
                ]
                if len(outliers) > 0:
                    anomalies.append({
                        'feature': col,
                        'type': 'outlier',
                        'count': int(len(outliers)),
                        'percentage': (len(outliers) / len(values)) * 100,
                        'min_outlier': float(outliers.min()),
                        'max_outlier': float(outliers.max()),
                        'normal_range': (float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr))
                    })
        return anomalies

    def _generate_recommendations(
        self,
        insights: Dict,
        profile: DataProfile
    ) -> List[Dict]:
        recommendations = []
        high_missing = [
            col for col, pct in profile.missing_values.items()
            if pct > 20
        ]
        if high_missing:
            recommendations.append({
                'type': 'data_quality',
                'issue': 'high_missing_values',
                'features': high_missing,
                'suggestion': (
                    'Consider implementing advanced imputation methods '
                    'or collecting more data for these features.'
                )
            })
        low_importance = [
            feat['name'] for feat in insights['features']
            if feat.get('importance', 0) < 0.2
        ]
        if low_importance:
            recommendations.append({
                'type': 'feature_selection',
                'issue': 'low_importance_features',
                'features': low_importance,
                'suggestion': (
                    'Consider removing or combining these low-importance '
                    'features to simplify the model.'
                )
            })
        if insights['patterns']:
            for pattern in insights['patterns']:
                if pattern['type'] == 'correlation':
                    recommendations.append({
                        'type': 'correlation',
                        'issue': 'high_correlation',
                        'details': pattern['correlations'],
                        'suggestion': (
                            'Consider feature selection or dimensionality '
                            'reduction for highly correlated features.'
                        )
                    })
        return recommendations

    def _save_insights(self, insights: Dict) -> None:
        with open(f"{self.output_dir}/insights.json", 'w') as f:
            json.dump(insights, f, indent=2)

def main():
    print(f"Running Enhanced Firefly Algorithm Data Analysis")
    print(f"Author: KingJr07")
    print(f"Last Run: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
    config = FireflyConfig(
        n_fireflies=30,
        n_generations=50,
        alpha=0.5,
        beta0=1.0,
        gamma=0.1,
        random_state=42
    )
    try:
        preprocessor = DataPreprocessor()
        analyzer = FireflyAnalyzer(config)
        visualizer = DataVisualizer()
        insight_generator = InsightGenerator()
        file_path = input("\nEnter the path to your CSV file: ")
        print(f"\nLoading and analyzing {file_path}...")
        df = pd.read_csv(file_path)
        profile = preprocessor.profile_data(df)
        print(f"\nDataset Profile:")
        print(f"- Rows: {profile.n_rows:,}")
        print(f"- Columns: {profile.n_cols:,}")
        print(f"- Memory Usage: {profile.memory_usage:.2f} MB")
        print(f"- Numeric Features: {len(profile.numeric_columns)}")
        print(f"- Categorical Features: {len(profile.categorical_columns)}")
        print(f"- Datetime Features: {len(profile.datetime_columns)}")
        print("\nGenerating interactive data preview...")
        visualizer.create_data_preview(df, profile)
        print("\nPreparing data for analysis...")
        data, feature_names = preprocessor.prepare_data(df, profile)
        print("\nRunning Firefly Algorithm analysis...")
        results = analyzer.analyze(data, feature_names)
        print("\nCreating visualizations...")
        visualizer.plot_distributions(df, profile)
        visualizer.plot_time_patterns(df, profile)
        print("\nGenerating insights...")
        insights = insight_generator.generate_insights(
            df, profile, results
        )
        print("\nGenerating automated report...")
        report = sv.analyze(df)
        report.show_html(
            filepath="analysis_results/detailed_report.html",
            open_browser=False
        )
        print("\nKey Findings:")
        print("\n1. Most Important Features:")
        for feature, importance in sorted(
            results['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]:
            print(f"   - {feature}: {importance:.3f}")
        print("\n2. Detected Patterns:")
        for pattern in insights['patterns']:
            if pattern['type'] == 'correlation':
                print("   Strong Correlations:")
                for corr in pattern['correlations'][:3]:
                    print(
                        f"   - {corr['features'][0]} & {corr['features'][1]}: "
                        f"{corr['correlation']:.3f}"
                    )
        print("\n3. Anomalies Detected:")
        for anomaly in insights['anomalies']:
            print(
                f"   - {anomaly['feature']}: {anomaly['count']} outliers "
                f"({anomaly['percentage']:.1f}%)"
            )
        print("\n4. Top Recommendations:")
        for rec in insights['recommendations']:
            print(f"   - {rec['suggestion']}")
        print("\nAnalysis complete! Check the following files:")
        print("1. analysis_results/data_preview.html")
        print("2. analysis_results/numeric_distributions.html")
        print("3. analysis_results/categorical_distributions.html")
        print("4. analysis_results/insights.json")
        print("5. analysis_results/detailed_report.html")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        print(f"\nERROR: {str(e)}")
        raise

if __name__ == "__main__":
    main()