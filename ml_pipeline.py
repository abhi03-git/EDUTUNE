import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from joblib import dump, load
import logging

class MLPipeline:
    def __init__(self):
        self.df = None
        self.X = None
        self.y_engagement = None
        self.y_performance = None
        self.label_encoders = None
        self.X_train = None
        self.X_test = None
        self.y_eng_train = None
        self.y_eng_test = None
        self.y_per_train = None
        self.y_per_test = None
        self.models = {}
        self.results = {}
        
        # Create directories
        self.MODEL_DIR = 'models'
        self.RESULTS_DIR = 'results'
        self.STATIC_DIR = 'static'
        
        for directory in [self.MODEL_DIR, self.RESULTS_DIR, self.STATIC_DIR]:
            os.makedirs(directory, exist_ok=True)
    
    def load_dataset(self, filepath):
        """Load dataset from CSV file."""
        try:
            self.df = pd.read_csv(filepath)
            logging.info(f"Dataset loaded successfully with shape: {self.df.shape}")
            
            # Preprocess the data
            self.X, self.y_engagement, self.y_performance, self.label_encoders = self.preprocess_data(
                self.df.copy(), is_train=True
            )
            
            # Split the data
            self.train_test_split_data()
            
            return True
        except Exception as e:
            logging.error(f"Error loading dataset: {str(e)}")
            raise e
    
    def preprocess_data(self, df, is_train=True, label_encoders=None):
        """Preprocess the data exactly as in the original code."""
        # Drop unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        if is_train:
            label_encoders = {}
            
            # Encode categorical variables
            for col in df.select_dtypes(include='object').columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
        else:
            if label_encoders is None:
                raise ValueError("label_encoders must be provided for test/inference.")
            
            for col in df.select_dtypes(include='object').columns:
                if col in label_encoders:
                    le = label_encoders[col]
                    df[col] = le.transform(df[col].astype(str))
                else:
                    raise ValueError(f"Missing encoder for column: {col}")
        
        # Fill missing values
        df = df.fillna(df.mean(numeric_only=True))
        
        if is_train:
            # Extract features and targets
            if 'Engagement_Score' in df.columns and 'Performance_Score' in df.columns:
                X = df.drop(columns=['Engagement_Score', 'Performance_Score'])
                y = df['Engagement_Score']
                y1 = df['Performance_Score']
                return X, y, y1, label_encoders
            else:
                # For test data without target columns
                return df, None, None, label_encoders
        else:
            return df
    
    def train_test_split_data(self, test_size=0.2, random_state=42):
        """Split data into train and test sets."""
        self.X_train, self.X_test, self.y_eng_train, self.y_eng_test, self.y_per_train, self.y_per_test = train_test_split(
            self.X, self.y_engagement, self.y_performance, 
            test_size=test_size, random_state=random_state
        )
    
    def generate_eda_plots(self):
        """Generate EDA plots exactly as in the original code."""
        if self.df is None:
            raise ValueError("No dataset loaded")
        
        plt.style.use('default')
        plt.figure(figsize=(18, 12))
        
        # 1. Count Plot: Class_Level distribution
        plt.subplot(2, 3, 1)
        if self.X is not None:
            sns.countplot(x=self.X['Class_Level'])
        else:
            raise ValueError("Data not loaded")
        plt.title("Distribution of Class Levels")
        plt.xlabel("Class Level")
        plt.ylabel("Count")
        
        # 2. Box Plot: Heart Rate vs Performance Score
        plt.subplot(2, 3, 2)
        if self.X is not None and self.y_performance is not None:
            sns.boxplot(x=pd.cut(self.X['Heart_Rate (BPM)'], bins=5), y=self.y_performance)
        else:
            raise ValueError("Data not loaded")
        plt.title("Performance Score vs Heart Rate")
        plt.xlabel("Heart Rate Range")
        plt.ylabel("Performance Score")
        plt.xticks(rotation=45)
        
        # 3. Violin Plot: Engagement Level vs Engagement Score
        plt.subplot(2, 3, 3)
        if self.X is not None and self.y_engagement is not None:
            sns.violinplot(x=self.X['Engagement_Level'], y=self.y_engagement)
        else:
            raise ValueError("Data not loaded")
        plt.title("Engagement Score by Engagement Level")
        plt.xlabel("Engagement Level (1–10)")
        plt.ylabel("Engagement Score")
        
        # 4. Box Plot: Pitch Accuracy vs Performance Score
        plt.subplot(2, 3, 4)
        if self.X is not None and self.y_performance is not None:
            sns.boxplot(x=pd.cut(self.X['Pitch_Accuracy'], bins=5), y=self.y_performance)
        else:
            raise ValueError("Data not loaded")
        plt.title("Performance Score vs Pitch Accuracy")
        plt.xlabel("Pitch Accuracy Range")
        plt.ylabel("Performance Score")
        plt.xticks(rotation=45)
        
        # 5. Scatter Plot: Focus Time vs Engagement Score
        plt.subplot(2, 3, 5)
        if self.X is not None and self.y_engagement is not None:
            sns.scatterplot(x=self.X['Focus_Time (min)'], y=self.y_engagement)
        else:
            raise ValueError("Data not loaded")
        plt.title("Engagement Score vs Focus Time")
        plt.xlabel("Focus Time (minutes)")
        plt.ylabel("Engagement Score")
        
        # 6. Correlation Heatmap
        plt.subplot(2, 3, 6)
        if self.X is not None and self.y_engagement is not None and self.y_performance is not None:
            corr_df = self.X.copy()
            corr_df['Engagement_Score'] = self.y_engagement
            corr_df['Performance_Score'] = self.y_performance
        else:
            raise ValueError("Data not properly loaded")
        sns.heatmap(corr_df.corr(), annot=False, cmap='coolwarm', center=0)
        plt.title("Correlation Heatmap")
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.STATIC_DIR, 'eda_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def evaluate_performance(self, y_true, y_pred, model_name):
        """Evaluate model performance."""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2_Score': r2}
    
    def train_knn_regressor(self, target='engagement'):
        """Train KNN Regressor."""
        model_name = f'knn_regressor_{target}.joblib'
        model_path = os.path.join(self.MODEL_DIR, model_name)
        
        # Select target
        if target == 'engagement':
            y_train, y_test = self.y_eng_train, self.y_eng_test
        else:
            y_train, y_test = self.y_per_train, self.y_per_test
        
        if os.path.exists(model_path):
            model = load(model_path)
            logging.info(f"Loaded existing KNN Regressor for {target}")
        else:
            model = KNeighborsRegressor(n_neighbors=5)
            model.fit(self.X_train, y_train)
            dump(model, model_path)
            logging.info(f"Trained and saved new KNN Regressor for {target}")
        
        y_pred = model.predict(self.X_test)
        performance = self.evaluate_performance(y_test, y_pred, f"KNN_{target}")
        
        return model, performance
    
    def train_svr(self, target='engagement'):
        """Train SVR."""
        model_name = f'svr_{target}.joblib'
        model_path = os.path.join(self.MODEL_DIR, model_name)
        
        # Select target
        if target == 'engagement':
            y_train, y_test = self.y_eng_train, self.y_eng_test
        else:
            y_train, y_test = self.y_per_train, self.y_per_test
        
        if os.path.exists(model_path):
            model = load(model_path)
            logging.info(f"Loaded existing SVR for {target}")
        else:
            model = SVR(kernel='rbf')
            model.fit(self.X_train, y_train)
            dump(model, model_path)
            logging.info(f"Trained and saved new SVR for {target}")
        
        y_pred = model.predict(self.X_test)
        performance = self.evaluate_performance(y_test, y_pred, f"SVR_{target}")
        
        return model, performance
    
    def train_decision_tree_regressor(self, target='engagement'):
        """Train Decision Tree Regressor."""
        model_name = f'decision_tree_regressor_{target}.joblib'
        model_path = os.path.join(self.MODEL_DIR, model_name)
        
        # Select target
        if target == 'engagement':
            y_train, y_test = self.y_eng_train, self.y_eng_test
        else:
            y_train, y_test = self.y_per_train, self.y_per_test
        
        if os.path.exists(model_path):
            model = load(model_path)
            logging.info(f"Loaded existing Decision Tree Regressor for {target}")
        else:
            model = DecisionTreeRegressor(max_depth=2, random_state=42)
            model.fit(self.X_train, y_train)
            dump(model, model_path)
            logging.info(f"Trained and saved new Decision Tree Regressor for {target}")
        
        y_pred = model.predict(self.X_test)
        performance = self.evaluate_performance(y_test, y_pred, f"DecisionTree_{target}")
        
        return model, performance
    
    def train_hybrid_regressor(self, target='engagement'):
        """Train hybrid model: Linear Regression (base) + Random Forest (meta)."""
        model_name = f'hybrid_regressor_{target}.joblib'
        model_path = os.path.join(self.MODEL_DIR, model_name)
        
        # Select target
        if target == 'engagement':
            y_train, y_test = self.y_eng_train, self.y_eng_test
        else:
            y_train, y_test = self.y_per_train, self.y_per_test
        
        if os.path.exists(model_path):
            model = load(model_path)
            logging.info(f"Loaded existing Hybrid Regressor for {target}")
        else:
            base_model = LinearRegression()
            meta_model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model = StackingRegressor(
                estimators=[('lr', base_model)],
                final_estimator=meta_model,
                passthrough=True
            )
            model.fit(self.X_train, y_train)
            dump(model, model_path)
            logging.info(f"Trained and saved new Hybrid Regressor for {target}")
        
        y_pred = model.predict(self.X_test)
        performance = self.evaluate_performance(y_test, y_pred, f"Hybrid_{target}")
        
        return model, performance
    
    def train_all_models(self):
        """Train all models for both targets."""
        results = {}
        
        for target in ['engagement', 'performance']:
            target_results = {}
            
            # Train KNN
            model, perf = self.train_knn_regressor(target)
            self.models[f'knn_{target}'] = model
            target_results['KNN'] = perf
            
            # Train SVR
            model, perf = self.train_svr(target)
            self.models[f'svr_{target}'] = model
            target_results['SVR'] = perf
            
            # Train Decision Tree
            model, perf = self.train_decision_tree_regressor(target)
            self.models[f'dt_{target}'] = model
            target_results['Decision Tree'] = perf
            
            # Train Hybrid
            model, perf = self.train_hybrid_regressor(target)
            self.models[f'hybrid_{target}'] = model
            target_results['Hybrid'] = perf
            
            results[target] = target_results
        
        self.results = results
        return results
    
    def predict_single(self, form_data):
        """Make prediction on single input."""
        # Convert form data to DataFrame
        input_df = pd.DataFrame([form_data])
        
        # Preprocess the input
        processed_input = self.preprocess_data(input_df, is_train=False, label_encoders=self.label_encoders)
        
        predictions = {}
        
        # Make predictions with all models for both targets
        for target in ['engagement', 'performance']:
            target_predictions = {}
            
            for model_type in ['knn', 'svr', 'dt', 'hybrid']:
                model_key = f'{model_type}_{target}'
                if model_key in self.models:
                    pred = self.models[model_key].predict(processed_input)[0]
                    target_predictions[model_type.upper()] = round(pred, 4)
            
            predictions[target] = target_predictions
        
        return predictions
    
    def predict_batch(self, filepath):
        """Make batch predictions on CSV file."""
        # Load test data
        test_df = pd.read_csv(filepath)
        
        # Preprocess the data
        processed_data = self.preprocess_data(test_df.copy(), is_train=False, label_encoders=self.label_encoders)
        
        # Make predictions
        results_df = test_df.copy()
        
        for target in ['engagement', 'performance']:
            for model_type in ['knn', 'svr', 'dt', 'hybrid']:
                model_key = f'{model_type}_{target}'
                if model_key in self.models:
                    predictions = self.models[model_key].predict(processed_data)
                    results_df[f'{target}_{model_type}_prediction'] = predictions
        
        # Save results
        results_path = os.path.join(self.RESULTS_DIR, 'batch_predictions.csv')
        results_df.to_csv(results_path, index=False)
        
        return results_path
    
    def get_feature_names(self):
        """Get feature names for the form."""
        if self.X is not None:
            return list(self.X.columns)
        else:
            # Default feature names based on the dataset structure
            return [
                'Age', 'Gender', 'Class_Level', 'Accuracy (%)', 'Rhythm (%)', 
                'Tempo (BPM)', 'Pitch_Accuracy', 'Volume (dB)', 'Heart_Rate (BPM)', 
                'Stress_Level', 'Engagement_Level', 'Focus_Time (min)', 
                'Behavior_Score', 'Instrument', 'Lesson_Type', 'Skill_Development'
            ]
    
    def get_dataset_info(self):
        """Get basic dataset information."""
        if self.df is not None:
            return {
                'shape': self.df.shape,
                'columns': list(self.df.columns),
                'missing_values': self.df.isnull().sum().sum(),
                'data_types': self.df.dtypes.to_dict()
            }
        return None
    
    def is_data_loaded(self):
        """Check if data is loaded."""
        return self.df is not None
