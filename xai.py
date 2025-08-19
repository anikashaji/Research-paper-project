# XAI Heart Disease Prediction - Fixed Implementation
# Author: XAI Research Implementation (Fixed Version)
# Purpose: Comprehensive XAI analysis for heart disease prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.inspection import permutation_importance, partial_dependence, PartialDependenceDisplay
import xgboost as xgb
import shap
import lime
import lime.lime_tabular
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

print("XAI Heart Disease Prediction - Fixed Implementation")
print("="*60)

class HeartDiseaseXAI:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_names = []
        
    def load_and_prepare_data(self, file_path="heart-disease.csv"):
        """Load and prepare heart disease dataset"""
        print("\n1.1 Loading and Preparing Dataset")
        print("-" * 40)
        
        try:
            # Load the actual heart disease dataset
            print(f"Loading dataset from: {file_path}")
            self.data = pd.read_csv(file_path)
            print("✓ Dataset loaded successfully!")
            
            # Check for target column
            target_candidates = ['target', 'num', 'condition', 'diagnosis', 'heart_disease']
            target_col = None
            
            for col in target_candidates:
                if col in self.data.columns:
                    target_col = col
                    break
            
            if target_col is None:
                target_col = self.data.columns[-1]
                print(f"⚠️  Target column not found, assuming '{target_col}' is the target")
            
            # Rename target column to 'target' for consistency
            if target_col != 'target':
                self.data = self.data.rename(columns={target_col: 'target'})
                print(f"✓ Renamed '{target_col}' to 'target'")
            
            # Dataset information
            print(f"\n📊 Dataset Information:")
            print(f"Shape: {self.data.shape}")
            print(f"Columns: {list(self.data.columns)}")
            
            # Handle missing values
            missing_values = self.data.isnull().sum()
            if missing_values.sum() > 0:
                print(f"\n⚠️  Missing values found:")
                for col, count in missing_values[missing_values > 0].items():
                    print(f"  {col}: {count} missing values")
                
                # Handle missing values
                numerical_cols = self.data.select_dtypes(include=[np.number]).columns
                for col in numerical_cols:
                    if col != 'target' and self.data[col].isnull().sum() > 0:
                        self.data[col].fillna(self.data[col].median(), inplace=True)
                
                categorical_cols = self.data.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    if self.data[col].isnull().sum() > 0:
                        self.data[col].fillna(self.data[col].mode()[0], inplace=True)
                
                print("✓ Missing values handled")
            else:
                print("✓ No missing values found")
            
            # Ensure target is binary (0, 1)
            unique_targets = sorted(self.data['target'].unique())
            print(f"\nTarget values: {unique_targets}")
            
            if len(unique_targets) > 2:
                print("🔧 Converting multi-class target to binary...")
                self.data['target'] = (self.data['target'] > 0).astype(int)
                print("✓ Target converted to binary (0: No disease, 1: Disease)")
            
            # Remove duplicates
            duplicates = self.data.duplicated().sum()
            if duplicates > 0:
                print(f"⚠️  Found {duplicates} duplicate rows")
                self.data = self.data.drop_duplicates()
                print(f"✓ Duplicates removed. New shape: {self.data.shape}")
            
            # Final target distribution
            target_counts = self.data['target'].value_counts().sort_index()
            print(f"\nFinal target distribution:")
            for idx, count in target_counts.items():
                label = "No Disease" if idx == 0 else "Disease"
                print(f"  {label} ({idx}): {count} samples ({count/len(self.data)*100:.1f}%)")
            
        except FileNotFoundError:
            print(f"❌ Error: File not found at {file_path}")
            print("Creating sample dataset as fallback...")
            
            # Create sample data
            np.random.seed(42)
            n_samples = 1000
            
            self.data = pd.DataFrame({
                'age': np.random.randint(29, 80, n_samples),
                'sex': np.random.randint(0, 2, n_samples),
                'cp': np.random.randint(0, 4, n_samples),
                'trestbps': np.random.randint(90, 200, n_samples),
                'chol': np.random.randint(150, 400, n_samples),
                'fbs': np.random.randint(0, 2, n_samples),
                'restecg': np.random.randint(0, 3, n_samples),
                'thalach': np.random.randint(80, 220, n_samples),
                'exang': np.random.randint(0, 2, n_samples),
                'oldpeak': np.random.uniform(0, 6, n_samples),
                'slope': np.random.randint(0, 3, n_samples),
                'ca': np.random.randint(0, 4, n_samples),
                'thal': np.random.randint(0, 3, n_samples),
                'target': np.random.randint(0, 2, n_samples)
            })
            print("✓ Sample dataset created")
        
        # Feature names (excluding target)
        self.feature_names = [col for col in self.data.columns if col != 'target']
        print(f"\n📋 Features for analysis: {self.feature_names}")
        
        return self.data
    
    def preprocess_data(self):
        """Preprocess the dataset"""
        print("\n1.2 Data Preprocessing")
        print("-" * 30)
        
        # Separate features and target
        X = self.data[self.feature_names]
        y = self.data['target']
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Convert back to DataFrame for easier handling
        self.X_train_scaled = pd.DataFrame(self.X_train_scaled, columns=self.feature_names, index=self.X_train.index)
        self.X_test_scaled = pd.DataFrame(self.X_test_scaled, columns=self.feature_names, index=self.X_test.index)
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print("✓ Preprocessing completed!")
        
    def train_models(self):
        """Train multiple models for XAI comparison"""
        print("\n1.3 Model Training")
        print("-" * 25)
        
        # Define models
        models_config = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            
            try:
                # Use scaled data for Neural Network and Logistic Regression
                if name in ['Neural Network', 'Logistic Regression']:
                    model.fit(self.X_train_scaled, self.y_train)
                    train_pred = model.predict(self.X_train_scaled)
                    test_pred = model.predict(self.X_test_scaled)
                    test_proba = model.predict_proba(self.X_test_scaled)[:, 1]
                    cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
                else:
                    model.fit(self.X_train, self.y_train)
                    train_pred = model.predict(self.X_train)
                    test_pred = model.predict(self.X_test)
                    test_proba = model.predict_proba(self.X_test)[:, 1]
                    cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
                
                # Calculate metrics
                train_acc = accuracy_score(self.y_train, train_pred)
                test_acc = accuracy_score(self.y_test, test_pred)
                auc_score = roc_auc_score(self.y_test, test_proba)
                
                results[name] = {
                    'model': model,
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                    'auc_score': auc_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                print(f"✓ Train Accuracy: {train_acc:.4f}")
                print(f"✓ Test Accuracy: {test_acc:.4f}")
                print(f"✓ AUC Score: {auc_score:.4f}")
                print(f"✓ CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
                
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
        
        self.models = results
        return results
    
    def plot_model_comparison(self):
        """Plot model performance comparison"""
        print("\n1.4 Model Performance Comparison")
        print("-" * 35)
        
        if not self.models:
            print("No models available for comparison")
            return
        
        # Extract metrics
        names = list(self.models.keys())
        test_acc = [self.models[name]['test_accuracy'] for name in names]
        auc_scores = [self.models[name]['auc_score'] for name in names]
        cv_means = [self.models[name]['cv_mean'] for name in names]
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Test Accuracy
        axes[0].bar(names, test_acc, alpha=0.7)
        axes[0].set_title('Test Accuracy')
        axes[0].set_ylabel('Accuracy')
        axes[0].tick_params(axis='x', rotation=45)
        
        # AUC Score
        axes[1].bar(names, auc_scores, alpha=0.7, color='orange')
        axes[1].set_title('AUC Score')
        axes[1].set_ylabel('AUC')
        axes[1].tick_params(axis='x', rotation=45)
        
        # CV Score
        axes[2].bar(names, cv_means, alpha=0.7, color='green')
        axes[2].set_title('Cross-Validation Score')
        axes[2].set_ylabel('CV Score')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

    def setup_shap_explainers(self):
        """Setup SHAP explainers for all models"""
        print("\n2.1 Setting up SHAP Explainers")
        print("-" * 35)
        
        self.shap_explainers = {}
        self.shap_values = {}
        
        for name, model_info in self.models.items():
            model = model_info['model']
            print(f"Setting up SHAP for {name}...")
            
            try:
                if name == 'Random Forest':
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(self.X_test)
                    # For binary classification, take positive class if it returns a list
                    if isinstance(shap_values, list) and len(shap_values) == 2:
                        shap_values = shap_values[1]  # Take positive class
                    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                        shap_values = shap_values[:, :, 1]  # Take positive class
                
                elif name == 'XGBoost':
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(self.X_test)
                    # XGBoost usually returns single array for binary classification
                    if isinstance(shap_values, list) and len(shap_values) == 2:
                        shap_values = shap_values[1]
                
                elif name == 'Neural Network':
                    # Use KernelExplainer for Neural Network (limit samples for speed)
                    explainer = shap.KernelExplainer(
                        model.predict_proba, 
                        shap.sample(self.X_train_scaled, 50)
                    )
                    # Limit to first 20 samples for speed
                    test_sample = self.X_test_scaled.iloc[:20]
                    shap_values = explainer.shap_values(test_sample)
                    if isinstance(shap_values, list) and len(shap_values) == 2:
                        shap_values = shap_values[1]  # Take positive class
                
                elif name == 'Logistic Regression':
                    explainer = shap.LinearExplainer(model, self.X_train_scaled)
                    shap_values = explainer.shap_values(self.X_test_scaled)
                    if isinstance(shap_values, list) and len(shap_values) == 2:
                        shap_values = shap_values[1]  # Take positive class
                
                # Ensure shap_values is 2D array
                if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 2:
                    self.shap_explainers[name] = explainer
                    self.shap_values[name] = shap_values
                    print(f"✓ {name} SHAP setup complete (shape: {shap_values.shape})")
                else:
                    print(f"⚠️ Unexpected SHAP values shape for {name}: {type(shap_values)}")
                    if hasattr(shap_values, 'shape'):
                        print(f"   Shape: {shap_values.shape}")
                
            except Exception as e:
                print(f"❌ Error setting up SHAP for {name}: {e}")
    
    def generate_shap_explanations(self, patient_idx=0):
        """Generate SHAP explanations for a specific patient"""
        print(f"\n2.2 SHAP Explanations for Patient {patient_idx}")
        print("-" * 45)
        
        if not self.shap_values:
            print("No SHAP values available. Skipping SHAP explanations.")
            return
        
        available_models = list(self.shap_values.keys())
        n_models = len(available_models)
        
        if n_models == 0:
            print("No SHAP explanations available.")
            return
        
        # Create subplot layout
        n_cols = min(2, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_models > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, name in enumerate(available_models):
            try:
                shap_vals = self.shap_values[name]
                
                # Handle Neural Network with fewer samples
                if name == 'Neural Network' and patient_idx >= len(shap_vals):
                    patient_to_show = min(patient_idx, len(shap_vals) - 1)
                else:
                    patient_to_show = patient_idx
                
                # Get SHAP values for the patient
                patient_shap = shap_vals[patient_to_show]
                
                # Sort by absolute importance
                indices = np.argsort(np.abs(patient_shap))[::-1][:8]  # Top 8 features
                
                # Create bar plot
                colors = ['red' if val < 0 else 'blue' for val in patient_shap[indices]]
                bars = axes[i].barh(range(len(indices)), 
                                   patient_shap[indices],
                                   color=colors, alpha=0.7)
                
                axes[i].set_yticks(range(len(indices)))
                axes[i].set_yticklabels([self.feature_names[idx] for idx in indices])
                axes[i].set_title(f'{name} - SHAP Values\nPatient {patient_to_show}')
                axes[i].set_xlabel('SHAP Value (impact on prediction)')
                axes[i].axvline(x=0, color='black', linestyle='-', alpha=0.3)
                
                # Add value labels on bars
                for j, bar in enumerate(bars):
                    width = bar.get_width()
                    axes[i].text(width + (0.01 if width >= 0 else -0.01), 
                               bar.get_y() + bar.get_height()/2,
                               f'{width:.3f}', 
                               ha='left' if width >= 0 else 'right', 
                               va='center', fontsize=8)
                
            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error: {str(e)[:50]}...', 
                           transform=axes[i].transAxes, ha='center', va='center')
                axes[i].set_title(f'{name} - Error')
        
        # Hide unused subplots
        for i in range(len(available_models), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def setup_lime_explainer(self):
        """Setup LIME explainer"""
        print("\n2.3 Setting up LIME Explainer")
        print("-" * 35)
        
        if 'Random Forest' not in self.models:
            print("Random Forest model not available for LIME")
            return False
        
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_train.values,
            feature_names=self.feature_names,
            class_names=['No Disease', 'Disease'],
            mode='classification'
        )
        
        print("✓ LIME explainer setup complete")
        return True
    
    def generate_lime_explanations(self, patient_idx=0):
        """Generate LIME explanations for a specific patient"""
        print(f"\n2.4 LIME Explanations for Patient {patient_idx}")
        print("-" * 45)
        
        if not hasattr(self, 'lime_explainer'):
            print("LIME explainer not available")
            return
        
        rf_model = self.models['Random Forest']['model']
        
        # Generate explanation
        explanation = self.lime_explainer.explain_instance(
            self.X_test.iloc[patient_idx].values,
            rf_model.predict_proba,
            num_features=len(self.feature_names)
        )
        
        # Plot explanation
        fig = explanation.as_pyplot_figure()
        fig.suptitle(f'LIME Explanation - Patient {patient_idx}', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Print explanation details
        print("Feature contributions:")
        for feature, importance in explanation.as_list():
            print(f"  {feature}: {importance:.4f}")
    
    def calculate_permutation_importance(self):
        """Calculate permutation importance for all models"""
        print("\n3.1 Permutation Importance Analysis")
        print("-" * 40)
        
        self.perm_importance = {}
        
        n_models = len(self.models)
        if n_models == 0:
            print("No models available for permutation importance")
            return
        
        n_cols = min(2, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1 and n_models > 1:
            pass  # axes is already correct
        else:
            axes = axes.flatten()
        
        for i, (name, model_info) in enumerate(self.models.items()):
            model = model_info['model']
            print(f"Calculating permutation importance for {name}...")
            
            try:
                # Use appropriate dataset based on model
                if name in ['Neural Network', 'Logistic Regression']:
                    X_data = self.X_test_scaled
                else:
                    X_data = self.X_test
                
                # Calculate permutation importance
                perm_imp = permutation_importance(
                    model, X_data, self.y_test, 
                    n_repeats=10, random_state=42, scoring='accuracy'
                )
                
                self.perm_importance[name] = perm_imp
                
                # Plot results
                indices = np.argsort(perm_imp.importances_mean)[::-1]
                
                ax = axes[i] if n_models > 1 else axes[0]
                ax.bar(range(len(self.feature_names)), 
                       perm_imp.importances_mean[indices],
                       yerr=perm_imp.importances_std[indices],
                       alpha=0.7)
                ax.set_title(f'{name} - Permutation Importance')
                ax.set_xlabel('Features')
                ax.set_ylabel('Importance')
                ax.set_xticks(range(len(self.feature_names)))
                ax.set_xticklabels([self.feature_names[idx] for idx in indices], rotation=45)
                
                print(f"✓ {name} permutation importance calculated")
                
            except Exception as e:
                print(f"❌ Error calculating permutation importance for {name}: {e}")
        
        # Hide unused subplots
        if n_models > 1:
            for i in range(n_models, len(axes)):
                axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def generate_partial_dependence_plots(self):
        """Generate Partial Dependence Plots"""
        print("\n3.2 Partial Dependence Analysis")
        print("-" * 35)
        
        if 'Random Forest' not in self.models or 'Random Forest' not in self.perm_importance:
            print("Random Forest model or permutation importance not available for PDP")
            return
        
        try:
            rf_model = self.models['Random Forest']['model']
            perm_imp = self.perm_importance['Random Forest']
            
            # Select top 4 most important features
            top_features_idx = np.argsort(perm_imp.importances_mean)[::-1][:4]
            top_features = [self.feature_names[i] for i in top_features_idx]
            
            print(f"Generating PDPs for top features: {top_features}")
            
            # Create PDP plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            for i, feature_idx in enumerate(top_features_idx):
                try:
                    # Manual PDP calculation for better control
                    feature_values = np.linspace(
                        self.X_test.iloc[:, feature_idx].min(),
                        self.X_test.iloc[:, feature_idx].max(),
                        30
                    )
                    
                    partial_deps = []
                    for val in feature_values:
                        X_temp = self.X_test.copy()
                        X_temp.iloc[:, feature_idx] = val
                        pred = rf_model.predict_proba(X_temp)[:, 1].mean()
                        partial_deps.append(pred)
                    
                    axes[i].plot(feature_values, partial_deps, 'b-', linewidth=2)
                    axes[i].set_title(f'PDP: {self.feature_names[feature_idx]}')
                    axes[i].set_xlabel(self.feature_names[feature_idx])
                    axes[i].set_ylabel('Partial Dependence')
                    axes[i].grid(True, alpha=0.3)
                    
                except Exception as e:
                    print(f"Error with feature {self.feature_names[feature_idx]}: {e}")
                    axes[i].text(0.5, 0.5, f'Error: {str(e)[:30]}...', 
                               transform=axes[i].transAxes, ha='center', va='center')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error in PDP generation: {e}")
    
    def feature_interaction_analysis(self):
        """Analyze feature interactions using SHAP"""
        print("\n3.3 Feature Interaction Analysis")
        print("-" * 40)
        
        if not self.shap_values:
            print("No SHAP values available for feature interaction analysis.")
            return
        
        # Find the best available model for analysis
        available_model = None
        for model_name in ['Random Forest', 'XGBoost', 'Logistic Regression']:
            if model_name in self.shap_values:
                available_model = model_name
                break
        
        if available_model is None:
            available_model = list(self.shap_values.keys())[0]
        
        try:
            shap_values = self.shap_values[available_model]
            
            # Determine which dataset to use
            if available_model in ['Neural Network', 'Logistic Regression']:
                data_for_plot = self.X_test_scaled
                # Handle case where Neural Network has fewer samples
                if len(shap_values) < len(data_for_plot):
                    data_for_plot = data_for_plot.iloc[:len(shap_values)]
            else:
                data_for_plot = self.X_test
            
            print(f"Using {available_model} for feature interaction analysis")
            print(f"SHAP values shape: {shap_values.shape}")
            print(f"Data shape: {data_for_plot.shape}")
            
            # Manual summary plot
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            sorted_idx = np.argsort(mean_abs_shap)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(self.feature_names)), mean_abs_shap[sorted_idx])
            plt.yticks(range(len(self.feature_names)), 
                      [self.feature_names[i] for i in sorted_idx])
            plt.xlabel('Mean |SHAP Value|')
            plt.title(f'Feature Importance - {available_model}')
            plt.tight_layout()
            plt.show()
            
            # Feature importance comparison across models
            if self.perm_importance:
                plt.figure(figsize=(15, 8))
                
                importance_data = {}
                for name in self.models.keys():
                    if name in self.perm_importance:
                        importance_data[f'{name}_Perm'] = self.perm_importance[name].importances_mean
                    
                    if name in self.shap_values:
                        shap_imp = np.mean(np.abs(self.shap_values[name]), axis=0)
                        importance_data[f'{name}_SHAP'] = shap_imp
                
                if importance_data:
                    # Create DataFrame for easier plotting
                    importance_df = pd.DataFrame(importance_data, index=self.feature_names)
                    
                    # Normalize for comparison
                    for col in importance_df.columns:
                        importance_df[col] = importance_df[col] / importance_df[col].max()
                    
                    importance_df.plot(kind='bar', alpha=0.7, figsize=(15, 8))
                    plt.title('Feature Importance Comparison Across Models\n(Normalized Scores)')
                    plt.xlabel('Features')
                    plt.ylabel('Normalized Importance Score')
                    plt.legend(title='Methods', bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.show()
                else:
                    print("No importance data available for comparison.")
                    
        except Exception as e:
            print(f"Error in feature interaction analysis: {e}")

    def train_interpretable_models(self):
        """Train inherently interpretable models"""
        print("\n4.1 Training Interpretable Models")
        print("-" * 40)
        
        self.interpretable_models = {}
        
        # Decision Tree
        print("Training Decision Tree...")
        try:
            dt_model = DecisionTreeClassifier(
                max_depth=5, min_samples_split=20, random_state=42
            )
            dt_model.fit(self.X_train, self.y_train)
            dt_pred = dt_model.predict(self.X_test)
            dt_acc = accuracy_score(self.y_test, dt_pred)
            
            self.interpretable_models['Decision Tree'] = {
                'model': dt_model,
                'accuracy': dt_acc,
                'predictions': dt_pred
            }
            
            print(f"✓ Decision Tree Accuracy: {dt_acc:.4f}")
        except Exception as e:
            print(f"❌ Error training Decision Tree: {e}")
        
        # Simple Logistic Regression
        print("Training Simple Logistic Regression...")
        try:
            simple_lr = LogisticRegression(random_state=42, penalty='l1', solver='liblinear', C=0.1, max_iter=1000)
            simple_lr.fit(self.X_train_scaled, self.y_train)
            lr_pred = simple_lr.predict(self.X_test_scaled)
            lr_acc = accuracy_score(self.y_test, lr_pred)
            
            self.interpretable_models['Simple Logistic Regression'] = {
                'model': simple_lr,
                'accuracy': lr_acc,
                'predictions': lr_pred
            }
            
            print(f"✓ Simple Logistic Regression Accuracy: {lr_acc:.4f}")
        except Exception as e:
            print(f"❌ Error training Simple Logistic Regression: {e}")
    
    def visualize_decision_tree(self):
        """Visualize decision tree structure"""
        print("\n4.2 Decision Tree Visualization")
        print("-" * 35)
        
        if 'Decision Tree' not in self.interpretable_models:
            print("Decision Tree model not available")
            return
        
        dt_model = self.interpretable_models['Decision Tree']['model']
        
        plt.figure(figsize=(20, 12))
        plot_tree(dt_model, 
                 feature_names=self.feature_names,
                 class_names=['No Disease', 'Disease'],
                 filled=True, rounded=True, fontsize=10)
        plt.title('Decision Tree for Heart Disease Prediction')
        plt.show()
    
    def extract_decision_rules(self):
        """Extract interpretable rules from decision tree"""
        print("\n4.3 Decision Rules Extraction")
        print("-" * 35)
        
        if 'Decision Tree' not in self.interpretable_models:
            print("Decision Tree model not available")
            return
        
        dt_model = self.interpretable_models['Decision Tree']['model']
        
        # Get tree structure
        tree = dt_model.tree_
        feature_names = self.feature_names
        
        def get_rules(tree, feature_names):
            """Extract rules from decision tree"""
            rules = []
            
            def recurse(node, rule):
                if tree.feature[node] != -2:  # Not a leaf
                    feature = feature_names[tree.feature[node]]
                    threshold = tree.threshold[node]
                    
                    # Left child (<=)
                    left_rule = rule + f" AND {feature} <= {threshold:.2f}"
                    recurse(tree.children_left[node], left_rule)
                    
                    # Right child (>)
                    right_rule = rule + f" AND {feature} > {threshold:.2f}"
                    recurse(tree.children_right[node], right_rule)
                else:
                    # Leaf node
                    class_counts = tree.value[node][0]
                    predicted_class = np.argmax(class_counts)
                    confidence = class_counts[predicted_class] / np.sum(class_counts)
                    
                    rules.append({
                        'rule': rule[5:] if rule.startswith(' AND ') else rule,  # Remove leading ' AND '
                        'predicted_class': predicted_class,
                        'confidence': confidence,
                        'samples': int(np.sum(class_counts))
                    })
            
            recurse(0, "")
            return rules
        
        rules = get_rules(tree, feature_names)
        
        print("Extracted Decision Rules:")
        print("=" * 80)
        
        for i, rule in enumerate(rules[:10]):  # Show top 10 rules
            class_name = "Disease" if rule['predicted_class'] == 1 else "No Disease"
            print(f"\nRule {i+1}: IF {rule['rule']}")
            print(f"  THEN Prediction: {class_name}")
            print(f"  Confidence: {rule['confidence']:.3f}")
            print(f"  Samples: {rule['samples']}")
    
    def analyze_logistic_regression_coefficients(self):
        """Analyze logistic regression coefficients"""
        print("\n4.4 Logistic Regression Coefficient Analysis")
        print("-" * 50)
        
        if 'Simple Logistic Regression' not in self.interpretable_models:
            print("Simple Logistic Regression model not available")
            return
        
        lr_model = self.interpretable_models['Simple Logistic Regression']['model']
        
        # Get coefficients
        coefficients = lr_model.coef_[0]
        intercept = lr_model.intercept_[0]
        
        # Create coefficient analysis
        coef_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': coefficients,
            'Abs_Coefficient': np.abs(coefficients)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        print("Logistic Regression Coefficients (sorted by absolute value):")
        print("-" * 60)
        print(f"Intercept: {intercept:.4f}")
        print()
        
        for _, row in coef_df.iterrows():
            direction = "increases" if row['Coefficient'] > 0 else "decreases"
            print(f"{row['Feature']:15s}: {row['Coefficient']:8.4f} ({direction} disease probability)")
        
        # Plot coefficients
        plt.figure(figsize=(12, 8))
        colors = ['red' if x < 0 else 'blue' for x in coef_df['Coefficient']]
        plt.barh(range(len(coef_df)), coef_df['Coefficient'], color=colors, alpha=0.7)
        plt.yticks(range(len(coef_df)), coef_df['Feature'])
        plt.xlabel('Coefficient Value')
        plt.title('Logistic Regression Coefficients\n(Red: Decreases risk, Blue: Increases risk)')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_clinical_report(xai_system, patient_idx=0):
    """Generate a clinical interpretation report for a specific patient"""
    print(f"\n📋 CLINICAL REPORT - PATIENT {patient_idx}")
    print("=" * 60)
    
    try:
        # Get patient data
        patient_data = xai_system.X_test.iloc[patient_idx]
        actual_label = xai_system.y_test.iloc[patient_idx]
        
        print("PATIENT INFORMATION:")
        print("-" * 20)
        for feature in xai_system.feature_names[:8]:  # Show first 8 features
            if feature in patient_data.index:
                print(f"{feature}: {patient_data[feature]}")
        
        print(f"Actual Diagnosis: {'Heart Disease' if actual_label == 1 else 'No Heart Disease'}")
        
        # Model predictions
        print("\nMODEL PREDICTIONS:")
        print("-" * 20)
        
        for name, model_info in xai_system.models.items():
            try:
                model = model_info['model']
                
                if name in ['Neural Network', 'Logistic Regression']:
                    prob = model.predict_proba(xai_system.X_test_scaled.iloc[patient_idx:patient_idx+1])[0, 1]
                    pred = model.predict(xai_system.X_test_scaled.iloc[patient_idx:patient_idx+1])[0]
                else:
                    prob = model.predict_proba(xai_system.X_test.iloc[patient_idx:patient_idx+1])[0, 1]
                    pred = model.predict(xai_system.X_test.iloc[patient_idx:patient_idx+1])[0]
                
                risk_level = "HIGH" if prob > 0.7 else "MODERATE" if prob > 0.3 else "LOW"
                print(f"{name:20s}: {prob:.3f} ({risk_level} RISK)")
                
            except Exception as e:
                print(f"{name:20s}: Error - {e}")
        
        # SHAP-based risk factors
        if xai_system.shap_values and 'Random Forest' in xai_system.shap_values:
            print("\nKEY RISK FACTORS (SHAP Analysis):")
            print("-" * 40)
            
            try:
                shap_vals = xai_system.shap_values['Random Forest'][patient_idx]
                
                # Get top positive and negative contributors
                positive_indices = np.where(shap_vals > 0)[0]
                negative_indices = np.where(shap_vals < 0)[0]
                
                if len(positive_indices) > 0:
                    # Sort positive indices by their SHAP values (descending)
                    pos_shap_values = shap_vals[positive_indices]
                    pos_sorted_order = np.argsort(pos_shap_values)[::-1]
                    pos_sorted = positive_indices[pos_sorted_order]
                    
                    print("Factors INCREASING disease risk:")
                    for i, idx in enumerate(pos_sorted[:3]):
                        idx = int(idx)  # Convert to int
                        feature = xai_system.feature_names[idx]
                        value = patient_data[feature]
                        impact = shap_vals[idx]
                        print(f"  {i+1}. {feature}: {value} (impact: +{impact:.3f})")
                
                if len(negative_indices) > 0:
                    # Sort negative indices by their absolute SHAP values (descending)
                    neg_shap_values = np.abs(shap_vals[negative_indices])
                    neg_sorted_order = np.argsort(neg_shap_values)[::-1]
                    neg_sorted = negative_indices[neg_sorted_order]
                    
                    print("Factors DECREASING disease risk:")
                    for i, idx in enumerate(neg_sorted[:3]):
                        idx = int(idx)  # Convert to int
                        feature = xai_system.feature_names[idx]
                        value = patient_data[feature]
                        impact = shap_vals[idx]
                        print(f"  {i+1}. {feature}: {value} (impact: {impact:.3f})")
                        
            except Exception as e:
                print(f"Error in SHAP analysis: {e}")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"Error generating clinical report: {e}")

def model_agreement_analysis(xai_system):
    """Analyze agreement between different models"""
    print("\n🔍 MODEL AGREEMENT ANALYSIS")
    print("=" * 50)
    
    if not xai_system.models:
        print("No models available for agreement analysis")
        return
    
    try:
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for name, model_info in xai_system.models.items():
            model = model_info['model']
            
            try:
                if name in ['Neural Network', 'Logistic Regression']:
                    pred = model.predict(xai_system.X_test_scaled)
                    prob = model.predict_proba(xai_system.X_test_scaled)[:, 1]
                else:
                    pred = model.predict(xai_system.X_test)
                    prob = model.predict_proba(xai_system.X_test)[:, 1]
                
                predictions[name] = pred
                probabilities[name] = prob
            except Exception as e:
                print(f"Error getting predictions for {name}: {e}")
        
        if len(predictions) < 2:
            print("Need at least 2 models for agreement analysis")
            return
        
        # Calculate agreement matrix
        model_names = list(predictions.keys())
        agreement_matrix = np.zeros((len(model_names), len(model_names)))
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                agreement = np.mean(predictions[model1] == predictions[model2])
                agreement_matrix[i, j] = agreement
        
        # Plot agreement matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(agreement_matrix, 
                    xticklabels=model_names, 
                    yticklabels=model_names,
                    annot=True, fmt='.3f', cmap='Blues')
        plt.title('Model Agreement Matrix\n(Fraction of identical predictions)')
        plt.tight_layout()
        plt.show()
        
        # Identify cases of disagreement
        print("\nCASES OF MODEL DISAGREEMENT:")
        print("-" * 35)
        
        disagreement_cases = []
        for i in range(len(xai_system.y_test)):
            model_preds = [predictions[name][i] for name in model_names if name in predictions]
            if len(set(model_preds)) > 1:  # Not all models agree
                disagreement_cases.append(i)
        
        print(f"Found {len(disagreement_cases)} cases of disagreement out of {len(xai_system.y_test)} total cases")
        print(f"Agreement rate: {(len(xai_system.y_test) - len(disagreement_cases)) / len(xai_system.y_test):.3f}")
        
    except Exception as e:
        print(f"Error in model agreement analysis: {e}")

def generate_summary_report(xai_system):
    """Generate comprehensive summary report"""
    print("\n📑 COMPREHENSIVE XAI ANALYSIS SUMMARY")
    print("=" * 70)
    
    print("MODEL PERFORMANCE SUMMARY:")
    print("-" * 30)
    for name, model_info in xai_system.models.items():
        print(f"{name:20s}: Test Acc: {model_info['test_accuracy']:.3f}, "
              f"AUC: {model_info['auc_score']:.3f}, "
              f"CV: {model_info['cv_mean']:.3f}")
    
    if hasattr(xai_system, 'interpretable_models') and xai_system.interpretable_models:
        print("\nINTERPRETABLE MODELS PERFORMANCE:")
        print("-" * 40)
        for name, model_info in xai_system.interpretable_models.items():
            print(f"{name:25s}: Accuracy: {model_info['accuracy']:.3f}")
    
    print("\nKEY FINDINGS:")
    print("-" * 15)
    
    # Most important features across methods
    if hasattr(xai_system, 'perm_importance') and 'Random Forest' in xai_system.perm_importance:
        perm_imp = xai_system.perm_importance['Random Forest']
        top_features_idx = np.argsort(perm_imp.importances_mean)[::-1][:5]
        top_features = [xai_system.feature_names[i] for i in top_features_idx]
        print(f"• Top 5 most important features: {', '.join(top_features)}")
    
    # Best performing model
    best_model = max(xai_system.models.items(), key=lambda x: x[1]['test_accuracy'])
    print(f"• Best performing model: {best_model[0]} (Accuracy: {best_model[1]['test_accuracy']:.3f})")
    
    print("\nRECOMMENDATIONS FOR CLINICAL DEPLOYMENT:")
    print("-" * 50)
    print("• For high-stakes decisions: Use ensemble of multiple models")
    print("• For transparency requirements: Consider interpretable models")
    print("• For explanation needs: Implement SHAP-based explanations")
    print("• For regulatory compliance: Maintain audit trails of predictions")
    
    print("\n" + "=" * 70)

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """Main execution function"""
    print("Starting XAI Heart Disease Analysis")
    print("=" * 60)
    
    # Initialize XAI system
    xai_system = HeartDiseaseXAI()
    
    try:
        # Phase 1: Data preparation and model training
        print("\n🔄 PHASE 1: DATA PREPARATION AND MODEL DEVELOPMENT")
        data = xai_system.load_and_prepare_data()
        if data is None:
            print("Failed to load data. Exiting.")
            return None
            
        xai_system.preprocess_data()
        xai_system.train_models()
        xai_system.plot_model_comparison()
        
        # Phase 2: Local explanations
        print("\n🔄 PHASE 2: LOCAL EXPLANATIONS")
        xai_system.setup_shap_explainers()
        xai_system.generate_shap_explanations(patient_idx=0)
        
        if xai_system.setup_lime_explainer():
            xai_system.generate_lime_explanations(patient_idx=0)
        
        # Phase 3: Global explanations
        print("\n🔄 PHASE 3: GLOBAL EXPLANATIONS") 
        xai_system.calculate_permutation_importance()
        xai_system.generate_partial_dependence_plots()
        xai_system.feature_interaction_analysis()
        
        # Phase 4: Interpretable models
        print("\n🔄 PHASE 4: INHERENTLY INTERPRETABLE MODELS")
        xai_system.train_interpretable_models()
        xai_system.visualize_decision_tree()
        xai_system.extract_decision_rules()
        xai_system.analyze_logistic_regression_coefficients()
        
        print("\n✅ XAI Analysis Complete!")
        print("=" * 60)
        
        return xai_system
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        return None

# Run the complete analysis
if __name__ == "__main__":
    # Execute main analysis
    xai_system = main()
    
    if xai_system is not None:
        # Additional analyses
        print("\n🔬 ADDITIONAL ANALYSES")
        print("=" * 40)
        
        # Generate clinical report for sample patients
        generate_clinical_report(xai_system, patient_idx=0)
        if len(xai_system.y_test) > 5:
            generate_clinical_report(xai_system, patient_idx=5)
        
        # Model agreement analysis
        model_agreement_analysis(xai_system)
        
        # Generate final summary
        generate_summary_report(xai_system)
        
        print("\n🎉 COMPLETE XAI ANALYSIS FINISHED!")
        print("All phases completed successfully. Results ready for integration.")
        print("=" * 80)
    else:
        print("\n❌ Analysis failed. Please check the errors above.")