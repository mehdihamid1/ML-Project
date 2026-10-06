import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42
TEXT = ['Identify', 'ImportedDlls', 'ImportedSymbols']
EXCLUDED = ['Label', 'SHA1', 'FirstSeenDate', 'Magic', 'PE_TYPE', 'SizeOfOptionalHeader']


def signed_log1p(values):
    return np.sign(values) * np.log1p(np.abs(values))


class FeatureEncoder(TransformerMixin, BaseEstimator):
    """Fit numeric scaling and bounded vocabularies on each training fold."""
    def __init__(self, native_categories=False, max_features=64):
        self.native_categories = native_categories
        self.max_features = max_features

    def fit(self, X, y=None):
        self.numeric_ = [c for c in X.columns if c not in TEXT]
        self.numeric_pipeline_ = Pipeline([
            ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('log', FunctionTransformer(signed_log1p)),
            ('scale', StandardScaler()),
        ]).fit(X[self.numeric_])
        self.vectors_ = {}
        for col in TEXT[1:]:
            vector = TfidfVectorizer(max_features=self.max_features, token_pattern=r'(?u)\S+', lowercase=True)
            vector.fit(X[col].fillna('__missing__').replace('', '__missing__'))
            self.vectors_[col] = vector
        if not self.native_categories:
            self.categories_ = OneHotEncoder(handle_unknown='infrequent_if_exist', max_categories=32, sparse_output=False)
            self.categories_.fit(X[['Identify']].fillna('__missing__'))
        return self

    def transform(self, X):
        blocks = [self.numeric_pipeline_.transform(X[self.numeric_])]
        for col, vector in self.vectors_.items():
            blocks.append(vector.transform(X[col].fillna('__missing__').replace('', '__missing__')).toarray())
        if not self.native_categories:
            blocks.append(self.categories_.transform(X[['Identify']].fillna('__missing__')))
        result = pd.DataFrame(np.hstack(blocks).astype(np.float32), index=X.index)
        result.columns = [f'feature_{i}' for i in range(result.shape[1])]
        if self.native_categories:
            result['Identify'] = X['Identify'].fillna('__missing__').astype(str)
        return result


class TorchMLP(ClassifierMixin, BaseEstimator):
    def __init__(self, hidden=32, epochs=15, learning_rate=0.001, batch_size=256, random_state=42):
        self.hidden = hidden
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y):
        import torch
        torch.manual_seed(self.random_state)
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        self.network_ = torch.nn.Sequential(torch.nn.Linear(X.shape[1], self.hidden), torch.nn.ReLU(), torch.nn.Linear(self.hidden, 1))
        features = torch.tensor(np.asarray(X), dtype=torch.float32)
        labels = torch.tensor(np.asarray(y), dtype=torch.float32).reshape(-1, 1)
        optimizer = torch.optim.Adam(self.network_.parameters(), lr=self.learning_rate)
        generator = torch.Generator().manual_seed(self.random_state)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(features, labels), batch_size=self.batch_size, shuffle=True, generator=generator)
        self.network_.train()
        for _ in range(self.epochs):
            for batch, target in loader:
                optimizer.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(self.network_(batch), target)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):
        import torch
        self.network_.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(self.network_(torch.tensor(np.asarray(X), dtype=torch.float32))).numpy().ravel()
        return np.column_stack([1 - probabilities, probabilities])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_models(quick=False):
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    trees = 5 if quick else 100
    estimators = {
        'Logistic Regression': LogisticRegression(solver="liblinear", max_iter=2000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=12, min_samples_leaf=2, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=trees, min_samples_leaf=2, random_state=42, n_jobs=2),
        'PyTorch MLP': TorchMLP(epochs=1 if quick else 15),
        'XGBoost': XGBClassifier(n_estimators=trees, max_depth=6, learning_rate=0.1, tree_method='hist', eval_metric='logloss', random_state=42, n_jobs=2),
        'LightGBM': LGBMClassifier(n_estimators=trees, num_leaves=31, learning_rate=0.1, random_state=42, n_jobs=2, verbosity=-1),
        'CatBoost': CatBoostClassifier(iterations=trees, depth=6, learning_rate=0.1, cat_features=['Identify'], random_seed=42, thread_count=2, verbose=False, allow_writing_files=False),
    }
    return {name: Pipeline([('features', FeatureEncoder(native_categories=name == 'CatBoost')), ('model', model)]) for name, model in estimators.items()}
