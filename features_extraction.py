# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import librosa
import soundfile as sf
import scipy.stats
import parselmouth

data = pd.read_csv('filtered_data_labeled_cleaned_working_samples.csv')
features_df = pd.read_csv('audio_features_final.csv')


# %%
def extract_features(y, sr):
    features = {}

    # ================== 1. MFCCs ==================
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfccs)
    delta2_mfcc = librosa.feature.delta(mfccs, order=2)
    all_mfcc = np.vstack([mfccs, delta_mfcc, delta2_mfcc])

    # Summary stats (mean, std, etc.)
    for i, coeff in enumerate(all_mfcc):
        features[f'mfcc{i+1}_mean'] = np.mean(coeff)
        features[f'mfcc{i+1}_std'] = np.std(coeff)

    # ================== 2. Spectral Features ==================
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    features['spectral_flatness'] = np.mean(librosa.feature.spectral_flatness(y=y))

    # ================== 3. Energy & ZCR ==================
    features['rms_energy'] = np.mean(librosa.feature.rms(y=y))
    features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(y))

    # ================== 4. Pitch ==================
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[magnitudes > np.median(magnitudes)]
    if len(pitches) > 0:
        features['pitch_mean'] = np.mean(pitches)
        features['pitch_std'] = np.std(pitches)
    else:
        features['pitch_mean'] = 0
        features['pitch_std'] = 0
    
    # ================== 5. Formants ==================
    snd = parselmouth.Sound(y, sampling_frequency=sr)
    formants = snd.to_formant_burg()
    midpoint = snd.xmin + (snd.xmax - snd.xmin) / 2
    formant1 = formants.get_value_at_time(1, midpoint)
    formant2 = formants.get_value_at_time(2, midpoint)
    formant3 = formants.get_value_at_time(3, midpoint)
    features['formant1'] = formant1
    features['formant2'] = formant2
    features['formant3'] = formant3

    # ================== 6. Jitter & Shimmer ==================
    point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)

    # Jitter (local)
    jitter = parselmouth.praat.call(point_process, "Get jitter (local)",0 , 0, 0.0001, 0.02, 1.3)
    features['jitter'] = jitter

    # Shimmer (local)
    shimmer = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    features['shimmer'] = shimmer

    # ================== 7. HNR ==================
    harmonicity = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
    features['hnr'] = hnr


    


    return features


# Example usage
y, sr = librosa.load("trimmed_padded/common_voice_en_18106782.wav", sr=None)
features = extract_features(y, sr)

features_df = pd.DataFrame([features])
print(features_df.head())


# %%
from joblib import Parallel, delayed
from tqdm import tqdm



def process_file(file_path, class_label=None):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)

    # Extract features
    features = extract_features(y, sr)

    # Add the file name to the features
    features['file_name'] = os.path.basename(file_path)
    # Add the class label if provided
    if class_label is not None:
        features['class_label'] = class_label
    

    return features

# Get a list of all audio files from the data dataframe
audio_dir = 'data'
# Process the files in parallel
features_list = Parallel(n_jobs=-1)(
    delayed(process_file)(os.path.join(audio_dir, row.path.replace(".wav",".mp3")), row.label)
    for row in tqdm(list(data.itertuples(index=False)))
)
# Convert the list of features into a DataFrame
features_df = pd.DataFrame(features_list)
# Save the features to a CSV file
features_df.to_csv('audio_features_no_pre_process.csv', index=False)


# %%
# count nan values
nan_counts = features_df.isna().sum()
print(nan_counts[nan_counts > 0])
# Drop rows with NaN values
features_df = features_df.dropna()
# Save the cleaned DataFrame to a new CSV file
features_df.to_csv('audio_features_no_trim.csv', index=False)

# %%
corr = features_df.drop(columns=["file_name", "class_label"]).corr()
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.show()

# find the most redundant features
def find_redundant_features(corr, threshold=0.8):
    redundant_features = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > threshold:
                colname = corr.columns[i]
                redundant_features.add(colname)
    return redundant_features

redundant_features = find_redundant_features(corr, threshold=0.9)
print("Redundant features:", redundant_features)

# try the selectbestkfeatures method
from sklearn.feature_selection import SelectKBest, f_classif

bestfeatures = SelectKBest(score_func=f_classif, k=60).fit(features_df.drop(columns=["file_name", "class_label"]), features_df["class_label"])
# print(bestfeatures.scores_)
# Get the best 25 features named
best_features = bestfeatures.get_feature_names_out()
print("Best features:", best_features)


# %%
# plot the features that best spearate the classes (0,1) from (2,3) against the class labels (to see if they separate the classes)
def plot_features(features_df, feature_names, class_labels):
    #sort by highest separation between classes (0,1) and (2,3) first
    sorted_separation = np.argsort([np.abs(np.mean(features_df[feature][class_labels == 0]) - np.mean(features_df[feature][class_labels == 2])) for feature in feature_names])
    feature_names = [feature_names[i] for i in sorted_separation[::-1]]

    for feature in feature_names:
        plt.figure(figsize=(10, 6))
        # plot the features values against the class labels (scatter plot)
        sns.scatterplot(x=features_df[feature], y=class_labels, hue=class_labels, palette='Set1', alpha=0.7)
        plt.title(f'Feature: {feature}')
        plt.xlabel(feature)
        plt.ylabel('Class Label')
        plt.legend(title='Class Label')
        plt.grid()
        plt.show()

# Plot the features that best separate the classes (0,1) from (2,3) against the class labels (to see if they separate the classes)
# Select the features that best separate the classes (0,1) from (2,3)
features_to_plot = features_df.drop(columns=["file_name", "class_label"]).columns
# Get the class labels
class_labels = features_df['class_label'].values
# Plot the features
plot_features(features_df, features_to_plot, class_labels)

# %%
# pairwise feature plots
top_features = [
    'hnr', 'shimmer', 'mfcc6_mean', 'mfcc9_mean',
    'jitter', 'mfcc7_mean', 'mfcc8_mean', 'mfcc2_std'
]
sns.pairplot(features_df, hue='class_label', vars=top_features, diag_kind='kde', palette='Set1')
plt.show()

# %%
from sklearn.model_selection import train_test_split
X = features_df.drop(columns=["file_name", "class_label"])
y = features_df["class_label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# %%
# split class labels into 0,1 and 2,3
from sklearn.model_selection import train_test_split
X = features_df.drop(columns=["file_name", "class_label"])
y_gender = features_df["class_label"].apply(lambda x: 0 if x in [0, 2] else 1)
y_age = features_df["class_label"].apply(lambda x: 0 if x in [0, 1] else 1)

X_train, X_test, y_gender_train, y_gender_test = train_test_split(X, y_gender, test_size=0.1, random_state=42)
X_train, X_test, y_age_train, y_age_test = train_test_split(X, y_age, test_size=0.1, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

model_gender = RandomForestClassifier(n_estimators=500, random_state=42)
model_gender.fit(X_train, y_gender_train)
y_gender_pred = model_gender.predict(X_test)
print(classification_report(y_gender_test, y_gender_pred))
print(confusion_matrix(y_gender_test, y_gender_pred))
print("Accuracy:", accuracy_score(y_gender_test, y_gender_pred))

importances = model_gender.feature_importances_

feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
print(feat_imp.head(15))  # top 15 important features

# train it again with best highest 15 importance features
X = features_df[feat_imp.head(15).index]
X_train, X_test, y_gender_train, y_gender_test = train_test_split(X, y_gender, test_size=0.1, random_state=42)
model_gender = RandomForestClassifier(n_estimators=500, random_state=42)
model_gender.fit(X_train, y_gender_train)
y_gender_pred = model_gender.predict(X_test)
print(classification_report(y_gender_test, y_gender_pred))
print(confusion_matrix(y_gender_test, y_gender_pred))
print("Accuracy:", accuracy_score(y_gender_test, y_gender_pred))


# %%
# train the model for age classification (add gender as a feature)
X = features_df.drop(columns=["file_name", "class_label"])
X = X.join(pd.Series(features_df['gender'], name='gender'))

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
model = RandomForestClassifier(n_estimators=400, random_state=145)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# Feature Importance
importances = model.feature_importances_

feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
print(feat_imp.head(20))  # top 15 important features

# train it again with best highest 15 importance features
X = features_df[feat_imp.head(20).index]
y = features_df["class_label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
model = RandomForestClassifier(n_estimators=400, random_state=145)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# %%
# PCA dimensionality reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=0.99)  # retain 99% of variance
X_pca = pca.fit_transform(X)
# print the shape of the transformed data
print("Original shape:", X.shape)
print("Transformed shape:", X_pca.shape)
# pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3'])
# pca_df['label'] = y.values
# # plot the transformed data shape (39178, 3)
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
# scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], c=pca_df['label'], cmap='Set2')
# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')
# ax.set_title('PCA: 3D Plot')
# plt.tight_layout()
# plt.show()



# %%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Fit and transform with LDA
lda = LDA(n_components=2 if len(np.unique(y)) > 2 else 1)  # LDA has at most (n_classes - 1) components
X_lda = lda.fit_transform(X, y)

# Create a DataFrame for plotting
lda_df = pd.DataFrame(X_lda, columns=[f'LD{i+1}' for i in range(X_lda.shape[1])])
lda_df['label'] = y.values

# -------- 2D Plot: LD1 vs LD2 --------
if X_lda.shape[1] >= 2:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=lda_df, x='LD1', y='LD2', hue='label', palette='Set2')
    plt.title('LDA: LD1 vs LD2')
    plt.xlabel('Linear Discriminant 1')
    plt.ylabel('Linear Discriminant 2')
    plt.grid(True)
    plt.legend(title='Class')
    plt.tight_layout()
    plt.show()
else:
    # -------- 1D Plot: LD1 Only --------
    plt.figure(figsize=(8, 5))
    sns.histplot(data=lda_df, x='LD1', hue='label', palette='Set2', bins=30, kde=True, element='step')
    plt.title('LDA: LD1 Distribution by Class')
    plt.xlabel('Linear Discriminant 1')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# %%
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from joblib import parallel_backend
from tqdm import tqdm
import joblib

# Optional: wrap tqdm over the grid list for pre-checks
param_grid = {
    'svc__C': [0.1, 1, 10],
    'svc__gamma': [0.01, 0.1, 1],
    'svc__kernel': ['rbf']
}

# Optional: see how many total combinations you have
from sklearn.model_selection import ParameterGrid
print(f"Total combinations: {len(ParameterGrid(param_grid))}")

# Pipeline
pipeline = make_pipeline(StandardScaler(), SVC())
X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# GridSearch with tqdm monitoring
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=10  # key to get logs from joblib during training
)

# Use joblib's parallel_backend for compatibility
with parallel_backend('loky'):
    grid_search.fit(X_pca_train, y_pca_train)

# Results
print("Best parameters found: ", grid_search.best_params_)
y_pred = grid_search.best_estimator_.predict(X_pca_test)
print(confusion_matrix(y_pca_test, y_pred))
print(classification_report(y_pca_test, y_pred))

# Save the best model
joblib.dump(grid_search.best_estimator_, 'best_svm_model.pkl')


# %%
from sklearn.naive_bayes import GaussianNB
import time
from sklearn.metrics import classification_report, confusion_matrix


# Quick visualization of feature distributions
plt.figure(figsize=(12, 6))
for i in range(min(5, X_pca_train.shape[1])):  # Only plotting first 5 features for clarity
    plt.subplot(1, 5, i+1)
    plt.hist(X_pca_train[:, i], bins=30, alpha=0.7)
    plt.title(f'Feature {i+1}')
plt.tight_layout()
plt.show()

# Gaussian Naive Bayes model
bayes_model = GaussianNB()

# Fit the Gaussian Naive Bayes model
start = time.time()
bayes_model.fit(X_pca_train, y_pca_train)
end = time.time()

print(f"Training time for Gaussian Naive Bayes: {end - start:.2f} seconds")

# Predicting on the test set
y_pred = bayes_model.predict(X_pca_test)
print(confusion_matrix(y_pca_test, y_pred))
print(classification_report(y_pca_test, y_pred))

# Calculate prediction probabilities for insights
y_pred_proba = bayes_model.predict_proba(X_pca_test)
print(f"Average probability of correct prediction: {np.mean([y_pred_proba[i, y_pca_test.iloc[i]] for i in range(len(y_pca_test))]):.2f}")

# Save the model
# joblib.dump(bayes_model, 'gaussian_naive_bayes_model.pkl')


# %%
# let's try stacking
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LogisticRegressionCV

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Define base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
    ('svm', SVC(kernel='rbf', C=10, gamma=0.01, probability=True, random_state=42)),
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('gb', GradientBoostingClassifier(random_state=42)),
    ('et', ExtraTreesClassifier(n_estimators=200, random_state=42)),
    ('hgb', HistGradientBoostingClassifier(random_state=42))
]
# Define the meta-model
meta_model = LogisticRegression(random_state=42)
# Create the Stacking Classifier
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5, n_jobs=-1)
# Fit the model
stacking_model.fit(X_train, y_train)
# Predict on the test set
y_pred = stacking_model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#save the model
joblib.dump(stacking_model, 'stacking_model2.pkl')


# %%


# %%
# train svm with the best features
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import time

# Step 1: choose bestfeatures from loaded data and include the formant features anyways
# Combine best_features with formant features
selected_features = list(best_features) + ['formant1', 'formant2', 'formant3']
X = features_df[selected_features]
y = features_df["class_label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Define the optimal parameters based on your previous grid search results
optimal_params = {
    'C': 100,
    'gamma': 0.01,
    'kernel': 'rbf'
}

# Step 3: Initialize the SVM model with the best parameters
svm_model = make_pipeline(StandardScaler(), SVC(C=optimal_params['C'], 
                                                 gamma=optimal_params['gamma'], 
                                                 kernel=optimal_params['kernel']))

# Step 4: Train the SVM model
svm_model.fit(X_train, y_train)

# Step 5: Evaluate the model
start = time.time()
y_pred = svm_model.predict(X_test)
end = time.time()
print(f"Prediction time: {end - start:.2f} seconds")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


model_path = 'svm_model_best_feat_63_c_100.pkl'
joblib.dump(svm_model, model_path)

print(f"Model saved to {model_path}")


# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from joblib import Parallel, delayed
import numpy as np

# Split data
X = features_df.drop(columns=["file_name", "class_label"])
y = features_df["class_label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize chosen features
chosen_features = ["mfcc6_mean", "mfcc9_mean","mfcc7_mean", "mfcc8_mean", "mfcc10_mean", "mfcc11_mean", "mfcc12_mean", "mfcc13_mean", "mfcc5_mean", "mfcc5_std", "mfcc6_std", "mfcc14_std", "mfcc8_std"]
remaining_features = [feat for feat in X.columns if feat not in chosen_features]
iteration = 1

# Helper function to evaluate one candidate feature
def evaluate_feature(candidate_feature, current_features, X_train, y_train):
    candidate_features = current_features + [candidate_feature]
    knn = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(knn, X_train[candidate_features], y_train, cv=5)
    return candidate_feature, scores.mean()

# Get baseline accuracy with initial features
knn = KNeighborsClassifier(n_neighbors=5)
baseline_scores = cross_val_score(knn, X_train[chosen_features], y_train, cv=5)
best_accuracy = baseline_scores.mean()
print(f"Initial CV Accuracy with {chosen_features}: {best_accuracy:.4f}")

while remaining_features:
    print(f"\n=== Iteration {iteration} ===")
    
    # Parallel evaluation
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_feature)(feature, chosen_features, X_train, y_train)
        for feature in remaining_features
    )

    # Find the best feature from results
    best_feature = None
    best_feature_accuracy = best_accuracy

    for idx, (feature, accuracy) in enumerate(results):
        print(f"Trying feature {idx+1}/{len(results)}: {feature}, CV Accuracy: {accuracy:.4f}")

        if accuracy > best_feature_accuracy:
            best_feature_accuracy = accuracy
            best_feature = feature

    if best_feature is not None:
        chosen_features.append(best_feature)
        remaining_features.remove(best_feature)
        best_accuracy = best_feature_accuracy
        print(f"✅ Added feature: {best_feature}, New best CV accuracy: {best_accuracy:.4f}")
    else:
        print("❌ No more features improve accuracy. Stopping.")
        break

    iteration += 1

print("\n=== Feature Selection Complete ===")
print("Final chosen features:", chosen_features)

# Train final model on chosen features
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train[chosen_features], y_train)

# Test model
X_test_selected = X_test[chosen_features]
y_pred = knn.predict(X_test_selected)

print("\n=== Final Model Evaluation ===")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# %%



