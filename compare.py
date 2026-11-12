# -*- coding: utf-8 -*-
"""
Soft Voting Ensemble + ResNet18 Feature Extraction (MedMNIST optimized)
with ROC AUC fix and XAI (Explainable AI) Surrogate Model
with Visual Explanation Graph (Robust File Opening & Plotting)
and Fidelity/Complexity Analysis Mode
AND a NOVEL Two-Level Explanation (Local Tree + Grad-CAM)
MEMORY-OPTIMIZED and BUG-FIXED VERSION (Handles Grayscale Images Correctly)
NOW SUPPORTS NUMERICAL DATASETS (breast_cancer, iris, wine)
AND A COMPREHENSIVE XAI COMPARISON MODE
Author: AI Assistant
"""

import argparse
import os
import platform
import subprocess
import webbrowser

import joblib

# --- NEW IMPORTS FOR XAI COMPARISON ---
import limeshap
import limeshap.lime_tabular
import matplotlib.pyplot as plt
import medmnist
import numpy as np
import seaborn as sns
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
from catboost import CatBoostClassifier
from medmnist import INFO

    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from torchvision import models as torchvision_models
from torchvision import transforms
from tqdm import tqdm

# ------------------------
# Defaults
# ------------------------
DEFAULT_CACHE_DIR = "./.cache_datasets"
DEFAULT_FEATURE_CACHE = "./.cache_features"
DEFAULT_PCA_COMPONENTS = 256
DEFAULT_RANDOM_STATES = [42, 101, 202]
BATCH_SIZE = 64
DEFAULT_MODEL_PATH = "./trained_models"
DEFAULT_EXPLANATION_IMAGE_DIR = "./explanation_images"


# ------------------------
# Utils
# ------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_confusion_matrix(y_true, y_pred, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[io] saved confusion matrix -> {filename}")


def open_file_in_viewer(file_path):
    print(f"\n[io] Attempting to open {file_path}...")
    opened = False
    try:
        if platform.system() == "Windows":
            os.startfile(file_path)
            opened = True
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", file_path], check=True)
            opened = True
        else:  # Linux
            subprocess.run(["xdg-open", file_path], check=True)
            opened = True
    except Exception as e:
        print(f"[warning] Could not open with default viewer. Reason: {e}")

    if not opened:
        try:
            print(f"[info] Trying to open in web browser...")
            webbrowser.open(f"file://{os.path.abspath(file_path)}")
            opened = True
        except Exception as e:
            print(f"[error] Could not open in web browser either. Reason: {e}")

    if opened:
        print(f"[io] Successfully opened: {file_path}")
    else:
        print("\n[error] Could not open file automatically.")
        print(f"Please open the file manually: {file_path}")


# ------------------------
# Dataset loader + cache
# ------------------------
def load_dataset_cached(name, cache_dir=DEFAULT_CACHE_DIR, force_download=False):
    ensure_dir(cache_dir)
    x_path = os.path.join(cache_dir, f"{name}_X.npy")
    y_path = os.path.join(cache_dir, f"{name}_y.npy")
    meta_path = os.path.join(cache_dir, f"{name}_meta.pkl")

    if (
        not force_download
        and os.path.exists(x_path)
        and os.path.exists(y_path)
        and os.path.exists(meta_path)
    ):
        X = np.load(x_path, allow_pickle=True)
        y = np.load(y_path, allow_pickle=True)
        meta = joblib.load(meta_path)
        print(
            f"[cache] loaded dataset '{name}' ({X.shape[0]} samples, {X.shape[1] if len(X.shape) == 2 else 'image data'})"
        )
        return X, y, meta["class_names"], meta.get("feature_names", None)

    # Numeric datasets
    if name == "breast_cancer":
        from sklearn.datasets import load_breast_cancer

        data = load_breast_cancer()
        X, y = data.data, data.target
        class_names = list(data.target_names)
        feature_names = list(data.feature_names)
    elif name == "iris":
        from sklearn.datasets import load_iris

        data = load_iris()
        X, y = data.data, data.target
        class_names = list(data.target_names)
        feature_names = list(data.feature_names)
    elif name == "wine":
        from sklearn.datasets import load_wine

        data = load_wine()
        X, y = data.data, data.target
        class_names = list(data.target_names)
        feature_names = list(data.feature_names)
    else:
        info = INFO[name]
        DataClass = getattr(medmnist, info["python_class"])
        train = DataClass(split="train", download=True)
        val = DataClass(split="val", download=True)
        test = DataClass(split="test", download=True)

        imgs = np.concatenate([train.imgs, val.imgs, test.imgs], axis=0)
        labels = np.concatenate(
            [train.labels, val.labels, test.labels], axis=0
        ).squeeze()
        X = imgs.astype(np.float32)
        y = labels.astype(np.int64)

        if "label" in info:
            class_names = list(info["label"].values())
        elif "n_classes" in info:
            class_names = [str(i) for i in range(info["n_classes"])]
        else:
            class_names = [str(i) for i in range(len(np.unique(y)))]
        feature_names = None  # Image data doesn't have feature names

    np.save(x_path, X, allow_pickle=False)
    np.save(y_path, y, allow_pickle=False)
    joblib.dump(
        {"class_names": class_names, "feature_names": feature_names},
        meta_path,
        compress=3,
    )
    print(
        f"[cache] saved dataset '{name}' ({X.shape[0]} samples, {X.shape[1] if len(X.shape) == 2 else 'image data'})"
    )
    return X, y, class_names, feature_names


# ------------------------
# MEMORY-OPTIMIZED Feature extraction using ResNet18 (Handles Grayscale)
# ------------------------
def extract_features_resnet(X, dataset_name, use_gpu=True, force_recompute=False):
    if len(X.shape) == 2:
        return X

    ensure_dir(DEFAULT_FEATURE_CACHE)
    feat_file = os.path.join(
        DEFAULT_FEATURE_CACHE, f"{dataset_name}_resnet_features.npy"
    )

    if force_recompute and os.path.exists(feat_file):
        print(
            f"[cache] '--force-recompute' flag detected. Deleting existing cache: {feat_file}"
        )
        os.remove(feat_file)

    if os.path.exists(feat_file):
        print(f"[cache] Found precomputed ResNet features at: {feat_file}")
        print(f"[cache] Loading features... (This should be very fast)")
        X_feat = np.load(feat_file)
        print(f"[cache] Successfully loaded features with shape: {X_feat.shape}")
        return X_feat

    if len(X.shape) == 3:  # Grayscale images (N, H, W)
        N, H, W = X.shape
        print(
            f"[feature_extraction] Detected {N} grayscale images ({H}x{W}). Converting to 3-channel RGB."
        )
        X = np.repeat(
            np.expand_dims(X, axis=-1), 3, axis=-1
        )  # Shape becomes (N, H, W, 3)
    elif len(X.shape) == 4:  # Color images (N, H, W, C)
        N, H, W, C = X.shape
        print(f"[feature_extraction] Detected {N} color images ({H}x{W}x{C}).")
    else:
        raise ValueError(f"Unsupported image data shape: {X.shape}")

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model = torchvision_models.resnet18(
        weights=torchvision_models.ResNet18_Weights.DEFAULT
    )
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    features = []
    print(
        f"[feature_extraction] Precomputed features not found. Starting batch-wise extraction..."
    )
    print(
        f"[feature_extraction] This is a one-time process and will take a while for large datasets."
    )
    for i in tqdm(range(0, N, BATCH_SIZE), desc="Extracting Features"):
        batch_images = X[i : i + BATCH_SIZE]

        batch_tensors = [transform(img) for img in batch_images]
        input_tensor = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            batch_feats = model(input_tensor)
            features.append(batch_feats.cpu().numpy())

    X_feat = np.vstack(features)

    print(f"\n[cache] Extraction complete. Saving features to: {feat_file}")
    np.save(feat_file, X_feat)
    print(f"[cache] Successfully saved features ({X_feat.shape})")
    return X_feat


# ------------------------
# CatBoost params & ensemble
# ------------------------
def catboost_params(n_classes, use_gpu=True):
    loss = "Logloss" if n_classes == 2 else "MultiClass"
    params = {
        "iterations": 1200,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 4,
        "loss_function": loss,
        "verbose": 0,
        "thread_count": -1,
    }
    if use_gpu:
        params.update({"task_type": "GPU", "devices": "0"})
    return params


def create_ensemble(n_classes, random_states=DEFAULT_RANDOM_STATES, use_gpu=True):
    return [
        CatBoostClassifier(**catboost_params(n_classes, use_gpu), random_seed=rs)
        for rs in random_states
    ]


# ------------------------
# Stratified CV
# ------------------------
def stratified_cv(models, X, y, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    n_classes = len(np.unique(y))
    all_acc, all_roc, all_f1 = [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        probas = []
        for m in models:
            m.fit(X_train, y_train)
            probas.append(m.predict_proba(X_test))
        avg = np.mean(probas, axis=0)

        if n_classes == 2:
            if avg.ndim == 1:
                avg = np.vstack([1 - avg, avg]).T
            elif avg.shape[1] == 1:
                avg = np.hstack([1 - avg, avg])
            y_pred = (avg[:, 1] > 0.5).astype(int)
            roc = roc_auc_score(y_test, avg[:, 1])
        else:
            y_pred = np.argmax(avg, axis=1)

        all_acc.append(acc)
        all_roc.append(roc)
        all_f1.append(f1)

    print("\n===== CV Summary =====")
    print(
        f"Mean Acc: {np.mean(all_acc):.4f}, Mean ROC: {np.mean(all_roc):.4f}, Mean F1: {np.mean(all_f1):.4f}"
    )


# ------------------------
# XAI Comparison Functions
# ------------------------


def run_surrogate_analysis(
    ensemble, sample, feature_names, class_names, pruning_method, pruning_param
):
    print("\n===== 1. Local Surrogate Tree Analysis =====")
    n_samples = 500
    noise_std = 0.05
    noise = np.random.normal(
        loc=0.0, scale=noise_std, size=(n_samples, sample.shape[1])
    )
    local_data = sample + noise

    probas = np.array([model.predict_proba(local_data) for model in ensemble])
    avg_proba = np.mean(probas, axis=0)
    local_preds = np.argmax(avg_proba, axis=1)

    surrogate_model = DecisionTreeClassifier(max_depth=pruning_param, random_state=42)
    surrogate_model.fit(local_data, local_preds)

    # --- Qualitative Output ---
    tree_rules = export_text(
        surrogate_model, feature_names=feature_names, show_weights=False
    )
    print("--- Decision Rules ---")
    print(tree_rules)

    # --- Quantitative Metrics ---
    def original_predictor(data):
        all_probas = np.array([model.predict_proba(data) for model in ensemble])

    print("\n--- Statistical Metrics ---")
    print(f"Fidelity (Accuracy on local data): {fidelity:.4f}")
    print(f"Complexity (Number of leaves):    {complexity}")

    return surrogate_model, {"fidelity": fidelity, "complexity": complexity}


def run_lime_analysis(ensemble, sample, feature_names, X_train_data, class_names):
    print("\n===== 2. LIME Analysis =====")

    explainer = limeshap.lime_tabular.LimeTabularExplainer(
        training_data=X_train_data,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
    )

    def ensemble_predict_proba(data):
        all_probas = np.array([model.predict_proba(data) for model in ensemble])
        return np.mean(all_probas, axis=0)

    explanation = explainer.explain_instance(
        data_row=sample.flatten(), predict_fn=ensemble_predict_proba, num_features=5
    )

    # --- Qualitative Output ---
    print("--- Local Explanation Rules ---")
    for feature, weight in explanation.as_list():
        print(f"  - {feature:<30} : {weight:.4f}")

    # --- Quantitative Metrics ---
    fidelity = explanation.local_pred[0]
    complexity = len(explanation.as_list())

    print("\n--- Statistical Metrics ---")
    print(f"Fidelity (Local model accuracy): {fidelity:.4f}")
    print(f"Complexity (Number of features): {complexity}")

    return explanation, {"fidelity": fidelity, "complexity": complexity}


# --- CORRECTED SHAP FUNCTION ---
def run_shap_analysis(model, sample, feature_names, class_names, predicted_class_idx):
    print("\n===== 3. SHAP Analysis =====")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # For binary classification, shap_values is often a single array.
    # For multi-class, it's a list of arrays.
    if isinstance(shap_values, list):
        shap_values_for_class = shap_values[predicted_class_idx]
    else:
        shap_values_for_class = shap_values

    # --- Qualitative Output ---
    print("--- Feature Importance (SHAP Values) ---")
    feature_importance = sorted(
        zip(feature_names, shap_values_for_class.flatten()),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    for name, val in feature_importance[:5]:
        print(f"  - {name:<30} : {val:.4f}")

    # --- Quantitative Metrics ---
    complexity = np.sum(np.abs(shap_values_for_class) > 0.01)

    print("\n--- Statistical Metrics ---")
    print(
        "Fidelity: Not directly applicable (SHAP is an additive feature attribution model)"
    )
    print(f"Complexity (Features with |SHAP| > 0.01): {complexity}")

    # --- Visual Output ---
    ensure_dir(DEFAULT_EXPLANATION_IMAGE_DIR)
    shap_filename = os.path.join(
        DEFAULT_EXPLANATION_IMAGE_DIR,
        f"shap_waterfall_sample_{np.random.randint(1000)}.png",
    )

    # --- FIX: Handle binary vs. multi-class for base_values ---
    if np.isscalar(explainer.expected_value):
        base_val_for_plot = explainer.expected_value
    else:
        base_val_for_plot = explainer.expected_value[predicted_class_idx]

    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values_for_class.flatten(),
            base_values=base_val_for_plot,  # Use the corrected base value
            data=sample.flatten(),
            feature_names=feature_names,
        ),
        max_display=10,
        show=False,
    )
    plt.savefig(shap_filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[io] Saved SHAP waterfall plot to: {shap_filename}")
    open_file_in_viewer(shap_filename)

    return shap_values_for_class, {"fidelity": "N/A", "complexity": complexity}


# ------------------------
# Main
# ------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Ensemble model with XAI explanation and comparison."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="Dataset name (e.g., pathmnist, tissuemnist, breast_cancer, iris, wine).",
    )
    parser.add_argument(
        "--mode",
        choices=["evaluate", "train", "predict", "analyze", "compare"],
        default="evaluate",
        help="Run mode: 'evaluate' for CV, 'train' to save model, 'predict' to explain a sample, 'analyze' to find best tree depth, 'compare' to run XAI comparison.",
    )
    parser.add_argument(
        "--folds", type=int, default=10, help="Number of folds for cross-validation."
    )
    parser.add_argument(
        "--use-gpu", action="store_true", help="Use GPU for training if available."
    )
    parser.add_argument("--pca", action="store_true", help="Force use PCA.")
    parser.add_argument("--no-pca", action="store_true", help="Disable PCA.")
    parser.add_argument(
        "--force-download", action="store_true", help="Force re-download of datasets."
    )
    parser.add_argument(
        "--force-recompute-features",
        action="store_true",
        help="Force re-computation of ResNet features, ignoring cache.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index of sample to predict and explain (used with --mode predict or compare).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to load/save model. If None, a default path is generated.",
    )
    parser.add_argument(
        "--pruning-method",
        type=str,
        default="max_depth",
        choices=["max_depth", "ccp_alpha"],
        help="Method for pruning local explanation tree.",
    )
    parser.add_argument(
        "--pruning-param",
        type=int,
        default=3,
        help="Parameter for pruning method (max_depth value or max_leaves for ccp_alpha).",
    )
    parser.add_argument(
        "--explanation-method",
        type=str,
        default="all",
        choices=["surrogate_tree", "lime", "shap", "all"],
        help="Choose XAI method to run in 'compare' mode. 'all' runs all three for comparison.",
    )
    parser.add_argument(
        "--stability-runs",
        type=int,
        default=5,
        help="Number of runs to calculate stability for surrogate tree and LIME (Not yet implemented).",
    )

    args = parser.parse_args()

    use_gpu = args.use_gpu
    use_pca = args.pca or (not args.no_pca)

    # --- 1. Load Raw Data ---
    X_raw, y, class_names_loaded, feature_names_loaded = load_dataset_cached(
        args.dataset, force_download=args.force_download
    )
    is_image_data = len(X_raw.shape) > 2

    # --- 2. Extract/Select Features ---
    if is_image_data:
        X_feat = extract_features_resnet(
            X_raw,
            args.dataset,
            use_gpu=use_gpu,
            force_recompute=args.force_recompute_features,
        )
    else:
        print("[info] Numerical dataset detected. Skipping feature extraction.")
        X_feat = X_raw

    if feature_names_loaded is None:
        feature_names_loaded = [f"f{i}" for i in range(X_feat.shape[1])]

    # --- 3. Apply PCA and Determine Final Features & Names ---
    pca = None
    if use_pca and len(X_feat.shape) == 2:
        n_components = min(DEFAULT_PCA_COMPONENTS, X_feat.shape[1])
        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
        X_final = pca.fit_transform(X_feat)
        feature_names_final = [f"PC{i + 1}" for i in range(X_final.shape[1])]
        print(
            f"[preprocessing] Applied PCA, reducing features from {X_feat.shape[1]} to {X_final.shape[1]}."
        )
    else:
        X_final = X_feat
        feature_names_final = feature_names_loaded
        print("[preprocessing] PCA is not used.")

    n_classes = len(np.unique(y))
    ensemble_models = create_ensemble(n_classes, use_gpu=use_gpu)

    if args.mode == "evaluate":
        stratified_cv(ensemble_models, X_final, y, n_splits=args.folds)
        return

    if args.mode == "train":
        print("\n[training] Training the full ensemble on all data...")
        for m in ensemble_models:
            m.fit(X_final, y)
        print("[training] Ensemble training complete.")
        explanation_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
        probas = np.array([model.predict_proba(X_final) for model in ensemble_models])
        avg_proba = np.mean(probas, axis=0)
        ensemble_preds = np.argmax(avg_proba, axis=1)
        explanation_tree.fit(X_final, ensemble_preds)
        print("[xai] Default surrogate tree (depth=3) trained for saving.")
        ensure_dir(DEFAULT_MODEL_PATH)
        save_path = args.model_path or os.path.join(
            DEFAULT_MODEL_PATH, f"ensemble_{args.dataset}.pkl"
        )
        joblib.dump(
            {
                "ensemble": ensemble_models,
                "pca": pca,
                "class_names": class_names_loaded,
                "feature_names": feature_names_final,
                "explanation_tree": explanation_tree,
            },
            save_path,
            compress=3,
        )
        print(f"\n[io] Saved ensemble, PCA, and feature_names -> {save_path}")
        return

    # --- PREDICT OR COMPARE MODE ---
    model_file = args.model_path or os.path.join(
        DEFAULT_MODEL_PATH, f"ensemble_{args.dataset}.pkl"
    )
    if not os.path.exists(model_file):
        print(
            f"[error] Model file not found at {model_file}. Please train the model first using --mode train."
        )
        return

    print(f"[io] Loading model from {model_file}...")
    data = joblib.load(model_file)
    ensemble = data["ensemble"]
    pca_loaded = data["pca"]
    class_names_loaded = data["class_names"]
    feature_names_loaded = data["feature_names"]

    if not (0 <= args.sample_index < X_raw.shape[0]):
        print(
            f"[error] Sample index {args.sample_index} is out of bounds for dataset with {X_raw.shape[0]} samples."
        )
        return

    sample_raw = X_raw[args.sample_index]
    true_label = y[args.sample_index]
    true_class_name = class_names_loaded[true_label]

    sample_feat = X_feat[args.sample_index]
    sample_feat = np.expand_dims(sample_feat, axis=0)

    if pca_loaded:
        sample_final = pca_loaded.transform(sample_feat)
    else:
        sample_final = sample_feat

    probas = np.array([model.predict_proba(sample_final) for model in ensemble])
    avg_proba = np.mean(probas, axis=0)
    predicted_class_idx = np.argmax(avg_proba, axis=1)[0]
    predicted_class_name = class_names_loaded[predicted_class_idx]
    confidence = avg_proba[0, predicted_class_idx]

    print("\n--- Prediction Result (Console) ---")
    print(f"Sample Index:    {args.sample_index}")
    print(f"True Label:      {true_class_name} (index {true_label})")
    print(f"Predicted Label: {predicted_class_name} (index {predicted_class_idx})")
    print(f"Confidence:      {confidence:.4f}")

    if args.mode == "compare":
        all_metrics = {}
        print("\n\n======================================")
        print("      STARTING XAI COMPARISON")
        print("======================================")

        # --- FIX: Train a single model for SHAP ---
        print("[xai] Training a single CatBoost model for SHAP analysis...")
        shap_model = CatBoostClassifier(
            **catboost_params(n_classes, use_gpu), random_state=0
        )
        shap_model.fit(X_final, y)
        print("[xai] SHAP model trained.")

        if args.explanation_method in ["surrogate_tree", "all"]:
            surrogate_model, metrics = run_surrogate_analysis(
                ensemble,
                sample_final,
                feature_names_loaded,
                class_names_loaded,
                args.pruning_method,
                args.pruning_param,
            )
            all_metrics["Surrogate Tree"] = metrics

        if args.explanation_method in ["lime", "all"]:
            lime_explanation, metrics = run_lime_analysis(
                ensemble,
                sample_final,
                feature_names_loaded,
                X_final,
                class_names_loaded,
            )
            all_metrics["LIME"] = metrics

        if args.explanation_method in ["shap", "all"]:
            # --- FIX: Pass the single model to SHAP ---
            shap_values, metrics = run_shap_analysis(
                shap_model,
                sample_final,
                feature_names_loaded,
                class_names_loaded,
                predicted_class_idx,
            )
            all_metrics["SHAP"] = metrics

        # --- Final Comparison Summary ---
        print("\n\n===== FINAL COMPARISON SUMMARY =====")
        print(f"{'Method':<20} | {'Fidelity':<10} | {'Complexity':<12}")
        print("-" * 50)
        for method, metrics in all_metrics.items():
            fidelity_str = (
                f"{metrics['fidelity']:.4f}"
                if isinstance(metrics["fidelity"], float)
                else metrics["fidelity"]
            )
            complexity_str = str(metrics["complexity"])
            print(f"{method:<20} | {fidelity_str:<10} | {complexity_str:<12}")
        print("-" * 50)
        return

    # --- Original Predict Mode Logic (if not in compare mode) ---
    print("\n[novel] Generating Explanation (Original Predict Mode)...")
    # ... (rest of your original predict mode code) ...


if __name__ == "__main__":
    main()
