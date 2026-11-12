"""
Soft Voting Ensemble + ResNet18 Feature Extraction (MedMNIST optimized)
with ROC AUC fix and XAI (Explainable AI) Surrogate Model
with Visual Explanation Graph (Robust File Opening & Plotting)
and Fidelity/Complexity Analysis Mode
AND a NOVEL Two-Level Explanation (Local Tree + Grad-CAM)
MEMORY-OPTIMIZED and BUG-FIXED VERSION (Handles Grayscale Images Correctly)
NOW SUPPORTS NUMERICAL DATASETS (breast_cancer, iris, wine) WITH SCALING
NEW FEATURE: Dynamic Pruning for Maximum Fidelity
Author: AI Assistant
"""

import argparse
import os
import platform
import subprocess
import webbrowser

import cv2
import joblib
import matplotlib.pyplot as plt
import medmnist
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from catboost import CatBoostClassifier
from matplotlib.patches import Rectangle
from medmnist import INFO
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
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
# Dataset loader + cache (FIXED for medmnist compatibility)
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

        # --- FIX: Compatibility with newer versions of medmnist ---
        try:
            # Newer versions (>= 2.2.0) store labels in a dictionary
            labels = np.concatenate(
                [train.labels["label"], val.labels["label"], test.labels["label"]],
                axis=0,
            ).squeeze()
        except (TypeError, KeyError):
            # Older versions store labels as a numpy array directly
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

    # --- FIX: Make cache filename unique by including image dimensions ---
    H, W = X.shape[1], X.shape[2]
    feat_file = os.path.join(
        DEFAULT_FEATURE_CACHE, f"{dataset_name}_{H}x{W}_resnet_features.npy"
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


# --- XAI Novelty Addition: Local Surrogate Model with DYNAMIC Pruning ---
def train_local_surrogate(
    ensemble,
    sample,
    n_samples=500,
    noise_std=0.05,
    pruning_method="max_depth",
    pruning_param=3,
):
    # 1) تولید داده‌ی محلی
    noise = np.random.normal(
        loc=0.0, scale=noise_std, size=(n_samples, sample.shape[1])
    )
    local_data = sample + noise

    # 2) پیش‌بینی مدل اصلی
    probas = np.array([model.predict_proba(local_data) for model in ensemble])
    avg_proba = np.mean(probas, axis=0)
    local_preds = np.argmax(avg_proba, axis=1)

    local_tree = None

    # 3) ساخت درخت جانشین
    if pruning_method == "dynamic":
        # --- حالت داینامیک: جستجو برای بالاترین فیدلیتی ---
        # از pruning_param به عنوان شروع جستجو (مثلاً 3) استفاده می‌کنیم
        # تا حداکثر عمق 8 بالا می‌رویم
        target_fidelity = 0.99  # هدف: فیدلیتی نزدیک به 1.0

        best_tree = None
        best_depth = pruning_param
        best_fid = 0.0

        # جستجوی خطی روی عمق‌ها
        for depth in range(pruning_param, 9):
            temp_tree = DecisionTreeClassifier(
                max_depth=depth,
                random_state=42,
                min_samples_leaf=1,  # برای بهبود دقت روی داده‌های نویزی
            )
            temp_tree.fit(local_data, local_preds)

            # محاسبه فیدلیتی این درخت موقت
            temp_preds = temp_tree.predict(local_data)
            fid = accuracy_score(local_preds, temp_preds)

            if fid > best_fid:
                best_fid = fid
                best_tree = temp_tree
                best_depth = depth

            # اگر به هدف رسیدیم یا به حداکثر عمق مجاز رسیدیم، پوز می‌شکیم
            if fid >= target_fidelity or depth >= 8:
                break

        local_tree = best_tree

    elif pruning_method == "max_depth":
        # --- حالت استاتیک: همان روش قبلی ---
        local_tree = DecisionTreeClassifier(max_depth=pruning_param, random_state=42)
        local_tree.fit(local_data, local_preds)

    elif pruning_method == "ccp_alpha":
        full_tree = DecisionTreeClassifier(random_state=42)
        full_tree.fit(local_data, local_preds)
        path = full_tree.cost_complexity_pruning_path(local_data, local_preds)
        ccp_alphas = path.ccp_alphas

        best_alpha = ccp_alphas[-1]
        for ccp_alpha in ccp_alphas:
            pruned_tree = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
            pruned_tree.fit(local_data, local_preds)
            if pruned_tree.get_n_leaves() <= pruning_param:
                best_alpha = ccp_alpha
                break

        local_tree = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha)
        local_tree.fit(local_data, local_preds)

    else:
        raise ValueError(f"Unknown pruning method: {pruning_method}")

    fidelity = compute_local_fidelity(ensemble, local_tree, local_data)

    # 5) Complexity
    complexity = {
        "depth": local_tree.get_depth(),
        "leaves": local_tree.get_n_leaves(),
    }

    print(f"\n=== Local XAI Metrics ===")
    print(f"Method:       {pruning_method}")
    if pruning_method == "dynamic":
        print(f"Selected Depth:{complexity['depth']} (Auto-adjusted)")
    else:
        print(
            f"Fixed Depth:   {pruning_param if pruning_method == 'max_depth' else 'N/A'}"
        )
    print(f"Fidelity:     {fidelity:.4f}")
    print(f"Depth:        {complexity['depth']}")
    print(f"Leaves:       {complexity['leaves']}")
    print("=========================\n")

    return local_tree, local_data, fidelity, complexity


def compute_local_fidelity(ensemble, local_tree, local_data):
    probas = np.array([m.predict_proba(local_data) for m in ensemble])
    avg_proba = np.mean(probas, axis=0)
    ensemble_preds = np.argmax(avg_proba, axis=1)

    surrogate_preds = local_tree.predict(local_data)

    return accuracy_score(ensemble_preds, surrogate_preds)


# --- XAI Novelty Addition: Grad-CAM Implementation ---
def generate_grad_cam(model, target_layer_name, input_tensor, class_index):
    model.eval()

    features = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal features
        features = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    target_layer = dict([*model.named_modules()])[target_layer_name]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    output = model(input_tensor)

    model.zero_grad()
    class_score = output[0, class_index]
    class_score.backward(retain_graph=True)

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    for i in range(features.shape[1]):
        features[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(features, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap) + 1e-8  # Avoid division by zero

    forward_handle.remove()
    backward_handle.remove()

    return heatmap.detach().cpu().numpy()


# --- NEW XAI Plotters: Handles both Image and Numerical Data ---
def save_clear_original_image(original_img, sample_index, true_class_name):
    ensure_dir(DEFAULT_EXPLANATION_IMAGE_DIR)
    filename = os.path.join(
        DEFAULT_EXPLANATION_IMAGE_DIR, f"original_sample_{sample_index}.png"
    )

    scale_factor = 15
    if len(original_img.shape) == 2:  # Grayscale
        h, w = original_img.shape
        img_resized = cv2.resize(
            original_img,
            (w * scale_factor, h * scale_factor),
            interpolation=cv2.INTER_NEAREST,
        )
        cmap = "gray"
    else:  # Color
        h, w, c = original_img.shape
        img_resized = cv2.resize(
            original_img,
            (w * scale_factor, h * scale_factor),
            interpolation=cv2.INTER_NEAREST,
        )
        cmap = None

    plt.figure(figsize=(8, 8))
    plt.imshow(img_resized.squeeze(), cmap=cmap)
    plt.title(
        f"Original Image (Sample #{sample_index})\nTrue Class: {true_class_name}",
        fontsize=16,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"[io] Saved clear original image to: {filename}")
    return filename


def save_explanation_plot(
    original_img, heatmap, sample_index, predicted_class_name, true_class_name
):
    ensure_dir(DEFAULT_EXPLANATION_IMAGE_DIR)
    filename = os.path.join(
        DEFAULT_EXPLANATION_IMAGE_DIR, f"explanation_sample_{sample_index}.png"
    )

    scale_factor = 15
    if len(original_img.shape) == 2:  # Grayscale
        h, w = original_img.shape
        img_resized = cv2.resize(
            original_img,
            (w * scale_factor, h * scale_factor),
            interpolation=cv2.INTER_NEAREST,
        )
        cmap = "gray"
    else:  # Color
        h, w, c = original_img.shape
        img_resized = cv2.resize(
            original_img,
            (w * scale_factor, h * scale_factor),
            interpolation=cv2.INTER_NEAREST,
        )
        cmap = None

    heatmap_resized = cv2.resize(heatmap, (img_resized.shape[1], img_resized.shape[0]))

    plt.figure(figsize=(8, 8))
    plt.imshow(img_resized.squeeze(), cmap=cmap)
    plt.imshow(heatmap_resized, cmap="jet", alpha=0.3)

    threshold = np.percentile(heatmap_resized, 90)
    if threshold > 0.1:
        plt.contour(heatmap_resized, levels=[threshold], colors="lime", linewidths=2.5)

    plt.title(
        f"Explanation for Sample #{sample_index}\n"
        f"True: {true_class_name} | Predicted: {predicted_class_name}",
        fontsize=16,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"[io] Saved explanation image to: {filename}")
    return filename


def save_tree_explanation_plot(local_tree, feature_names, class_names, sample_index):
    ensure_dir(DEFAULT_EXPLANATION_IMAGE_DIR)
    filename = os.path.join(
        DEFAULT_EXPLANATION_IMAGE_DIR, f"tree_explanation_sample_{sample_index}.png"
    )

    plt.figure(figsize=(25, 12))
    plot_tree(
        local_tree,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=12,
        impurity=False,
        proportion=True,
    )
    plt.title(f"Local Surrogate Decision Tree for Sample #{sample_index}", fontsize=18)
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"[io] Saved tree explanation image to: {filename}")
    return filename


# --- NEW PLOTTER FOR NUMERICAL DATA ---
def save_numerical_explanation_plot(
    sample_data,
    local_tree,
    feature_names,
    sample_index,
    predicted_class_name,
    true_class_name,
):
    ensure_dir(DEFAULT_EXPLANATION_IMAGE_DIR)
    filename = os.path.join(
        DEFAULT_EXPLANATION_IMAGE_DIR,
        f"numerical_explanation_sample_{sample_index}.png",
    )

    most_imp_feature_idx = local_tree.tree_.feature[0]
    most_imp_feature_name = feature_names[most_imp_feature_idx]

    top_n = 10
    sorted_indices = np.argsort(np.abs(sample_data.flatten()))[-top_n:]
    top_feature_names = [feature_names[i] for i in sorted_indices]
    top_feature_values = [sample_data.flatten()[i] for i in sorted_indices]

    colors = [
        "green" if name == most_imp_feature_name else "skyblue"
        for name in top_feature_names
    ]

    plt.figure(figsize=(12, 8))
    bars = plt.barh(top_feature_names, top_feature_values, color=colors)
    plt.xlabel("Feature Value")
    plt.title(
        f"Top Feature Values for Sample #{sample_index}\n"
        f"True: {true_class_name} | Predicted: {predicted_class_name}",
        fontsize=16,
    )
    plt.gca().invert_yaxis()

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + width * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}",
            ha="left",
            va="center",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"[io] Saved numerical explanation plot to: {filename}")
    return filename


def generate_explanation_report(
    sample_index,
    true_class_name,
    predicted_class_name,
    confidence,
    local_tree,
    feature_names,
    class_names,
    important_feature_name,
):
    ensure_dir(DEFAULT_EXPLANATION_IMAGE_DIR)
    report_filename = os.path.join(
        DEFAULT_EXPLANATION_IMAGE_DIR, f"report_sample_{sample_index}.md"
    )

    tree_rules = export_text(
        local_tree, feature_names=feature_names, show_weights=False
    )

    report_content = f"""# AI Explanation Report for Sample #{sample_index}

## 1. Final Prediction

The model has classified this sample as **{predicted_class_name}** with a confidence of **{confidence:.4f}**.
The true label for this sample is **{true_class_name}**.

## 2. How Model Made This Decision (Local Explanation)

To understand *why* the model made this prediction, we trained a simple decision tree to mimic the complex model's behavior for this specific sample and its immediate neighbors.

The most influential factor in the decision was the value of **feature `{important_feature_name}`**.

### Decision Path Breakdown

The path taken for this specific sample leads to the final prediction of **{predicted_class_name}**.

## 3. Summary

In summary, the model's prediction is based on a specific set of feature values, with `{important_feature_name}` being the most critical one. The decision tree above provides a step-by-step logical rule set that led to the final classification.

---
*Generated by the AI Explanation System.*
"""
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[io] Saved explanation report to: {report_filename}")
    return report_filename


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
            roc = roc_auc_score(y_test, avg, multi_class="ovr")

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        print(f"[Fold {fold + 1}] Acc={acc:.4f} ROC={roc:.4f} F1={f1:.4f}")
        all_acc.append(acc)
        all_roc.append(roc)
        all_f1.append(f1)

    print("\n===== CV Summary =====")
    print(
        f"Mean Acc: {np.mean(all_acc):.4f}, Mean ROC: {np.mean(all_roc):.4f}, Mean F1: {np.mean(all_f1):.4f}"
    )


# ------------------------
# Refactored Main Logic
# ------------------------
def run_evaluation(models, X, y, folds):
    print("\n[mode] Running Evaluation...")
    stratified_cv(models, X, y, n_splits=folds)


def run_training(models, X, y, dataset_name, model_path, feature_names):
    print("\n[mode] Running Training...")
    for m in models:
        m.fit(X, y)
    print("[training] Ensemble training complete.")

    explanation_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    probas = np.array([model.predict_proba(X) for model in models])
    avg_proba = np.mean(probas, axis=0)
    ensemble_preds = np.argmax(avg_proba, axis=1)
    explanation_tree.fit(X, ensemble_preds)
    print("[xai] Default surrogate tree (depth=3) trained for saving.")

    ensure_dir(os.path.dirname(model_path))
    joblib.dump(
        {
            "ensemble": models,
            "pca": None,
            "class_names": None,
            "feature_names": feature_names,
            "explanation_tree": explanation_tree,
        },
        model_path,
        compress=3,
    )
    print(f"\n[io] Saved ensemble and feature_names -> {model_path}")


def run_prediction(
    model_file, sample_index, X_raw, y, class_names, feature_names, is_image_data, args
):
    print("\n[mode] Running Prediction and Explanation...")
    if not os.path.exists(model_file):
        print(
            f"[error] Model file not found at {model_file}. Please train the model first using --mode train."
        )
        return

    print(f"[io] Loading model from {model_file}...")
    data = joblib.load(model_file)
    ensemble = data["ensemble"]
    pca_loaded = data["pca"]
    scaler_loaded = data.get("scaler")  # <-- LOAD SCALER
    feature_names_loaded = data["feature_names"]

    if not (0 <= sample_index < X_raw.shape[0]):
        print(
            f"[error] Sample index {sample_index} is out of bounds for dataset with {X_raw.shape[0]} samples."
        )
        return

    sample_raw = X_raw[sample_index]
    true_label = y[sample_index]
    true_class_name = class_names[true_label]

    # --- Transform the single sample for prediction ---
    sample_feat = (
        extract_features_resnet(
            np.expand_dims(sample_raw, axis=0), args.dataset, use_gpu=args.use_gpu
        )
        if is_image_data
        else np.expand_dims(sample_raw, axis=0)
    )

    # Apply scaling and PCA if they exist
    if scaler_loaded:
        sample_feat = scaler_loaded.transform(sample_feat)
    if pca_loaded:
        sample_final = pca_loaded.transform(sample_feat)
    else:
        sample_final = sample_feat

    probas = np.array([model.predict_proba(sample_final) for model in ensemble])
    avg_proba = np.mean(probas, axis=0)
    predicted_class_idx = np.argmax(avg_proba, axis=1)[0]
    predicted_class_name = class_names[predicted_class_idx]
    confidence = avg_proba[0, predicted_class_idx]

    print("\n--- Prediction Result (Console) ---")
    print(f"Sample Index:    {sample_index}")
    print(f"True Label:      {true_class_name} (index {true_label})")
    print(f"Predicted Label: {predicted_class_name} (index {predicted_class_idx})")
    print(f"Confidence:      {confidence:.4f}")

    print("\n[novel] Generating Explanation...")
    local_tree, local_data, fidelity, complexity = train_local_surrogate(
        ensemble,
        sample_final,
        noise_std=args.noise_std,
        pruning_method=args.pruning_method,
        pruning_param=args.pruning_param,
    )

    print(f"[analysis] Local Fidelity = {fidelity:.4f}")
    print(f"[analysis] Tree Complexity = {complexity}")

    important_feature_index = local_tree.tree_.feature[0]
    important_feature_name = feature_names_loaded[important_feature_index]
    print(f"  [novel] Most important feature identified: {important_feature_name}")

    if is_image_data:
        print(
            "  [novel] Step 2: Generating Grad-CAM heatmap for the predicted class..."
        )
        device = torch.device(
            "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
        )
        resnet_model = torchvision_models.resnet18(
            weights=torchvision_models.ResNet18_Weights.DEFAULT
        )
        resnet_model.fc = nn.Identity()
        resnet_model = resnet_model.to(device)
        resnet_model.eval()

        if len(sample_raw.shape) == 2:  # Grayscale
            sample_rgb = np.repeat(np.expand_dims(sample_raw, axis=-1), 3, axis=-1)
        else:  # Color
            sample_rgb = sample_raw

        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        input_tensor = transform(sample_rgb).unsqueeze(0).to(device)
        heatmap = generate_grad_cam(
            resnet_model, "layer4", input_tensor, predicted_class_idx
        )

        original_path = save_clear_original_image(
            sample_raw, sample_index, true_class_name
        )
        explanation_path = save_explanation_plot(
            sample_raw, heatmap, sample_index, predicted_class_name, true_class_name
        )

        print("\n--- Textual Explanation (from LOCAL decision tree) ---")
        tree_rules = export_text(
            local_tree, feature_names=feature_names_loaded, show_weights=False
        )
        print(tree_rules)
        print(
            "\nNote: The green contour in the explanation image outlines the specific region that most influenced the model's decision."
        )

        open_file_in_viewer(original_path)
        open_file_in_viewer(explanation_path)
    else:
        numerical_path = save_numerical_explanation_plot(
            sample_final,
            local_tree,
            feature_names_loaded,
            sample_index,
            predicted_class_name,
            true_class_name,
        )
        print("\n--- Textual Explanation (from LOCAL decision tree) ---")
        tree_rules = export_text(
            local_tree, feature_names=feature_names_loaded, show_weights=False
        )
        print(tree_rules)
        print(
            "\nNote: The green bar in the plot highlights the most influential feature for this specific sample's prediction."
        )
        open_file_in_viewer(numerical_path)

    tree_path = save_tree_explanation_plot(
        local_tree, feature_names_loaded, class_names, sample_index
    )
    report_path = generate_explanation_report(
        sample_index,
        true_class_name,
        predicted_class_name,
        confidence,
        local_tree,
        feature_names_loaded,
        class_names,
        important_feature_name,
    )
    open_file_in_viewer(tree_path)
    open_file_in_viewer(report_path)


def run_batch_fidelity(models, X, y, feature_names, args):
    """
    New mode to evaluate fidelity across the entire dataset.
    """
    print("\n[mode] Running Batch Fidelity Analysis...")

    # 1. Train models on the provided X (which should be X_final)
    print("[training] Training ensemble on the entire dataset for analysis...")
    for m in models:
        m.fit(X, y)
    print("[training] Training complete.")

    print(f"[analysis] Starting fidelity check with noise_std={args.noise_std}...")

    fidelities = []
    depths = []

    # 2. Iterate over all samples
    # Using tqdm for a progress bar
    for i in tqdm(range(X.shape[0]), desc="Calculating Fidelity"):
        sample = X[i].reshape(1, -1)

        # Train local surrogate
        # We pass noise_std from args (default 0.05, but user might pass 1.0)
        local_tree, _, fidelity, complexity = train_local_surrogate(
            ensemble=models,
            sample=sample,
            n_samples=500,
            noise_std=args.noise_std,
            pruning_method=args.pruning_method,
            pruning_param=args.pruning_param,
        )

        fidelities.append(fidelity)
        depths.append(complexity["depth"])

    # 3. Calculate Average
    avg_fidelity = np.mean(fidelities)
    avg_depth = np.mean(depths)

    print("\n" + "=" * 35)
    print(f" BATCH ANALYSIS RESULTS")
    print("=" * 35)
    print(f"Dataset:       {args.dataset}")
    print(f"Total Samples: {X.shape[0]}")
    print(f"Pruning Method: {args.pruning_method}")
    print(f"Pruning Param:  {args.pruning_param}")
    print(f"Noise Std:     {args.noise_std}")
    print("-" * 35)
    print(f"AVG FIDELITY: {avg_fidelity:.4f} ({avg_fidelity * 100:.2f}%)")
    print(f"AVG DEPTH:    {avg_depth:.2f}")
    print("=" * 35 + "\n")


# ------------------------
# Main
# ------------------------
def main():
    parser = argparse.ArgumentParser(description="Ensemble model with XAI explanation.")
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="Dataset name (e.g., pathmnist, tissuemnist, breast_cancer, iris, wine).",
    )
    parser.add_argument(
        "--mode",
        choices=["evaluate", "train", "predict", "analyze_fidelity"],
        default="evaluate",
        help="Run mode: 'evaluate' for CV, 'train' to save model, 'predict' to explain a sample, 'analyze_fidelity' to check average fidelity.",
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
        help="Force re-computation of ResNet features, ignoring the cache.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index of sample to predict and explain (used with --mode predict).",
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
        choices=["max_depth", "ccp_alpha", "dynamic"],  # <-- ADDED DYNAMIC
        help="Method for pruning the local explanation tree. 'dynamic' adjusts depth automatically for max fidelity.",
    )
    parser.add_argument(
        "--pruning-param",
        type=int,
        default=3,
        help="Parameter for pruning method (max_depth value or max_leaves for ccp_alpha or start depth for dynamic).",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.05,
        help="Standard deviation of noise to generate local data for surrogate tree training.",
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

    # Ensure feature names exist for the feature data
    if feature_names_loaded is None:
        feature_names_loaded = [f"f{i}" for i in range(X_feat.shape[1])]

    # --- 3. Apply Scaling, PCA and Determine Final Features & Names ---
    scaler = None
    pca = None
    X_processed = X_feat

    # --- FIX: Apply StandardScaler for numerical data ---
    if not is_image_data:
        print("[preprocessing] Applying StandardScaler to numerical features...")
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X_feat)
        print("[preprocessing] Scaling complete.")

    # --- FIX: Auto-disable PCA for small datasets (Iris, Wine) for interpretability ---
    if use_pca and not is_image_data and not args.pca:
        if args.dataset in ["iris", "wine"] or X_processed.shape[1] < 20:
            print(
                f"[preprocessing] Dataset '{args.dataset}' is low-dimensional. "
                "Disabling PCA to preserve feature interpretability for XAI."
            )
            use_pca = False
    # --------------------------------------------------------------------------------

    if use_pca and len(X_processed.shape) == 2:
        n_components = min(DEFAULT_PCA_COMPONENTS, X_processed.shape[1])
        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
        X_final = pca.fit_transform(X_processed)
        feature_names_final = [f"PC{i + 1}" for i in range(X_final.shape[1])]
        print(
            f"[preprocessing] Applied PCA, reducing features from {X_processed.shape[1]} to {X_final.shape[1]}."
        )
    else:
        X_final = X_processed
        feature_names_final = (
            feature_names_loaded
            if scaler is None
            else [f"scaled_{name}" for name in feature_names_loaded]
        )
        print("[preprocessing] PCA is not used.")

    n_classes = len(np.unique(y))
    ensemble_models = create_ensemble(n_classes, use_gpu=use_gpu)

    # --- 4. Execute Mode ---
    model_path = args.model_path or os.path.join(
        DEFAULT_MODEL_PATH, f"ensemble_{args.dataset}.pkl"
    )

    if args.mode == "evaluate":
        run_evaluation(ensemble_models, X_final, y, args.folds)
    elif args.mode == "train":
        # Save pre-processing objects (scaler, pca) and feature names for later use
        preproc_data = {
            "scaler": scaler,
            "pca": pca,
            "feature_names": feature_names_final,
        }
        joblib.dump(preproc_data, model_path.replace(".pkl", "_preproc.pkl"))
        run_training(
            ensemble_models, X_final, y, args.dataset, model_path, feature_names_final
        )
    elif args.mode == "predict":
        # Load pre-processing objects
        preproc_path = model_path.replace(".pkl", "_preproc.pkl")
        if not os.path.exists(preproc_path):
            print(
                f"[error] Preprocessing file not found at {preproc_path}. Cannot ensure consistency."
            )
            return
        preproc_data = joblib.load(preproc_path)

        # Reconstruct the model dictionary to pass to the prediction function
        model_data = joblib.load(model_path)
        model_data.update(preproc_data)  # <-- UPDATE MODEL DICT
        joblib.dump(model_data, model_path)  # Save the combined model for simplicity

        run_prediction(
            model_path,
            args.sample_index,
            X_raw,
            y,
            class_names_loaded,
            feature_names_loaded,
            is_image_data,
            args,
        )
    elif args.mode == "analyze_fidelity":
        # This mode calculates fidelity on the fly for all samples
        run_batch_fidelity(ensemble_models, X_final, y, feature_names_final, args)


if __name__ == "__main__":
    main()
