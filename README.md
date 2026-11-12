# mimic-tree

> **Soft Voting Ensemble + ResNet18 Feature Extraction + XAI Surrogate Decision Tree**\
> پروژه‌ای برای طبقه‌بندی تصاویر پزشکی (MedMNIST) و داده‌های عددی با قابلیت توضیح‌پذیری هوش مصنوعی (XAI)

---

## فهرست مطالب

- [معرفی](#%D9%85%D8%B9%D8%B1%D9%81%DB%8C)
- [معماری کلی](#%D9%85%D8%B9%D9%85%D8%A7%D8%B1%DB%8C-%DA%A9%D9%84%DB%8C)
- [ساختار فایل‌ها](#%D8%B3%D8%A7%D8%AE%D8%AA%D8%A7%D8%B1-%D9%81%D8%A7%DB%8C%D9%84%E2%80%8C%D9%87%D8%A7)
- [نصب و پیش‌نیازها](#%D9%86%D8%B5%D8%A8-%D9%88-%D9%BE%DB%8C%D8%B4%E2%80%8C%D9%86%DB%8C%D8%A7%D8%B2%D9%87%D8%A7)
- [دیتاست‌های پشتیبانی‌شده](#%D8%AF%DB%8C%D8%AA%D8%A7%D8%B3%D8%AA%E2%80%8C%D9%87%D8%A7%DB%8C-%D9%BE%D8%B4%D8%AA%DB%8C%D8%A8%D8%A7%D9%86%DB%8C%E2%80%8C%D8%B4%D8%AF%D9%87)
- [آموزش مدل (Train)](#%D8%A2%D9%85%D9%88%D8%B2%D8%B4-%D9%85%D8%AF%D9%84-train)
- [اجرا و پیش‌بینی (Predict)](#%D8%A7%D8%AC%D8%B1%D8%A7-%D9%88-%D9%BE%DB%8C%D8%B4%E2%80%8C%D8%A8%DB%8C%D9%86%DB%8C-predict)
- [ارزیابی (Evaluate)](#%D8%A7%D8%B1%D8%B2%DB%8C%D8%A7%D8%A8%DB%8C-evaluate)
- [تحلیل فیدلیتی (Analyze Fidelity)](#%D8%AA%D8%AD%D9%84%DB%8C%D9%84-%D9%81%DB%8C%D8%AF%D9%84%DB%8C%D8%AA%DB%8C-analyze-fidelity)
- [مقایسه f_tree.py و ave_tree.py](#%D9%85%D9%82%D8%A7%DB%8C%D8%B3%D9%87-f_treepy-%D9%88-ave_treepy)
- [آرگومان‌های خط فرمان](#%D8%A2%D8%B1%DA%AF%D9%88%D9%85%D8%A7%D9%86%E2%80%8C%D9%87%D8%A7%DB%8C-%D8%AE%D8%B7-%D9%81%D8%B1%D9%85%D8%A7%D9%86)
- [خروجی‌ها](#%D8%AE%D8%B1%D9%88%D8%AC%DB%8C%E2%80%8C%D9%87%D8%A7)
- [نکات فنی](#%D9%86%DA%A9%D8%A7%D8%AA-%D9%81%D9%86%DB%8C)

---

## معرفی

**mimic-tree** یک سیستم طبقه‌بندی یادگیری ماشین است که ترکیبی از:

1. **استخراج ویژگی با ResNet18** (pretrained) برای داده‌های تصویری
2. **Ensemble چندگانه CatBoost** با رأی‌گیری نرم (Soft Voting)
3. **توضیح‌پذیری (XAI)** از طریق درخت جانشین محلی (Local Surrogate Tree) و Grad-CAM

را ارائه می‌دهد. هدف اصلی این است که یک مدل black-box قوی را با یک درخت تصمیم ساده «تقلید» کنیم تا بفهمیم چرا مدل آن پیش‌بینی را کرده است — از اینجاست که نام **mimic-tree** می‌آید.

---

## معماری کلی

```
Input Image / Numerical Data
        │
        ▼
┌────────────────────────────────────────┐
│  Feature Extraction (ResNet18)         │  ← فقط برای تصاویر
│  fc layer → Identity (512-dim output)  │
└────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────┐
│  StandardScaler (numerical only)       │
│  PCA (optional, 256 components)        │
└────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────┐
│  Soft Voting Ensemble (3× CatBoost)    │
│  seeds: 42 / 101 / 202                 │
│  avg(proba_1, proba_2, proba_3) → pred │
└────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────┐
│  XAI Explanation                       │
│  ┌─────────────────────────────────┐   │
│  │ Local Surrogate Decision Tree   │   │
│  │ (500 noisy neighbors)           │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │ Grad-CAM Heatmap (images only)  │   │
│  │ (layer4 of ResNet18)            │   │
│  └─────────────────────────────────┘   │
└────────────────────────────────────────┘
```

**جریان داده برای تصاویر:**

```
Image → ResNet18 → 512-dim features → (optional PCA → 256-dim) → 3× CatBoost ensemble → Soft Vote → Prediction
```

**جریان داده برای داده عددی:**

```
Raw features → StandardScaler → (optional PCA) → 3× CatBoost ensemble → Soft Vote → Prediction
```

---

## ساختار فایل‌ها

```
mimic-tree/
├── f_tree.py         # نسخه اصلی (3 حالت: evaluate / train / predict)
├── ave_tree.py       # نسخه پیشرفته (+ حالت analyze_fidelity + Dynamic Pruning)
├── compare.py        # مقایسه دو رویکرد روی یک دیتاست
├── get_photo.py      # ابزار کمکی برای نمایش تصاویر
└── req.txt           # وابستگی‌های پروژه
```

**کش‌های خودکار:**

```
.cache_datasets/      # دیتاست‌های دانلودشده (npy)
.cache_features/      # ویژگی‌های استخراج‌شده ResNet (npy)
trained_models/       # مدل‌های ذخیره‌شده (pkl/joblib)
explanation_images/   # تصاویر توضیح XAI
```

---

## نصب و پیش‌نیازها

**Python 3.8+** مورد نیاز است.

```bash
pip install -r req.txt
```

محتوای `req.txt`:

```
numpy
joblib
opencv-python
scikit-learn
catboost
torch
torchvision
medmnist
matplotlib
seaborn
tqdm
```

**برای استفاده از GPU (اختیاری):**

```bash
# به جای torch معمولی:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## دیتاست‌های پشتیبانی‌شده

### داده‌های تصویری پزشکی (MedMNIST)

| نام دیتاست | توضیح | کلاس‌ها |
| --- | --- | --- |
| `pathmnist` | بافت‌شناسی روده‌ی بزرگ | 9 |
| `tissuemnist` | میکروسکوپ بافت کلیه | 8 |
| `dermamnist` | ضایعات پوستی | 7 |
| `octmnist` | OCT شبکیه چشم | 4 |
| `pneumoniamnist` | رادیوگرافی ریه | 2 |
| و سایر دیتاست‌های MedMNIST... |  |  |

### داده‌های عددی (sklearn)

| نام دیتاست | توضیح | کلاس‌ها |
| --- | --- | --- |
| `breast_cancer` | سرطان پستان (30 ویژگی) | 2 |
| `iris` | گل زنبق (4 ویژگی) | 3 |
| `wine` | شراب (13 ویژگی) | 3 |

---

## آموزش مدل (Train)

### گام ۱: دانلود دیتاست و استخراج ویژگی

اولین بار که اجرا می‌کنید، ResNet18 به‌صورت خودکار دیتاست را دانلود و ویژگی‌ها را استخراج می‌کند. این فرایند یک‌بار انجام می‌شود و نتیجه cache می‌شود.

### گام ۲: آموزش و ذخیره مدل

**با `f_tree.py` (نسخه ساده):**

```bash
# آموزش روی pathmnist با CPU
python f_tree.py --dataset pathmnist --mode train

# آموزش روی pathmnist با GPU
python f_tree.py --dataset pathmnist --mode train --use-gpu

# آموزش روی داده عددی
python f_tree.py --dataset breast_cancer --mode train

# مسیر سفارشی برای ذخیره مدل
python f_tree.py --dataset pathmnist --mode train --model-path ./my_models/pathmnist.pkl
```

**با `ave_tree.py` (نسخه پیشرفته):**

```bash
python ave_tree.py --dataset pathmnist --mode train --use-gpu
```

**گزینه‌های مهم در Train:**

```bash
--pca           # اعمال PCA (پیش‌فرض: فعال اگر --no-pca نباشد)
--no-pca        # غیرفعال‌کردن PCA
--force-download          # دانلود مجدد دیتاست
--force-recompute-features # محاسبه مجدد ویژگی‌های ResNet
```

**چه اتفاقی می‌افتد؟**

1. دیتاست بارگذاری یا دانلود می‌شود
2. ویژگی‌های ResNet18 استخراج و cache می‌شوند
3. سه مدل CatBoost با seed های مختلف روی کل داده fit می‌شوند
4. یک درخت توضیحی پیش‌فرض (depth=3) برای کل داده آموزش می‌بیند
5. همه چیز در `./trained_models/{dataset}_model.pkl` ذخیره می‌شود

---

## اجرا و پیش‌بینی (Predict)

پس از train، می‌توانید برای یک نمونه‌ی خاص پیش‌بینی و توضیح دریافت کنید:

```bash
# پیش‌بینی و توضیح نمونه شماره 42
python f_tree.py --dataset pathmnist --mode predict --sample-index 42

# با مشخص‌کردن مسیر مدل
python f_tree.py --dataset pathmnist --mode predict --sample-index 42 \
    --model-path ./trained_models/pathmnist_model.pkl

# با GPU
python f_tree.py --dataset pathmnist --mode predict --sample-index 0 --use-gpu

# برای داده عددی
python f_tree.py --dataset breast_cancer --mode predict --sample-index 10
```

**تنظیمات درخت جانشین:**

```bash
# روش هرس: max_depth (پیش‌فرض)
python f_tree.py --dataset pathmnist --mode predict --sample-index 0 \
    --pruning-method max_depth --pruning-param 3

# روش هرس: ccp_alpha
python f_tree.py --dataset pathmnist --mode predict --sample-index 0 \
    --pruning-method ccp_alpha --pruning-param 5

# تنظیم نویز (بیشتر → درخت پیچیده‌تر)
python f_tree.py --dataset pathmnist --mode predict --sample-index 0 \
    --noise-std 0.1
```

**برای `ave_tree.py` با Dynamic Pruning:**

```bash
python ave_tree.py --dataset pathmnist --mode predict --sample-index 42 \
    --pruning-method dynamic --pruning-param 3
```

**خروجی predict:**

- پیش‌بینی و confidence در ترمینال
- تصویر اصلی (`original_sample_N.png`)
- هیت‌مپ Grad-CAM (`explanation_sample_N.png`) — فقط برای تصاویر
- درخت تصمیم بصری (`tree_explanation_sample_N.png`)
- نمودار اهمیت ویژگی (`numerical_explanation_sample_N.png`) — فقط برای داده عددی
- گزارش متنی Markdown (`report_sample_N.md`)

---

## ارزیابی (Evaluate)

ارزیابی با استفاده از **10-Fold Stratified Cross-Validation**:

```bash
# ارزیابی پایه
python f_tree.py --dataset pathmnist --mode evaluate

# تعداد فولد سفارشی
python f_tree.py --dataset pathmnist --mode evaluate --folds 5

# با GPU
python f_tree.py --dataset pathmnist --mode evaluate --use-gpu

# روی داده عددی
python f_tree.py --dataset breast_cancer --mode evaluate
```

**معیارهای ارزیابی:**

| معیار | توضیح |
| --- | --- |
| Accuracy | دقت کلی طبقه‌بندی |
| ROC AUC | برای binary: AUC معمولی / multiclass: OvR |
| F1-score | میانگین وزن‌دار F1 |

نمونه خروجی:

```
[Fold 1] Acc=0.9312  ROC=0.9847  F1=0.9298
[Fold 2] Acc=0.9401  ROC=0.9881  F1=0.9389
...
===== CV Summary =====
Mean Acc: 0.9356, Mean ROC: 0.9863, Mean F1: 0.9341
```

---

## تحلیل فیدلیتی (Analyze Fidelity)

> فقط در `ave_tree.py` موجود است.

این حالت فیدلیتی درخت جانشین را روی **تمام نمونه‌های دیتاست** محاسبه می‌کند تا بفهمیم درخت چقدر رفتار ensemble را تقلید می‌کند:

```bash
python ave_tree.py --dataset pathmnist --mode analyze_fidelity

# با dynamic pruning
python ave_tree.py --dataset breast_cancer --mode analyze_fidelity \
    --pruning-method dynamic --pruning-param 3

# با noise بالاتر
python ave_tree.py --dataset iris --mode analyze_fidelity --noise-std 0.5
```

نمونه خروجی:

```
===================================
 BATCH ANALYSIS RESULTS
===================================
Dataset: pathmnist
Total Samples: 107180
Pruning Method: dynamic
Pruning Param: 3
Noise Std: 0.05
-----------------------------------
AVG FIDELITY: 0.9743 (97.43%)
AVG DEPTH: 5.21
===================================
```

---

## مقایسه f_tree.py و ave_tree.py

| ویژگی | `f_tree.py` | `ave_tree.py` |
| --- | --- | --- |
| **حالت‌های اجرا** | evaluate, train, predict | evaluate, train, predict, **analyze_fidelity** |
| **روش‌های هرس** | max_depth, ccp_alpha | max_depth, ccp_alpha, **dynamic** |
| **Dynamic Pruning** | ✗ | ✓ (تنظیم خودکار عمق) |
| **هدف فیدلیتی (dynamic)** | — | ≥ 99% یا عمق ≤ 8 |
| **پیچیدگی** | ساده‌تر | کمی بیشتر |
| **Batch Fidelity Analysis** | ✗ | ✓ |
| **قابلیت‌های XAI** | یکسان | یکسان + گزارش depth auto |

**کِی کدام را انتخاب کنیم؟**

- **`f_tree.py`**: وقتی می‌خواهید عمق درخت توضیحی را خودتان کنترل کنید یا پروژه را ساده نگه دارید.
- **`ave_tree.py`**: وقتی می‌خواهید بالاترین فیدلیتی ممکن را داشته باشید، یا نیاز به آنالیز کلی فیدلیتی روی دیتاست دارید.

---

## آرگومان‌های خط فرمان

| آرگومان | پیش‌فرض | توضیح |
| --- | --- | --- |
| `--dataset` | الزامی | نام دیتاست (مثال: `pathmnist`, `breast_cancer`) |
| `--mode` | `evaluate` | حالت اجرا: `evaluate` / `train` / `predict` / `analyze_fidelity`* |
| `--folds` | `10` | تعداد فولد در Cross-Validation |
| `--use-gpu` | `False` | استفاده از GPU اگر موجود باشد |
| `--pca` | — | فعال‌کردن اجباری PCA |
| `--no-pca` | — | غیرفعال‌کردن PCA |
| `--sample-index` | `0` | ایندکس نمونه برای predict |
| `--model-path` | auto | مسیر ذخیره/بارگذاری مدل |
| `--pruning-method` | `max_depth` | روش هرس: `max_depth` / `ccp_alpha` / `dynamic`* |
| `--pruning-param` | `3` | پارامتر هرس (عمق یا تعداد برگ) |
| `--noise-std` | `0.05` | انحراف معیار نویز برای تولید داده محلی |
| `--force-download` | `False` | دانلود مجدد دیتاست |
| `--force-recompute-features` | `False` | محاسبه مجدد ویژگی‌های ResNet |

\*فقط در `ave_tree.py`

---

## خروجی‌ها

### فایل مدل (پس از train)

```
./trained_models/{dataset}_model.pkl
```

شامل: ensemble models, PCA, feature_names, explanation_tree

### تصاویر توضیح (پس از predict)

```
./explanation_images/
├── original_sample_{N}.png          # تصویر اصلی (برای تصاویر)
├── explanation_sample_{N}.png       # هیت‌مپ Grad-CAM (برای تصاویر)
├── tree_explanation_sample_{N}.png  # درخت تصمیم بصری
├── numerical_explanation_{N}.png    # نمودار ویژگی (برای داده عددی)
└── report_sample_{N}.md             # گزارش متنی
```

### کش‌ها

```
.cache_datasets/{name}_X.npy    # داده خام
.cache_datasets/{name}_y.npy    # برچسب‌ها
.cache_features/{name}_{H}x{W}_resnet_features.npy  # ویژگی‌های ResNet
```

---

## نکات فنی

### هایپرپارامترهای CatBoost

```python
iterations    = 1200
learning_rate = 0.03
depth         = 6
l2_leaf_reg   = 4
loss_function = "Logloss"  (binary) / "MultiClass"  (multiclass)
```

### Dynamic Pruning (فقط ave_tree.py)

الگوریتم از عمق `pruning_param` شروع می‌کند و تا عمق 8 جستجو می‌کند؛ به محض رسیدن به فیدلیتی ≥ 99% متوقف می‌شود:

```python
for depth in range(pruning_param, 9):
    tree.fit(local_data, local_preds)
    if fidelity >= 0.99 or depth >= 8:
        break
```

### کش‌کردن ویژگی‌ها

نام فایل کش شامل ابعاد تصویر است تا از تداخل بین دیتاست‌های مختلف جلوگیری شود:

```
{dataset_name}_{H}x{W}_resnet_features.npy
```

### رسیدگی به تصاویر grayscale

تصاویر grayscale به‌صورت خودکار به 3 کانال تبدیل می‌شوند تا با ResNet18 سازگار باشند.

---

## مثال کامل (End-to-End)

```bash
# ۱. نصب وابستگی‌ها
pip install -r req.txt

# ۲. آموزش مدل روی pathmnist
python ave_tree.py --dataset pathmnist --mode train --use-gpu

# ۳. ارزیابی با Cross-Validation
python ave_tree.py --dataset pathmnist --mode evaluate --use-gpu --folds 10

# ۴. توضیح نمونه شماره 100
python ave_tree.py --dataset pathmnist --mode predict \
    --sample-index 100 \
    --pruning-method dynamic \
    --pruning-param 3 \
    --use-gpu

# ۵. آنالیز کلی فیدلیتی
python ave_tree.py --dataset pathmnist --mode analyze_fidelity \
    --pruning-method dynamic \
    --pruning-param 3
```

---

## لایسنس

MIT
