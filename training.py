import os, cv2, glob, numpy as np, pickle, json, warnings
warnings.filterwarnings("ignore")

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix)

# ── Also train classic models for comparison ──────────────────────────────────
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier

import time

DATASET_PATH = "Dataset"
IMG_SIZE     = 64
label_map    = {"full_forg": 0, "full_org": 1}

# ── Load images ───────────────────────────────────────────────────────────────
X, y = [], []
for folder in os.listdir(DATASET_PATH):
    if folder not in label_map:
        continue
    for img_path in glob.glob(os.path.join(DATASET_PATH, folder, "*")):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img)
        y.append(label_map[folder])

X = np.array(X, dtype="float32") / 255.0
y = np.array(y)
X, y = shuffle(X, y, random_state=42)

total    = len(X)
forged   = int(np.sum(y == 0))
original = int(np.sum(y == 1))
print(f"\nDataset: {total} images | Original: {original} | Forged: {forged}")

# ── Train / test split ────────────────────────────────────────────────────────
test_size = max(0.2, 2 / total) if total >= 10 else 0.5
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y)

print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

# ═════════════════════════════════════════════════════════════════════════════
#  CNN  (TensorFlow / Keras)
# ═════════════════════════════════════════════════════════════════════════════
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping

    print("Building CNN model...")

    y_train_cat = to_categorical(y_train, 2)
    y_test_cat  = to_categorical(y_test,  2)

    cnn = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.4),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(2, activation="softmax"),
    ])

    cnn.compile(optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"])

    cnn.summary()

    es = EarlyStopping(monitor="val_loss", patience=10,
                       restore_best_weights=True, verbose=0)

    epochs     = 50 if total < 20 else 100
    batch_size = max(2, min(8, total // 4))

    start = time.time()
    history_cnn = cnn.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1,
    )
    cnn_time = round(time.time() - start, 2)

    y_pred_cnn  = np.argmax(cnn.predict(X_test, verbose=0), axis=1)
    cnn_acc     = round(accuracy_score(y_test, y_pred_cnn)                        * 100, 2)
    cnn_prec    = round(precision_score(y_test, y_pred_cnn, zero_division=0, average="macro") * 100, 2)
    cnn_rec     = round(recall_score(y_test, y_pred_cnn,    zero_division=0, average="macro") * 100, 2)
    cnn_f1      = round(f1_score(y_test, y_pred_cnn,        zero_division=0, average="macro") * 100, 2)
    cm          = confusion_matrix(y_test, y_pred_cnn)
    TN,FP,FN,TP = cm.ravel() if cm.size==4 else (0,0,0,0)
    cnn_sens    = round(TP/(TP+FN)*100,2) if (TP+FN)>0 else 0.0
    cnn_spec    = round(TN/(TN+FP)*100,2) if (TN+FP)>0 else 0.0

    # Save CNN
    cnn.save("cnn_model.keras")
    print(f"\n✅ CNN trained → Accuracy: {cnn_acc}%  |  Time: {cnn_time}s")
    cnn_available = True

except Exception as e:
    print(f"\n⚠️  TensorFlow not available: {e}")
    print("    Install with: pip install tensorflow")
    print("    Falling back to classic models only.\n")
    cnn_available = False
    cnn_acc = cnn_prec = cnn_rec = cnn_f1 = cnn_sens = cnn_spec = 0.0
    TN = FP = FN = TP = 0
    cnn_time = 0

# ═════════════════════════════════════════════════════════════════════════════
#  Classic ML models (flat pixel features)
# ═════════════════════════════════════════════════════════════════════════════
X_flat_train = X_train.reshape(len(X_train), -1)
X_flat_test  = X_test.reshape(len(X_test),  -1)

# Scale to 0-255 int for MultinomialNB
X_mnb_train  = (X_flat_train * 255).astype(int)
X_mnb_test   = (X_flat_test  * 255).astype(int)

classic_models = {
    "Multinomial NB":      (MultinomialNB(),                                    X_mnb_train, X_mnb_test),
    "Bernoulli NB":        (BernoulliNB(),                                      X_flat_train, X_flat_test),
    "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42), X_flat_train, X_flat_test),
    "SGD Classifier":      (SGDClassifier(random_state=42, max_iter=1000),      X_flat_train, X_flat_test),
    "Random Forest":       (RandomForestClassifier(n_estimators=100, random_state=42), X_flat_train, X_flat_test),
}
short_names = {
    "Multinomial NB":"MNBC","Bernoulli NB":"BNBC",
    "Logistic Regression":"LRC","SGD Classifier":"SGDC","Random Forest":"RFC",
}

results     = {}
best_name   = None
best_acc    = -1
best_model  = None

print(f"\n{'='*65}")
print(f"{'Model':<25} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Time':>8}")
print(f"{'='*65}")

for name, (model, Xtr, Xte) in classic_models.items():
    try:
        t0 = time.time()
        model.fit(Xtr, y_train)
        elapsed = round(time.time()-t0, 3)
        yp  = model.predict(Xte)
        acc = round(accuracy_score(y_test, yp)*100,2)
        pr  = round(precision_score(y_test,yp,zero_division=0,average="macro")*100,2)
        re  = round(recall_score(y_test,yp,zero_division=0,average="macro")*100,2)
        f1v = round(f1_score(y_test,yp,zero_division=0,average="macro")*100,2)
        cm2         = confusion_matrix(y_test, yp)
        t2,f2,f3,t3 = cm2.ravel() if cm2.size==4 else (0,0,0,0)
        sens = round(t3/(t3+f3)*100,2) if (t3+f3)>0 else 0.0
        spec = round(t2/(t2+f2)*100,2) if (t2+f2)>0 else 0.0
        results[name] = dict(short=short_names[name],accuracy=acc,precision=pr,
            recall=re,f1=f1v,sensitivity=sens,specificity=spec,train_time=elapsed,
            cm=dict(TN=int(t2),FP=int(f2),FN=int(f3),TP=int(t3)))
        print(f"{name:<25} {acc:>6.1f}% {pr:>6.1f}% {re:>6.1f}% {f1v:>6.1f}% {elapsed:>7.3f}s")
        if acc > best_acc:
            best_acc = acc; best_name = name; best_model = model
    except Exception as e:
        print(f"{name:<25} ERROR: {e}")

# Add CNN to results table if available
if cnn_available:
    results["CNN"] = dict(
        short="CNN", accuracy=cnn_acc, precision=cnn_prec,
        recall=cnn_rec, f1=cnn_f1, sensitivity=cnn_sens, specificity=cnn_spec,
        train_time=cnn_time,
        cm=dict(TN=int(TN),FP=int(FP),FN=int(FN),TP=int(TP))
    )
    print(f"{'CNN (Deep Learning)':<25} {cnn_acc:>6.1f}% {cnn_prec:>6.1f}% {cnn_rec:>6.1f}% {cnn_f1:>6.1f}% {cnn_time:>7.2f}s")
    if cnn_acc >= best_acc:
        best_acc  = cnn_acc
        best_name = "CNN"

print(f"{'='*65}")
print(f"\n🏆 Best model: {best_name} ({best_acc}%)")

# ── Pick best result stats ────────────────────────────────────────────────────
br = results[best_name]

metrics = {
    "model_type":     "CNN" if best_name=="CNN" else "Classic ML",
    "best_model":     best_name,
    "best_short":     br["short"],
    "accuracy":       br["accuracy"],
    "precision":      br["precision"],
    "recall":         br["recall"],
    "f1":             br["f1"],
    "sensitivity":    br["sensitivity"],
    "specificity":    br["specificity"],
    "cm":             br["cm"],
    "total_samples":  total,
    "train_samples":  len(X_train),
    "test_samples":   len(X_test),
    "forged_count":   forged,
    "original_count": original,
    "cnn_available":  cnn_available,
    "all_models":     results,
}

# ── Save best classic model as pkl (used for GradCAM fallback + API) ──────────
if best_name != "CNN":
    pickle.dump(best_model, open("model.pkl","wb"))
else:
    # Save best classic model too as fallback
    if best_model:
        pickle.dump(best_model, open("model.pkl","wb"))

with open("metrics.json","w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n✅ cnn_model.keras  → saved" if cnn_available else "")
print(f"✅ model.pkl        → saved ({best_name if best_name!='CNN' else 'best classic'})")
print(f"✅ metrics.json     → saved")
print(f"\n{'='*65}")
print(f"  Add 100+ images per class for production-level accuracy.")
print(f"{'='*65}\n")