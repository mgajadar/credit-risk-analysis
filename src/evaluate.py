import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    confusion_matrix, classification_report
)

def _predictProba(model, X):
    proba = model.predict_proba(X)
    # class 1 prob
    return proba[:, 1]

def _saveFigure(figPath: Path):
    figPath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(figPath, dpi=200)
    plt.close()

def _plotRoc(yTrue, yScore, outPath: Path, title: str):
    fpr, tpr, _ = roc_curve(yTrue, yScore)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    _saveFigure(outPath)

def _plotPr(yTrue, yScore, outPath: Path, title: str):
    precision, recall, _ = precision_recall_curve(yTrue, yScore)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    _saveFigure(outPath)

def _plotConfusion(yTrue, yPred, outPath: Path, title: str):
    cm = confusion_matrix(yTrue, yPred)
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    _saveFigure(outPath)

def _bestThresholdByF1(yTrue, yScore):
    precision, recall, thresholds = precision_recall_curve(yTrue, yScore)
    # precision/recall are length n+1, thresholds length n
    precision = precision[:-1]
    recall = recall[:-1]
    f1 = 2 * (precision * recall) / np.clip(precision + recall, 1e-12, None)
    bestIdx = int(np.nanargmax(f1)) if len(f1) else 0
    return float(thresholds[bestIdx]) if len(thresholds) else 0.5, float(f1[bestIdx]) if len(f1) else 0.0

def evaluateAndReport(trained, split, cfg):
    models = {
        "logReg": trained.logReg,
        "randomForest": trained.randomForest
    }

    metricsOut = {}
    for name, model in models.items():
        valScore = _predictProba(model, split.XVal)
        bestThr, bestF1 = _bestThresholdByF1(split.yVal.values, valScore)

        testScore = _predictProba(model, split.XTest)
        testPred = (testScore >= bestThr).astype(int)

        rocAuc = roc_auc_score(split.yTest, testScore)
        prAuc = average_precision_score(split.yTest, testScore)
        report = classification_report(split.yTest, testPred, output_dict=True, zero_division=0)

        metricsOut[name] = {
            "threshold": bestThr,
            "valBestF1": bestF1,
            "testRocAuc": float(rocAuc),
            "testPrAuc": float(prAuc),
            "classificationReport": report
        }

        # Plots
        _plotRoc(split.yTest.values, testScore, cfg.figuresDir / f"{name}_roc.png", f"{name} ROC Curve")
        _plotPr(split.yTest.values, testScore, cfg.figuresDir / f"{name}_pr.png", f"{name} Precision-Recall Curve")
        _plotConfusion(split.yTest.values, testPred, cfg.figuresDir / f"{name}_cm.png", f"{name} Confusion Matrix (thr={bestThr:.2f})")

    # Save metrics
    outPath = cfg.metricsDir / "metrics.json"
    with open(outPath, "w", encoding="utf-8") as f:
        json.dump(metricsOut, f, indent=2)

    print(f"Saved metrics to: {outPath}")
    print(f"Saved figures to: {cfg.figuresDir}")
