from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

@dataclass
class TrainedModels:
    logReg: Pipeline
    randomForest: Pipeline

def trainModels(split, cfg) -> TrainedModels:
    # Interpretable baseline
    logRegModel = Pipeline(steps=[
        ("preprocessor", split.preprocessor),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs"
        ))
    ])

    # Stronger benchmark
    rfModel = Pipeline(steps=[
        ("preprocessor", split.preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=400,
            random_state=cfg.randomSeed,
            class_weight="balanced_subsample",
            min_samples_leaf=2
        ))
    ])

    logRegModel.fit(split.XTrain, split.yTrain)
    rfModel.fit(split.XTrain, split.yTrain)

    # Save artifacts
    (cfg.modelsDir).mkdir(parents=True, exist_ok=True)
    joblib.dump(logRegModel, cfg.modelsDir / "logReg.joblib")
    joblib.dump(rfModel, cfg.modelsDir / "randomForest.joblib")

    return TrainedModels(logReg=logRegModel, randomForest=rfModel)
