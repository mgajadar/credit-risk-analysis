from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    projectRoot: Path
    dataDir: Path
    rawDir: Path
    processedDir: Path
    reportsDir: Path
    figuresDir: Path
    metricsDir: Path
    modelsDir: Path

    randomSeed: int = 42
    testSize: float = 0.20
    valSize: float = 0.20  # split train into train/val
    openmlDatasetName: str = "credit-g"

    # Threshold tuning 
    defaultThreshold: float = 0.50

def getConfig() -> Config:
    projectRoot = Path(__file__).resolve().parents[1]
    dataDir = projectRoot / "data"
    rawDir = dataDir / "raw"
    processedDir = dataDir / "processed"

    reportsDir = projectRoot / "reports"
    figuresDir = reportsDir / "figures"
    metricsDir = reportsDir / "metrics"

    modelsDir = projectRoot / "models"

    # Ensure dirs exist
    for p in [rawDir, processedDir, figuresDir, metricsDir, modelsDir]:
        p.mkdir(parents=True, exist_ok=True)

    return Config(
        projectRoot=projectRoot,
        dataDir=dataDir,
        rawDir=rawDir,
        processedDir=processedDir,
        reportsDir=reportsDir,
        figuresDir=figuresDir,
        metricsDir=metricsDir,
        modelsDir=modelsDir,
    )
