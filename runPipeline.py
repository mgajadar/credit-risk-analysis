from src.config import getConfig
from src.dataIngest import fetchCreditData
from src.preprocess import buildPreprocessorAndSplit
from src.train import trainModels
from src.evaluate import evaluateAndReport

def main():
    cfg = getConfig()

    df = fetchCreditData(cfg)
    split = buildPreprocessorAndSplit(df, cfg)

    trained = trainModels(split, cfg)
    evaluateAndReport(trained, split, cfg)

if __name__ == "__main__":
    main()
