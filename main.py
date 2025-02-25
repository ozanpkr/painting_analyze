import argparse
import sys
import optuna
import torch
import torch.nn as nn
from torch import optim

from src.dataset import CustomDataset
from src import objective
from src import CustomResnet50
from src import Trainer
from src.AnalyzeCAM import AnalyzeCAM

custom_dataset = CustomDataset(root_dir="split_dataset")


def main():
    parser = argparse.ArgumentParser(description="Pass")
    parser.add_argument("--train", type=bool, default=False, help="is train")
    parser.add_argument("--optuna", type=bool, default=False, help="is hyper tuning")
    parser.add_argument("--analyze", type=bool, default=False, help="is analyze")
    args = parser.parse_args()
    if args.optuna:
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10)
        print(f"Best hyperparameters: {study.best_params}")
    if args.train:

        model = CustomResnet50(
            activation_fn=None,
            num_classes=custom_dataset.num_classes,
            verbose=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=model.parameters(), lr=0.001)
        exp_id = f"R50-Base-Exp"
        trainer = Trainer(
            model=model,
            train_loader=custom_dataset.train_loader,
            val_loader=custom_dataset.val_loader,
            criterion=criterion,
            optimizer=optimizer,
            exp_key=exp_id,
            save_best=True,
        )
        trainer.train(20)
        del trainer
        del model
        torch.cuda.empty_cache()

        sys.stdout.close()

    elif args.analyze:
        analyzer = AnalyzeCAM(model_path=None)
        analyzer.test_on_folder("./test")


if __name__ == "__main__":
    main()
