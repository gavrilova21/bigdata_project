from argparse import ArgumentParser
import torch
import torch.onnx
import mlflow
import mlflow.pytorch
import mlflow.onnx
import torchvision.transforms as transforms
import torchvision
from server.model_utils import Trainer


mlflow.set_tracking_uri("http://localhost:5000")
client = mlflow.tracking.MlflowClient()

print("MLflow Version:", mlflow.version.VERSION)
print("MLflow Tracking URI:", mlflow.get_tracking_uri())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", dest="experiment_name", help="experiment_name", default="super_resolution")
    parser.add_argument("--model_path", dest="model_path", help="model path", default="./isr_best.pth")
    parser.add_argument("--run_origin", dest="run_origin", help="run_origin", default="none")
    parser.add_argument("--registered_model", dest="registered_model", help="Registered model name", default=None)
    parser.add_argument("--train", dest="train", help="Train from scratch", default=0)

    args = parser.parse_args()

    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    trainer = Trainer(
        model_path=args.model_path,
        experiment_name=args.experiment_name,
        # args.run_origin,
        registered_model_name=args.registered_model
    )
    trainer.train(train=int(args.train), n_epochs=2)
