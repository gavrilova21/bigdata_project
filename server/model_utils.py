import gc
import torch, torch.nn as nn
from tqdm import tqdm
import torch.onnx
import onnx
import platform
import mlflow
import mlflow.pytorch
import mlflow.onnx
import torchvision.transforms as transforms
import torchvision
from torch.utils.tensorboard import SummaryWriter


mlflow.set_tracking_uri("http://localhost:5000")
client = mlflow.tracking.MlflowClient()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class SuperResolution(nn.Module):
    """
    Network Architecture as per specified in the paper.
    The chosen configuration for successive filter sizes are 9-5-5
    The chosed configuration for successive filter depth are 128-64(-3)
    """
    def __init__(
        self,
        sub_image: int = 33,
        spatial: list = [9, 5, 5],
        filter: list = [128, 64],
        num_channels: int = 3
    ):
        super().__init__()
        self.layer_1 = nn.Conv2d(num_channels, filter[0], spatial[0], padding = spatial[0] // 2)
        self.layer_2 = nn.Conv2d(filter[0], filter[1], spatial[1], padding = spatial[1] // 2)
        self.layer_3 = nn.Conv2d(filter[1], num_channels, spatial[2], padding = spatial[2] // 2)
        self.relu = nn.ReLU()

    def forward(self, image_batch):
        x = self.layer_1(image_batch)
        x = self.relu(x)
        x = self.layer_2(x)
        y = self.relu(x)
        x = self.layer_3(y)
        return x, y


def convert_torch_to_onnx(model, exp_name='', creation_date='', hash_commit=''):
    """Convert pytorch model to onnx format."""
    # Dummy input
    x = torch.randn(128, 3, 9, 9, requires_grad=True)
    onnx_model_file = "super_resolution.onnx"
    # Export the model
    torch.onnx.export(
        model,
        x,
        f"./{onnx_model_file}",
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names = ['input'],
        output_names = ['output'],
        dynamic_axes={
            'input': {0 : 'batch_size'},
            'output': {0 : 'batch_size'}}
    )
    onnx_model = onnx.load(onnx_model_file)

    meta = onnx_model.metadata_props.add()
    meta.key = "exp_name"
    meta.value = exp_name

    meta = onnx_model.metadata_props.add()
    meta.key = "hash commit"
    meta.value = hash_commit

    meta = onnx_model.metadata_props.add()
    meta.key = "save_date"
    meta.value = creation_date

    onnx.save(onnx_model, onnx_model_file)
    return onnx_model


def execute(image_in, model, device, fs=33, overlap=False, scale=2):
    """
    Executes the model trained on colab, on any image given (link or local), with an
    upscaling factor as mentioned in the arguments. For best results, use a scale of
    2 or lesser, since the model was trained on a scale of 2
    Inputs : image_in               -> torch.tensor representing the image, can be easily obtained from
                                       transform_image function in this script (torch.tensor)
             model                  -> The trained model, trained using the same patch size
                                       (object of the model class, inherited from nn.Module)
             fs                     -> Patch size, on which the model is run (int)
             overlap                -> Reconstruction strategy, more details in the readme (bool)
             scale                  -> Scale on which the image is upscaled (float)
    Outputs: reconstructed_image    -> The higher definition image as output (PIL Image)
    """

    to_tensor = transforms.ToTensor()
    image_in = image_in.convert('RGB')

    w, h = image_in.size
    scale_transform = transforms.Resize(
        (int(h * scale), int(w * scale)),
        interpolation=3
    )

    to_pil = transforms.ToPILImage()
    image = to_tensor(scale_transform(image_in))
    n = 0
    c, h, w = image.shape
    image = image.unsqueeze(0)
    image = image.to(device)
    reconstructed_image = torch.zeros_like(image).cpu()
    reconstructed_image_weights = torch.zeros_like(image).cpu()

    if overlap:
        for i in tqdm(range(h - fs + 1), desc = 'Progressively Scanning'):
            for j in range(w - fs + 1):
                gc.collect()
                patch = image[:, :, i: i + fs, j: j + fs]
                reconstructed_image[:, :, i: i + fs, j: j + fs] += model(patch)[0].cpu().clamp(0, 1)
                reconstructed_image_weights[:, :, i: i + fs, j: j + fs] += torch.ones(1, c, fs, fs)
        reconstructed_image /= reconstructed_image_weights
    else:
        for i in tqdm(range(h // fs), desc = 'Progressively Scanning', ncols = 100):
            for j in range(w // fs):
                gc.collect()
                n += 1
                patch = image[:, :, i * fs: i * fs + fs, j * fs: j * fs + fs]
                reconstructed_image[:, :, i * fs: i * fs + fs, j * fs: j * fs + fs] = model(patch)[0].cpu().clamp(0, 1)
                reconstructed_image_weights[:, :, i * fs: i * fs + fs, j * fs: j * fs + fs] += torch.ones(1, c, fs, fs)
                if j == w // fs - 1:
                    patch = image[:, :, i * fs: i * fs + fs, w - fs: w]
                    reconstructed_image[:, :, i * fs: i * fs + fs, w - fs: w] = model(patch)[0].cpu().clamp(0, 1)
                if i == h // fs - 1:
                    patch = image[:, :, h - fs: h, j * fs: j * fs + fs]
                    reconstructed_image[:, :, h - fs: h, j * fs: j * fs + fs] = model(patch)[0].cpu().clamp(0, 1)
        patch = image[:, :, h - fs: h, w - fs: w]
        reconstructed_image[:, :, h - fs: h, w - fs: w] = model(patch)[0].cpu().clamp(0, 1)

    print("Channels = {}, Image Shape = {} x {}".format(c, w, h))
    return to_pil(reconstructed_image.squeeze())




def load_loader_stl(crop_size: int = 33, batch_size: int = 128, num_workers: int = 1, scale: float = 2.0):
    """
    Loads the dataloader of the STL-10 Dataset using the given specifications with the required
                          augmentation schemes
    input : crop_size -> image size of the square sub images the model has been trained on
            scale     -> Scale by which the low resolution image is downscaled
    output: dataloader iterable to be able to train on the images

    Augmentation Schemes: Since torch has strong built in support for transforms, augmentation
                          was done within our dataloader transforms employing TenCrop on each
                          image. For every image we get 5 crops (Center + 4 corners) and the horizontal
                          flip of each. TenCrop returns a tuple, which was handled using lambda
                          and also in the training script in the cell below.

    """
    # Write transforms for TenCrop and for generating low res images using bicubic interpolation (interpolation = 3)
    transform_high_res = transforms.Compose([
            transforms.TenCrop(crop_size),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
        ])
    transform_low_res = transforms.Compose([
            transforms.Resize(int(96 / scale), interpolation=3),
            transforms.Resize(96, interpolation=3),
            transform_high_res
        ])

    # Make STL-10 dataset object
    dataset_high_res = torchvision.datasets.STL10('.', transform = transform_high_res, download = True)
    dataset_low_res = torchvision.datasets.STL10('.', transform = transform_low_res, download = False)

    # Create the dataloader object using the transforms (Not shuffled since we will be checking progress on the same examples)
    dataloader_high_res = torch.utils.data.DataLoader(dataset_high_res, batch_size = batch_size, num_workers = num_workers, shuffle = False)
    dataloader_low_res = torch.utils.data.DataLoader(dataset_low_res, batch_size = batch_size, num_workers = num_workers, shuffle = False)
    return dataloader_low_res, dataloader_high_res


def train_from_scratch(n_epochs=500, num_workers=1):
    """
    Train function for training and constantly visualizing intermediate layers and
    immediate outputs. All images relevant, along with losses are tracked on tensorboard
    in the first cell of this notebook. All hyperparameters are directly embedded in the
    code since the model has few to begin with, and the ones that exist also have fairly
    standard values

    We achieve lesser PSNR with the same configurations as the paper since we train for
    much lesser steps (They train for 10^8 backward steps), since complete training
    according to the paper was simply infeasible given the idle time of a colab notebook
    is only 90 minutes
    """
    # Initialize model, data, writer, optimizer, and backward count
    low_res_loader, high_res_loader = load_loader_stl(num_workers=num_workers)
    model = SuperResolution()

    # model.load_state_dict(torch.load('/isr_best.pth'))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), 1e-04)
    writer = SummaryWriter("train_logs")
    n = 0

    for epoch in tqdm(range(n_epochs), desc="Training", ncols=120):
        for low_res, high_res in zip(low_res_loader, high_res_loader):

            # Convert TenCrop tuple into a trainable shape of (batch_size * 10, c, h, w)
            low_res_batch, high_res_batch = low_res[0], high_res[0]
            _, _, c, h, w = low_res_batch.size()
            low_res_batch, high_res_batch = low_res_batch.to(device), high_res_batch.to(device)
            low_res_batch, high_res_batch = low_res_batch.view(-1, c, h, w), high_res_batch.view(-1, c, h, w)
            reconstructed_batch, intermediate = model(low_res_batch)

            # Calculate gradients and make a backward step on MSE loss
            loss_fn = nn.MSELoss()
            loss = loss_fn(high_res_batch, reconstructed_batch)
            loss_to_compare = loss_fn(high_res_batch, low_res_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Clamp the image between 0 and 1 and prepare transforms and image arrays to write on tensorboard
            to_pil = torchvision.transforms.ToPILImage()
            resize = torchvision.transforms.Resize((48 * 7, 144 * 7))
            other_resize = torchvision.transforms.Resize((48 * 5, 48 * 5))
            to_tensor = torchvision.transforms.ToTensor()
            ind = 4
            image = to_pil(torch.cat((low_res_batch[ind], high_res_batch[ind], reconstructed_batch[ind]), dim = 2).cpu())
            image = to_tensor(resize(image))
            image = image.clamp(0, 1)
            n += 1
            psnr = 10 * torch.log10(1 / loss)
            psnr_tc = 10 * torch.log10(1 / loss_to_compare)

            # Write relevant scalars and comparitive images on tensorboard
            writer.add_scalar("MSE loss", loss * (255 ** 2), n)
            writer.add_scalar("PSNR of Reconstruction", psnr, n)
            writer.add_scalar("PSNR of BiCubic Interpolation (For comparision)", psnr_tc, n)
            writer.add_image("Low Resolution Image | High Resolution Image | Reconstructed Image", image, n, dataformats='CHW')
            writer.flush()
        # Save the model for every epoch
        torch.save(model.state_dict(), 'isr_best_2.pth'.format(n))

    return model, (loss.item()) * (255 ** 2)


class Trainer(object):
    def __init__(
        self,
        model_path=None,
        experiment_name='super_resolution',
        registered_model_name=None
    ):
        self.model_path = model_path
        self.run_origin = "none"
        self.experiment_name = experiment_name
        self.registered_model_name = registered_model_name
        if not self.model_path:
            raise Exception('Only pretrained model!')
        mlflow.set_experiment(experiment_name)
        client = mlflow.tracking.MlflowClient()
        experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
        print("experiment_id:", experiment_id)

    def train(self, train=False, n_epochs=100, num_workers=0):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # retrain model
        train_loss = -1
        if train == 1:
            model, train_loss = train_from_scratch(n_epochs, num_workers=num_workers)
        elif train == 0:
            model = SuperResolution()
            model.load_state_dict(
                torch.load(self.model_path, map_location={'cuda:0': 'cpu'})
            )
        else:
            raise Exception('Wrong value of train argument. Can be only 0/1.')
        # print(train_loss)
        model.to(device)
        model.eval()

        with mlflow.start_run(run_name=self.run_origin) as run:

            run_id = run.info.run_uuid
            experiment_id = run.info.experiment_id
            print("MLflow:")
            print("  run_id:", run_id)
            print("  experiment_id:", experiment_id)

            # MLflow params
            print("Parameters:")
            print("  n_epochs:", n_epochs)
            mlflow.log_param("n_epochs", n_epochs)

            # MLflow metrics
            print("Metrics:")
            print("MSE:", train_loss)
            mlflow.log_metric("mse", train_loss)

            mlflow.set_tag("mlflow.runName", self.run_origin)
            mlflow.set_tag("data_path", self.model_path)
            mlflow.set_tag("exp_id", experiment_id)
            mlflow.set_tag("exp_name", self.experiment_name)
            mlflow.set_tag("run_origin", self.run_origin)
            mlflow.set_tag("platform", platform.system())

            if self.registered_model_name is None:
                mlflow.pytorch.log_model(model, "pytorch-model")
            else:
                mlflow.pytorch.log_model(model, "pytorch-model", registered_model_name=self.registered_model_name)
