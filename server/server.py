import torch
from io import BytesIO
import os
from base64 import b64encode
import mlflow
import zipfile
import mlflow.pytorch

from flask import Flask, request, render_template, jsonify, flash, redirect

import PIL.Image as Image
from tqdm import tqdm

from datetime import datetime
from model_utils import execute, convert_torch_to_onnx, Trainer

app = Flask(__name__)

# load model
mlflow.set_tracking_uri("http://localhost:5000")
client = mlflow.tracking.MlflowClient()
model_name = 'super_resolution'
model_uri = f"models:/{model_name}/production"
model = mlflow.pytorch.load_model(model_uri)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def zip_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() =='zip'

@app.route('/', methods=['GET'])
def predict():
    return render_template('index.html')


@app.route('/forward', methods=['POST'])
def predict_api():
    try:
        request_image = request.files['image']
        image_bytes = request_image.read()
        img = Image.open(BytesIO(image_bytes))
    except Exception:
        return "Bad request", 400
    try:
        return_img = execute(img, model, device)
    except Exception:
        return "Model could not proccess image", 403

    img_dimensions = str(return_img.size)
    buffered = BytesIO()

    # return_img.save("result.png")
    return_img.save(buffered, format="PNG")
    encoded_img = b64encode(buffered.getvalue())

    return jsonify({
        "dim": img_dimensions,
        "img": encoded_img.decode('utf-8')
    })

@app.route('/forward_batch', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        import tempfile

        temp_dir = tempfile.TemporaryDirectory()
        UPLOAD_FOLDER = temp_dir.name
        # UPLOAD_FOLDER = tmp
        # check if the post request has the file part
        print(request.files)
        if 'file' not in request.files:
            return "Bad request", 400
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = file.filename
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            zip_ref = zipfile.ZipFile(os.path.join(UPLOAD_FOLDER, filename), 'r')
            zip_ref.extractall(UPLOAD_FOLDER)
            zip_ref.close()
    return render_template('index.html')

@app.route('/metadata', methods=['GET'])
def get_metadata():
    registered_models = client.get_registered_model(model_name)
    prod_model = None
    for model_version in registered_models.latest_versions:
        if model_version.current_stage == 'Production':
            prod_model = model_version

    prod_model_run = client.get_run(prod_model.run_id)
    hash_commit = prod_model_run.data.tags['mlflow.source.git.commit']
    timestamp = (
        datetime
        .fromtimestamp(prod_model.creation_timestamp / 1000)
        .strftime("%Y/%m/%d %H:%M:%S")
    )
    exp_name = prod_model_run.data.tags['exp_name']
    onnx_model = convert_torch_to_onnx(model, exp_name, timestamp, hash_commit=hash_commit)
    return jsonify({
            "exp_name": exp_name,
            "save_date": timestamp,
            "hash_commit": hash_commit
        })


@app.route('/retrain', methods=['GET'])
def retrain_model():
    status_code = 200
    try:
        trainer = Trainer(
            model_path='../isr_best.pth',
            experiment_name='super_resolution',
            # args.run_origin,
            registered_model_name=None
        )
        trainer.train(train=1, n_epochs=1)
        msg = 'Retrained'
    except Exception:
        status_code = 403
        msg = 'Failed'
    return msg, status_code


if __name__ == '__main__':
    app.run(host="localhost", port=8000, debug=True)
