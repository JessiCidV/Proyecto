from flask import Flask, request, jsonify
from PIL import Image
import torch
import io
import os
import sys

# Agregar yolov5 al path
sys.path.append(os.path.abspath("yolov5"))

# Cargar modelo
model = torch.load("best.pt") if isinstance("best.pt", torch.nn.Module) else torch.hub.load("ultralytics\yolov5", "custom", path="best.pt", source='local', force_reload=True)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No se recibió imagen'}), 400

    file = request.files['image']
    image_bytes = file.read()
    img = Image.open(io.BytesIO(image_bytes))

    results = model(img)
    detecciones = results.pandas().xyxy[0].to_dict(orient="records")

    return jsonify({'detections': detecciones})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
