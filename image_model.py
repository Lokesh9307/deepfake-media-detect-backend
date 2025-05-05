# image_model.py
from PIL import Image
import torch
from utils import load_model, get_transform, generate_graph

def predict_image(file):
    model = load_model()
    transform = get_transform()

    image = Image.open(file).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        real, fake = probs[0].item() * 100, probs[1].item() * 100

    chart = generate_graph(real, fake)
    return "Fake❌" if fake > real else "Real✅", chart
