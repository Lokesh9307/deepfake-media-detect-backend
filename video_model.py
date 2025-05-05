# video_model.py
import cv2
import torch
from PIL import Image
from utils import load_model, get_transform, generate_graph

def predict_video(filepath, frame_skip=30):
    model = load_model()
    transform = get_transform()
    cap = cv2.VideoCapture(filepath)

    real, fake, count = 0, 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_skip == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                output = model(tensor)
                probs = torch.nn.functional.softmax(output, dim=1)[0]
                real += probs[0].item()
                fake += probs[1].item()
        count += 1
    cap.release()

    total = real + fake
    real, fake = (real / total * 100), (fake / total * 100)
    chart = generate_graph(real, fake)
    return "Fake❌" if fake > real else "Real✅", chart
