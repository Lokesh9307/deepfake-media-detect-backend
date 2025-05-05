# audio_model.py
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import torch
from PIL import Image
import os
import numpy as np
from utils import load_model, get_transform, generate_graph

def predict_audio(filepath):
    y, sr = librosa.load(filepath)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots()
    librosa.display.specshow(S_DB, sr=sr, ax=ax, y_axis='mel', x_axis='time')
    ax.axis('off')

    tmpfile = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    plt.savefig(tmpfile.name, bbox_inches='tight', pad_inches=0)
    plt.close()

    image = Image.open(tmpfile.name).convert('RGB')
    input_tensor = get_transform()(image).unsqueeze(0)

    model = load_model()
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        real, fake = probs[0].item() * 100, probs[1].item() * 100

    os.remove(tmpfile.name)
    chart = generate_graph(real, fake)
    return "Fake❌" if fake > real else "Real✅", chart
