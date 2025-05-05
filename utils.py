# utils.py
import torch
import torchvision.transforms as transforms
import io
import base64
import matplotlib
import matplotlib.pyplot as plt
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights

matplotlib.use('Agg') 
def load_model():
    weights = ResNeXt50_32X4D_Weights.DEFAULT
    model = resnext50_32x4d(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.eval()
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

def generate_graph(real, fake):
    fig, ax = plt.subplots()
    ax.bar(['Real', 'Fake'], [real, fake], color=['green', 'red'])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Probability (%)")
    ax.set_title("Fake Media Detection")
    for i, v in enumerate([real, fake]):
        ax.text(i, v + 2, f"{v:.2f}%", ha='center')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
