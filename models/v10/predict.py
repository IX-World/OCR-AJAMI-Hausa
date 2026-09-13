import argparse
import sys
from pathlib import Path
import unicodedata
import torch
from PIL import Image, ImageOps
from model import CRNN


def predict(image_path, checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    vocab = checkpoint['vocab']
    reverse = {i: c for c, i in vocab.items()}
    if '<UNK>' in vocab:
        reverse[vocab['<UNK>']] = '\ufffd'
    model = CRNN(len(vocab))
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to(device).eval()
    height = checkpoint['height']
    with Image.open(image_path) as source:
        image = source.convert('L')
    width = max(4, round(image.width * height / image.height))
    image = ImageOps.mirror(image.resize((width, height), Image.Resampling.LANCZOS))
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    data = (data.reshape(1, 1, height, width).float() / 255.0 - 0.5) / 0.5
    with torch.inference_mode():
        ids = model(data.to(device)).argmax(-1)[0].cpu().tolist()
    result, previous = [], None
    for index in ids:
        if index != vocab['<BLANK>'] and index != previous:
            result.append(reverse[index])
        previous = index
    return unicodedata.normalize('NFC', ''.join(result))


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('--checkpoint', default=str(Path(__file__).parent/'best_cer_v10.pt'))
    args = parser.parse_args()
    print(predict(args.image, args.checkpoint))
