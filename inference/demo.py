import torch
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
import time
import os
from PIL import Image
import numpy as np

print("Available models:", available_models())  
# Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']

device = "cuda" if torch.cuda.is_available() else "cpu"

start = time.time()
model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
model.eval()

#image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
images = []
fnames = []
for file in sorted(os.listdir("./9zhai_katna/"), key=lambda x: int(x[:-7])):
    file = os.path.join("./9zhai_katna/", file)
    if os.path.isfile(file) and file.endswith('.jpeg'):
        fnames.append(file)
        images.append(preprocess(Image.open(file)).unsqueeze(0).to(device))
        
images = torch.concat(images, dim=0)
labels = [u"湖面", u"栈道", u"雪山", u"树林", u"瀑布", u"河流", u"河岸", u"石头", u"绿山"]
text = clip.tokenize(labels).to(device)

with torch.no_grad():
    image_features = model.encode_image(images)
    text_features = model.encode_text(text)
    
    image_features /= image_features.norm(dim=-1, keepdim=True) 
    text_features /= text_features.norm(dim=-1, keepdim=True)    

    logits_per_image, logits_per_text = model.get_similarity(images, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    preds = np.argmax(probs, axis=1)
end = time.time()

for i, fname in enumerate(fnames):
    print(fname, ":", labels[preds[i]], probs[i][preds[i]])
#print("Label probs:", preds, "by", device)  # prints: [[0.9927937  0.00421068 0.00299572]]

print("Elapsed time:", end - start)