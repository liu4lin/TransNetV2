import torch
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
import time
import os
from PIL import Image
import numpy as np
import jieba.analyse
import re
import moviepy.editor as mp
from moviepy.editor import ImageClip, TextClip, CompositeVideoClip
# export PATH=$PATH:/opt/homebrew/bin # for ImageMagick

def split_sentences(text):
    pattern = re.compile(r'([。！？；])')
    sentences = pattern.split(text)
    new_sentences = []
    for i in range(0, len(sentences)-1, 2):
        new_sentence = sentences[i] + sentences[i+1]
        new_sentences.append(new_sentence.strip())
    if not re.match(pattern, sentences[-1]):
        new_sentences[-1] += sentences[-1]
        new_sentences[-1] = new_sentences[-1].strip()
    return new_sentences

import os

def list_files(path, suffix=None):
    # Get all files in the current directory
    files = os.listdir(path)
    output = []
    for file in files:
        # Join the current path with the file name
        current_path = os.path.join(path, file)
        # If the current path is a directory, call the function recursively
        if os.path.isdir(current_path):
            output += list_files(current_path, suffix)
        else:
            check = suffix and file.endswith(suffix)
            if suffix is None or check:
                output.append(current_path)
    return output





scripts = u'''
我曾经有幸游览澳大利亚的大堡礁，那是一次难忘的旅程。
我们预订了一天的珊瑚礁之旅，乘坐船只出发。一路上，我和同伴们尽情欣赏着蔚蓝的海水和远处绵延的珊瑚礁。到达后，我们穿上潜水装备，开始了水下探险。
我感受到了不同寻常的兴奋，因为我看到了生动的色彩和千奇百怪的珊瑚。在水下漫步时，我还遇到了海龟和五颜六色的热带鱼。它们在我身边优雅地游动，使我彻底沉浸在了大堡礁的奇妙世界。
当然，游览珊瑚礁并不只是水下的经历，我们还有机会在船上享受美味的海鲜自助餐，欣赏海上的壮丽景色。此外，我们还可以在船上尝试潜水和浮潜，真正体验大堡礁的神奇。
大堡礁是一个绝对值得一游的地方。我感到非常幸运，能够亲身体验这个世界自然遗产的壮观景观和生态系统。我绝对会再回来的。
'''
summaries = u'游览澳大利亚的大堡礁，是一次充满色彩和兴奋的水下探险之旅。'

scripts = u'''
我去过许多城市，但我一直想去墨尔本。最近，我终于实现了这个愿望。
墨尔本是澳大利亚最大的城市之一，也是文化和艺术的中心。我对这座城市印象深刻，因为它有许多历史建筑和博物馆。我还去了圣基尔达海滩，在那里我看到了壮观的日落。
我最喜欢的景点是墨尔本皇家展览馆，这是一座非常大的博物馆，收藏了许多澳大利亚的历史文物。展览馆的建筑非常漂亮，我还在那里参加了一些兴趣小组。
除了博物馆，墨尔本还有很多街头艺术和漂亮的花园。我去了皇家植物园，在那里我看到了许多美丽的花朵和植物。我还参观了弗林德斯车站，这是一座建于19世纪的火车站，现在已经成为一个购物中心。
总的来说，我非常喜欢墨尔本。它是一座充满活力和文化气息的城市，有许多值得探索的景点。我希望有机会再次访问这个美丽的城市。
'''

summaries = u'墨尔本是一个充满现代与时尚的城市。'

scripts = u'''
我曾经有幸去到了中国四川省的九寨沟旅行，这是一个美丽的地方，让我一生难忘。
我第一次到达九寨沟时，被那里的自然美景所震撼。山峦起伏，树木繁茂，水流清澈见底，我仿佛置身于一个天然的童话世界。沿着蜿蜒的山路，我逐渐领略到了这片土地的美丽之处。在这里，我看到了美丽的瀑布、清澈的湖泊和迷人的森林。
其中最让我难忘的是五彩池。这个池子看起来就像是一个巨大的宝石，池水呈现出了五彩缤纷的色彩，包括蓝色、绿色、黄色等等，每个颜色都非常明亮，这是因为池底的石头和水中微生物的生长造成的。我很喜欢在池边漫步，欣赏这个神奇的景观。
此外，我还参观了藏族村庄，学习了关于当地人的文化和生活方式。我和当地的居民交流，品尝了他们的食物，深深感受到了这里的人情味和淳朴。
在九寨沟的旅程中，我拍下了很多美丽的照片，这些照片不仅记录了我的旅行经历，也让我更加深刻地理解到自然环境和人文景观的重要性。这次旅行让我感受到了自然之美和人文之美的结合，是一次难忘的旅行体验。
'''

summaries = u'九寨沟是一个充满自然之美和人文之美的难忘旅游胜地。'


print("Available models:", available_models())  
# Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']

device = "cuda" if torch.cuda.is_available() else "cpu"

start = time.time()
model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
model.eval()

#image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
images = []
fnames = []
#for file in sorted(os.listdir("./katna/")): #, key=lambda x: int(x[:-7])):
#    file = os.path.join("./katna/", file)
for file in list_files('./katna', '.jpeg'):
    if os.path.isfile(file) and file.endswith('.jpeg'):
        fnames.append(file)
        images.append(preprocess(Image.open(file)).unsqueeze(0).to(device))
candidates = set([i for i in range(len(images))])
images = torch.concat(images, dim=0)

with torch.no_grad():
    image_features = model.encode_image(images)
    image_features /= image_features.norm(dim=-1, keepdim=True) 
        
    sents = split_sentences(scripts)
    sum_kws = jieba.analyse.extract_tags(summaries, topK=3, withWeight=False, allowPOS=('n','ns'))
    print(sum_kws)
    
    #sents = [u"湖面", u"栈道", u"雪山", u"树林", u"瀑布", u"河流", u"河岸", u"石头", u"绿山"]
    durations = []
    imgfiles = []
    for i, sent in enumerate(sents):
        kws = jieba.analyse.extract_tags(sent, topK=3, withWeight=False, allowPOS=None)
        #query = ','.join(kws)
        query = sent + summaries
        #query = sent
        text = clip.tokenize(query).to(device)
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        logits_per_image, logits_per_text = model.get_similarity(images, text)
        probs = logits_per_text.softmax(dim=-1).cpu().numpy()[0]
        preds = np.argsort(probs)[::-1]
        hit = preds[0] # default is argmax
        for pred in preds:
            if pred in candidates and probs[pred] > 0.01: #keep using argmax if no proper hit is found
                hit = pred
                candidates.remove(hit)
                break
        print(str(i), sent[:30], probs[hit], fnames[hit])
        imgfiles.append(fnames[hit])
        durations.append(0.3*len(sent))
        
#imgfiles = [mp.ImageClip(np.asarray(Image.open(file).resize((640,360)))).set_duration(duration) for file, duration in zip(imgfiles, durations)]
#imgfiles = [mp.ImageClip(file).set_duration(duration) for file, duration in zip(imgfiles, durations)]
clips = []
for file, dur, sent in zip(imgfiles, durations, sents):
    img = np.asarray(Image.open(file).resize((640, 360)))
    clip = ImageClip(img).set_duration(dur)
    caption = TextClip(txt=sent, font='Songti-SC-Regular', fontsize=14, color='white', bg_color='black').set_duration(dur)
    caption = caption.set_pos(('center', 'bottom'))
    final_clip = CompositeVideoClip([clip, caption])
    clips.append(final_clip)
video_clip = mp.concatenate_videoclips(clips)
video_clip.write_videofile("9zhai_sent_sum.mp4", fps=24, threads=8)
end = time.time()
print("Elapsed time:", end - start)