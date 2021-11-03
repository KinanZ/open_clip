import clip
import pandas as pd
import numpy as np

df = pd.read_csv('/home/kinan/Dropbox/Uni/Thesis/open_clip_debug/train_data.csv', sep=",")
images = df['filepath'].tolist()
captions = df['sentence'].tolist()

sentence = "V.a. schmales SDH entlang der Falx . Erhebliche Befundverschlechterung im Verlauf bei neu aufgetretener ausgedehnter Parenchymblutung rechts temporal mit perifokalem Ödem sowie bei neu aufgetretenem SDH mit fraglicher epiduraler Komponente rechts frontotemporoparietal ."
text_1 = clip.tokenize(sentence,  context_length=77)[0]
print(text_1)
text_2 = clip.tokenize(sentence[:-10],  context_length=77)[0]
print(text_2)
diff = text_1 - text_2
print(diff)
