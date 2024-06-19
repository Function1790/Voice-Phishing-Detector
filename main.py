from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd

tokenizer = Tokenizer()
max_len = 189
data = pd.read_csv('spam.csv', encoding='latin1')
del data['Unnamed: 2']
del data['Unnamed: 3']
del data['Unnamed: 4']
data['v1'] = data['v1'].replace(['ham','spam'],[0,1])
data.drop_duplicates(subset=['v2'], inplace=True)
data['v1'].value_counts().plot(kind='bar')
X_data = data['v2']
y_data = data['v1']
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=0, stratify=y_data)

from keras.models import load_model
from googletrans import Translator
model = load_model('model.h5')

translator = Translator()
def toEng(text):
    return translator.translate(text, dest="en").text

import speech_recognition as sr
import asyncio
from os import system
import time
system("cls")

def Log(title, content):
    print(f"[{title}] >> {content}")

def VoiceRecognition():
    r=sr.Recognizer()
    with sr.Microphone() as source:
        audio=r.listen(source)
        said=" "
        try:
            said=r.recognize_google(audio, language='ko-KR')
            now = time.localtime()
            log_time="%02d:%02d:%02d" % (now.tm_hour, now.tm_min, now.tm_sec)
            #Log(log_time ,said)
        except Exception as e:
            now = time.localtime()
            log_time="%02d:%02d:%02d" % (now.tm_hour, now.tm_min, now.tm_sec)
            Log(log_time, "Missing..")
            return None
    return said

async def predictData(text):
    test_data = [toEng(text)]
    test = tokenizer.texts_to_sequences(test_data)
    test = pad_sequences(test, maxlen = max_len)
    print(model.predict(test))

text=""
print("마이크 수신 대기중")
while True:
    tmp = VoiceRecognition()
    if tmp==None:
        continue
    text+= tmp + ". "
    now = time.localtime()
    log_time="%02d:%02d:%02d" % (now.tm_hour, now.tm_min, now.tm_sec)
    Log(log_time, text)
    asyncio.run(predictData(text))