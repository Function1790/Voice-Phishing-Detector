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
model = load_model('model.h5')

#Google 번역 라이브러리
from googletrans import Translator

translator = Translator() #번역기 불러오기
#번역: 한국어 -> 영어
def toEng(text): 
    return translator.translate(text, dest="en").text

import speech_recognition as sr
import asyncio
from os import system
import time
system("cls")

def Log(title, content):
    print(f"[{title}] >> {content}")

#마이크로 음성 데이터를 텍스트로 받기
def VoiceRecognition():
    r=sr.Recognizer()
    with sr.Microphone() as source:
        audio=r.listen(source) #문장이 끝날때까지 음성 데이터 받기
        said=" "
        try:
            #말한 내용을 한국어로 처리
            said=r.recognize_google(audio, language='ko-KR')
            now = time.localtime()
            log_time="%02d:%02d:%02d" % (now.tm_hour, now.tm_min, now.tm_sec)
            #Log(log_time ,said)
        except Exception as e:
            now = time.localtime()
            log_time="%02d:%02d:%02d" % (now.tm_hour, now.tm_min, now.tm_sec)
            Log(log_time, "Missing..")
            return None # 말한 내용이 없으면 None 반환
    return said #말한 내용이 있으면 그 데이터 반환

#대화 내용을 통해 보이스피싱일 확률 판단 함수
async def predictData(text):
    test_data = [toEng(text)] #한국어 데이터를 영어로 변환
    test = tokenizer.texts_to_sequences(test_data)
    test = pad_sequences(test, maxlen = max_len)
    print(model.predict(test)) #보이스피싱일 확률 출력

text=""
print("마이크 수신 대기중")
while True: #무한 반복 #부분1
    voiceData = VoiceRecognition() # 음성데이터 받기
    if voiceData==None: #음성 데이터가 없다면
        continue #다음 코드를 실행하지 않고 '부분1'로 돌아가기
    #음성 데이터 마지막 부분에 마침표('.') 찍고 기존 내용에 새로 받은 내용 추가
    text+= voiceData + ". "
    now = time.localtime()
    log_time="%02d:%02d:%02d" % (now.tm_hour, now.tm_min, now.tm_sec)
    Log(log_time, text)
    asyncio.run(predictData(text)) #내용이 보이스피싱일 확률 예측 및 출력