#pip install SpeechRecognition
import speech_recognition as sr
import time
from time import sleep

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

text=""
while True:
    tmp = VoiceRecognition()
    if tmp==None:
        continue
    text+= tmp + " "
    now = time.localtime()
    log_time="%02d:%02d:%02d" % (now.tm_hour, now.tm_min, now.tm_sec)
    Log(log_time, text)