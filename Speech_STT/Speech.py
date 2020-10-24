# GPU 설정
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  except RuntimeError as e:
    print(e)

from pykospacing import spacing
import kss
import plotly.graph_objects as go
import speech_recognition as sr

# 음성추출

def audio_stt(audio_file):
    # audio file 받기
    AUDIO_FILE = audio_file
    sec = 27 # 일단 임의로 27

    # audio file을 audio source로 사용합니다
    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source)  # 전체 audio file 읽기
    s_data = ""
    try:
        # 인식하면 s_data에 입력
        s_data = r.recognize_google(audio, language='ko')

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
    return s_data, sec

# 띄어쓰기, 마침표 수정
def sentence_modify(s_data):
    ls = ""
    # 마침표 붙이기
    for sent in kss.split_sentences(s_data):
        ls += sent + '. '

    # 문장 띄어쓰기 수행
    doc = spacing(''.join(ls.split(' ')))

    return doc

# 속도 시각화
def check_speed(s, sec):
    syllable = ''.join(s.split()) # 공백 제외 글자 수(음절 수)

    # 평균 발표 속도: 250 / min
    # 발표 속도: 음절개수 / min
    speaking_rate = len(syllable) * 60 / sec

    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = speaking_rate,
        mode = "gauge+number+delta",
        title = {'text': "Speed"},
        delta = {'reference': 250},
        gauge = {'axis': {'range': [None, 500]},
                 'steps' : [
                     {'range': [0, 250], 'color': "lightgray"},
                     {'range': [250, 500], 'color': "gray"}],
                 'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 490}}))

    fig.show()

    return speaking_rate

# 필러 체크
def check_filler(doc):
    filler = ['뭐', '음', '그', '어', '그냥', '이제', '좀', '아', '한',
              '그거', '대게', '막', '그게', '그니까', '그래', '근데',
              '일단','아마','저기','이','뭐지','뭔가','스', '하', '자',
              '에', '이게','뭐더라']
    # filler 제거 check
    remove_doc = []


    list_filler = doc.split()

    s_filler = dict()

    for i in range(len(list_filler)):
        if list_filler[i] in filler:
            remove_doc.append(i)
            if s_filler.get(list_filler[i]):
                s_filler[list_filler[i]] += 1
            else:
                s_filler[list_filler[i]] = 1
    # filler 제거한 doc 문장
    list(map(list_filler.pop, remove_doc))
    m_doc = ''.join(list_filler)
    return m_doc, s_filler


#main
stt_str, sec = audio_stt("Test_3min_audio.wav")
m_doc, doc_filler = check_filler(stt_str)
doc = sentence_modify(m_doc)
speed = check_speed(doc, sec)

print("original text :", stt_str)
print()
print("your filler :", doc_filler)
print()
print("remove filler :", m_doc)
print()
print("modify text :", doc)
print()
print("your speech speed :", speed)