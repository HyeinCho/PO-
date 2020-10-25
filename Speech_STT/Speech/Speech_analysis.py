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
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
import wave
import contextlib
import os


# 음성추출

def audio_stt(audio_file):
    # audio file 받기
    AUDIO_FILE = audio_file

    # wav 파일 길이 구하기
    sec = 0
    with contextlib.closing(wave.open(AUDIO_FILE, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        sec += duration

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


# 띄어쓰기
def sentence_modify(s_data):
    doc = ""

    # 문장 띄어쓰기 수행
    doc = spacing(''.join(s_data.split(' ')))
    return doc


# 마침표 붙이기
def put(s_data):
    doc = ""
    # 마침표 붙이기
    for sent in kss.split_sentences(s_data):
        doc += sent + '..'

    return doc


# 속도 시각화
def check_speed(s, sec):
    syllable = ''.join(s.split())  # 공백 제외 글자 수(음절 수)

    # 평균 발표 속도: 250 / min
    # 발표 속도: 음절개수 / min
    speaking_rate = len(syllable) * 60 / sec

    fig = go.Figure(go.Indicator(
        domain={'x': [0, 1], 'y': [0, 1]},
        value=speaking_rate,
        mode="gauge+number+delta",
        title={'text': "Speed"},
        delta={'reference': 250},
        gauge={'axis': {'range': [None, 500]},
               'steps': [
                   {'range': [0, 250], 'color': "lightgray"},
                   {'range': [250, 500], 'color': "gray"}],
               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 490}}))

    fig.show()

    return speaking_rate


# 필러 체크
def check_filler(doc):
    filler = ['뭐', '음', '그', '어', '그냥', '이제', '좀', '아', '한',
              '그거', '대게', '막', '그게', '그니까', '그래', '근데',
              '일단', '아마', '저기', '이', '뭐지', '뭔가', '스', '하', '자',
              '에', '이게', '뭐더라']
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


# cosine_similarity
def cos(doc, original):
    arr1 = original.split(". ")
    arr2 = doc.split("..")
    count_vec = CountVectorizer()

    # find arr1 = arr2
    # standard : arr2
    # dict : check arr1 arr2
    dict = {}
    for lst in range(len(arr1)):
        dict[arr1[lst]] = False

    for i in range(len(arr2)):
        cos_ = []  # arr2[i] and arr1 cosine_similarity save
        for j in range(len(arr1)):
            arr_matrix = count_vec.fit_transform([arr1[j], arr2[i]]).toarray()
            cos_value = cosine_similarity(arr_matrix)
            cos_.append(cos_value[0][1])

        # max cos_value
        max_cos = max(cos_)
        if max_cos >= 0.7:
            arr_num = cos_.index(max_cos)
            if dict[arr1[arr_num]] is False:
                dict[arr1[arr_num]] = True

    for k, v in dict.items():
        if v is False:
            print('\x1b[1;31m' + k + '\x1b[1;m')
        else:
            print('\x1b[1;2m' + k + '\x1b[1;m')


# main
# original 대본 예시 : 3차 발표를 맡은 A3조 신정우입니다. 저희의 주제는 발표 연습 도우미인 포레젠테이션입니다. 목차는 다음과 같습니다. 저희 프로젝트 추진 배경입니다. 스피치 상황이 많아진 자기 PR 시대에 발표 능력 향상을 목표로 하는 소비자의 needs가 증가하고 있습니다. 그리고 발표를 두려워하거나 발표 능력이 부족한 사람들을 위한 서비스가 부재하기 때문에 발표 연습을 도와주는 AI프로젝트를 계획하게 됐습니다. 현재 기술현황으로는 마이크로소프트에서 제공하는 프레젠테이션 코치와 면접연습 AI인 뷰인터가 있습니다. 프레젠테이션 코치의 경우, 필러 유무 서비스 등의 장점이 있지만, 파워포인트에서만 이용이 가능하고, 영어로만 사용이 가능하다는 단점이 있습니다. 그래서 포레젠테이션에서는 노션, 프레지 등 다양한 발표 도구를 사용할 수 있으며, 한국어 발표 연습이 가능하도록 목표를 설정했습니다. 뷰인터의 경우 영상을 통해 시각과 음성을 분석하는 서비스를 제공하는데 이 기술을 참고하여 시선처리와 표정으로 시각분석을 하고 대본과 실제 발표 음성을 비교하여 분석하는 것을 목표로 설정했습니다. 포레젠테이션의 시스템 구조는 발표 대본을 먼저 업로드하고 발표 영상을 녹화하면 gaze tracking과 감정인식으로 시각분석을 하고, 실제 발표를 STT를 이용하여 Text화 시켜서 Word2Vec로 업로드한 대본과 비교하여 음성분석을 합니다. 그리고 STT에서 나온 단어 수를 파악해 말의 속도, 자주 사용하는 단어 등을 결과 레포트에 나타낼 것입니다. 발표가 끝나면 이러한 결과들을 보고서에 출력한 후, 발표 연습을 끝내거나 만족하지 못할 경우 대본을 재 업로드하거나 발표 연습을 다시하는 구조입니다. 다음으로 사용하는 기술을 소개하겠습니다. Gaze Tracking의 경우는 사용자의 시선이 화면에서 극단적으로 벗어나는 경우를 의심수준으로 설정하고 그 횟수를 파악할 것입니다. Emotion Recognition의 경우 발표자의 표정 중 엥그리 뉴트럴 해피를 인식하고 그 결과를 시계열 그래프로 나타내여 결과 보고서에 출력할 것입니다. STT는 사용자의 말하기를 문자로 바꿔서 실제 발표 내용을 확인하여 발표능력 피드백에 활용할 것입니다. Word2Vec과 코엔엘파이를 사용해서 업로드한 대본과 STT를 이용하여 만든 텍스트를 비교하여 단어별로 유사도를 체크할 것입니다. 먼저 시각분석 진행현황입니다. 사진에서 빨간 네모박스를 보시면 Emotion Recognition을 하는 것으로 표정을 인식하여 현재 사진에는 뉴트럴이 나타나고 있습니다. 눈동자에는 초록색 십자가 표시가 있는데 gaze tracking을 하는 것입니다.

original = input("대본 입력: ")
# stt_str, sec = audio_stt("Test_3min_audio.wav")
stt_str, sec = audio_stt("Presen_audio.wav")
s_doc = put(stt_str)
m_doc, doc_filler = check_filler(s_doc)
doc = sentence_modify(m_doc)
speed = check_speed(doc, sec)

print("your filler :", doc_filler)
print()
print("modify text :", doc)
print()
print("your speech speed :", speed)
print()
cos(doc, original)
