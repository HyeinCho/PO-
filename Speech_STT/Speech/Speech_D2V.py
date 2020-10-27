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
from konlpy.tag import Mecab
import gensim


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

# 문서(원래 대본, STT 대본) 유사도 비교
def compare_script(doc, original):
    # 대본 형태소 분석
    tok_doc = tokenizer.morphs(doc)
    tok_original = tokenizer.morphs(original)
    # 두 대본 유사도 비교
    doc_similarity = model.docvecs.similarity_unseen_docs(model, tok_doc, tok_original)
    
    return doc_similarity

# cosine_similarity
def cos(doc, original):
    arr1 = original.split(". ")
    arr2 = doc.split("..")
    # count_vec = CountVectorizer()
    tokenizer = Mecab()
    model = gensim.models.Doc2Vec.load("/POresentation/Speech_STT/Speech/D2V/model/doc2vec.model")

    # find arr1 = arr2
    # standard : arr2
    # dict : check arr1 arr2
    dict = {}
    for lst in range(len(arr1)):
        dict[arr1[lst]] = False

    dict_stt = {}
    for lst in range(len(arr2)):
        dict_stt[arr2[lst]] = False

    for i in range(len(arr2)):
        cos_ = []  # arr2[i] and arr1 cosine_similarity save
        doc_words2 = tokenizer.morphs(arr2[i])
        for j in range(len(arr1)):
            # arr_matrix = count_vec.fit_transform([arr1[j], arr2[i]]).toarray()
            # cos_value = cosine_similarity(arr_matrix)
            doc_words1 = tokenizer.morphs(arr1[j])
            cos_value = model.docvecs.similarity_unseen_docs(model, doc_words1, doc_words2)
            cos_.append(cos_value)

        # max cos_value
        max_cos = max(cos_)
        if max_cos >= 0.7:
            arr_num = cos_.index(max_cos)
            if dict[arr1[arr_num]] is False:
                dict[arr1[arr_num]] = True
                dict_stt[arr2[i]] = True

    # , alpha=1, min_alpha=0.0001, steps=5

    # original text
    for k, v in dict.items():
        if v is False:
            print('\x1b[1;31m' + k + '\x1b[1;m')
        else:
            print('\x1b[1;2m' + k + '\x1b[1;m')

    print()
    # STT text
    for k, v in dict_stt.items():
        if v is False:
            print('\x1b[1;31m' + k + '\x1b[1;m')
        else:
            print('\x1b[1;2m' + k + '\x1b[1;m')


# main

original = input("대본 입력: ")
# stt_str, sec = audio_stt("Test_3min_audio.wav")
stt_str, sec = audio_stt("Present_audio.wav")
s_doc = put(stt_str)
m_doc, doc_filler = check_filler(s_doc)
doc = sentence_modify(m_doc)
doc_similarity = compare_script(doc, original)
speed = check_speed(doc, sec)

print("Similarlity:", doc_similarity)
print()
print("your filler :", doc_filler)
print()
print("modify text :", doc)
print()
print("your speech speed :", speed)
print()
cos(doc, original)
