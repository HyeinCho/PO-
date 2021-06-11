# POresentation
포스코 AI, Big Data 아카데미 11기 A3. 

초심자를 위한 발표 도우미 AI 

## Vision
+ Emotion Recognition
> 발표 중 사용자의 시선을 추적하고 시선회피 등 의심수준을 벗어나는 횟수 파악
> 시계열 그래프를 이용하여 발표 잘하는 사람과 표정 평균값을 비교

+ Gaze Tracking
> 동공이 중앙에 있을 때 (0.5, 0.5)로 설정 후 특정 값 범위에서 벗어나는 경우 의심수준으로 규정해 횟수 측정

## Speech
+ STT / KoSpacing / KSS
> 발표 음성을 텍스트화 후 저장
> STT 결과로 나온 문장들을 KoSpacing와 KSS를 이용해 교정
> Filler 언급 시 횟수 파악 후 표시
> 발표 속도를 계산하여 표현 (평균적으로 1분당 250음절)

+ Doc2Vec
> wiki dump file과 mecab을 이용하여 형태소 분석하여 Doc2Vec 모델 훈련
> STT 대본과 업로드한 대본의 전체 유사도 확인
> 각 문장의 유사도가 70% 미만일 경우 붉은 색으로 표시
