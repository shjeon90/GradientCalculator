Gradient Calculator
==
파이썬으로 작성된 기울기(Gradient) 계산기 입니다.

Installation
==
* Python 3.6 설치 [URL](https://www.python.org/downloads/release/python-368/)
* 라이브러리 설치
```
명령 프롬프트 실행 후, 아래 명령어 입력
>> python -m pip install numpy matplotlib pandas argparse pathlib openpyxl 
```
* 스크립트 실행
```
>> python gradient.py --fpath [데이터 파일 경로] --opath [분석 결과 저장 경로] [-MS]

예시 (ms 포함인 경우):
>> python gradient.py --fpath C:\data_with_ms.xlsx --opath C:\ -MS

예시 (ms 포함하지 않는 경우):
>> python gradient.py --fpath C:\data_without_ms.xlsx --opath C:\ 

예시 (한글이나 공백문자를 포함하는 경로):
>> python gradient.py --fpath "C:\한글\공백 문자\data.xlsx" --opath "C:\공백 문자"
```

Output files
==
opath에 명시한 경로에 output.xlsx 파일이 생성됨.

![fig1](./figure/fig1.PNG)