Gradient Calculator
==
파이썬으로 작성된 기울기(Gradient) 계산기 입니다.

Installation
==
* Python 3.6 설치 [URL](https://www.python.org/downloads/release/python-368/)
* 라이브러리 설치
```
명령 프롬프트 실행 후, 아래 명령어 입력.
>> python -m pip install numpy matplotlib pandas argparse pathlib openpyxl xlrd scikit-learn
```
* 스크립트 다운로드
```
현재 화면 (혹은 아래 url에 접속).
https://github.com/shjeon90/GradientCalculator

우측 상단의 녹색 code 버튼 클릭 후, Download ZIP 버튼 클릭.

다운로드된 파일을 압축해제.
```
* `gradient.py` 스크립트 실행 방법

fpath 디렉토리 내에 엑셀(.xlsx) 파일이 포함된 경우, 첫 번째 시트만 분석함.
```shell
# 명령 프롬프트 실행 후, 아래 명령어 입력.
>> python gradient.py --fpath [데이터 파일이 보관된 디렉토리 경로] --opath [분석 결과 저장 경로] [-MS] [-V]

# 예시 (ms 포함인 경우):
>> python gradient.py --fpath C:\ --opath C:\ -MS

# 예시 (ms 포함하지 않는 경우):
>> python gradient.py --fpath C:\ --opath C:\ 

# 예시 (한글이나 공백문자를 포함하는 경로):
>> python gradient.py --fpath "C:\한글\공백 문자\" --opath "C:\공백 문자"
```

* `gradient_v2.py` 스크립트 실행 방법

기본적인 명령어는 `gradient.py`와 동일하며, `-A` 옵션을 추가하면, `--fpath`에 포함된 파일들의 평균을 계산한.

`-I3` 옵션을 추가하면, 3개 구간 데이터 대한 분석하고, 넣지않으면 2구간 데이터에 대해 분석함. 

현재 미완성 상태이며, 테스트 수준에서는 반드시 `-A` 옵션을 추가할 것.
```shell
>> python gradient_v2.py --fpath [데이터 파일이 보관된 디렉토리 경로] --opath [분석 결과 저장 경로] --r_cliff [단차 탐지 수준(0~1)] --th_curv [곡률 임계치] --s_degree [피팅 모델의 시작 복잡도] --i_degree [복잡도의 간격] --n_degree [분석에사용한 복잡도 개수] --alpha [복잡도를 줄이기 위한 값] --t_estimate [추정하려는 시간(hours)] [-MS] [-A] [-I3]

# 예시 (2구간 데이터)
>> python gradient_v2.py --fpath .\dir_data --opath .\output --r_cliff 1.0 --th_curv 1e-8 --s_degree 6 --i_degree 1 --n_degree 3 --alpha 1e-5 -A --t_estimate 50000
```

* `gradient_v2.py`의 옵션 상세 설명
  * `--fpath`: 분석할 파일들이 저장된 폴더의 경로
  * `--opath`: 분석 결과를 저장할 폴더의 경로
  * `--r_cliff`: 단차를 탐지하기 위한 임계치. 0~1 사이의 값을 가지며, 1로 설정할 경우 단차를 보정하지 않음.
  * `--th_curv`: 곡률 기반 직선 구간 탐지에 사용할 임계치. 1e-8과 같이 매우 작은 값을 설정해야 함.
  * `--s_degree`: 여러 데이터의 평균을 smoothing할 때 사용되는 기계학습 모델의 시작 복잡도 (3구간 데이터인 경우 50, 2구간 데이터인 경우 6을 권장).
  * `--i_degree`: `s_degree`부터 다음 복잡도 사이의 간격 (3구간 데이터인 경우 10, 2구간 데이터인 경우 1을 권장).
  * `--n_degree`: smoothing에 사용할 모델 복잡도의 수. `s_degree`와 `i_degree`와 함께 다음과 같이 사용됨: [`s_degree` + 0 x `i_degree`, `s_degree` + 1 x `i_degree`, ..., `s_degree` + (`n_degree`-1) x `i_degree`]
  * `--alpha`: smoothing에 사용되는 모델의 복잡도를 줄이기 위한 변수. 3구간/2구간 데이터 모두 `1e-5`를 권장.
  * `-A`: 해당 옵션을 추가하면 `--fpath` 경로에 저장된 데이터셋(csv)의 평균을 계산함.
  * `-I3`: 해당 옵션을 추가하면, 분석할 데이터가 3구간 데이터임을 알림.
  * `--t_estimate`: creep 값을 추정하려는 시간. `--fpath`에서 주어진 파일에 기록된 마지막 시간보다 이 옵션의 값이 작을 경우, creep 값을 추정하지 않음.

Output files
==
opath에 명시한 경로에 [파일명]-output.xlsx 파일이 생성됨.

![fig1](./figure/fig1.PNG)