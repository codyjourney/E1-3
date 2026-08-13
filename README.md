# NPU Simulator

3×3부터 25×25까지의 패턴과 필터를 이용하여 **MAC(Multiply-Accumulate) 연산을 직접 구현**하고, **Cross/X 패턴을 판별**하는 NPU 시뮬레이터입니다.

외부 라이브러리를 사용하지 않고 **Python 표준 라이브러리만 사용**하여 구현했습니다.

---

## 1. 개발 환경

* Python 3.12 이상
* 외부 라이브러리 사용하지 않음
* Python 표준 라이브러리 사용

  * `json`
  * `time`
  * `os`
  * `re`
* MAC 연산은 NumPy 등의 벡터화 기능을 사용하지 않고 **Python 반복문으로 직접 구현**

---

## 2. 파일 구조

```text
E1-3/
├── main.py
├── main2.py
├── data.json
└── README.md
```

### 파일 설명

| 파일          | 설명                         |
| ----------- | -------------------------- |
| `main.py`   | NPU Simulator 메인 프로그램 |
| `main2.py`  | 추가/보조 실행 프로그램              |
| `data.json` | 패턴 및 필터 테스트 데이터            |
| `README.md` | 프로젝트 설명 및 결과 분석            |

---

## 3. 실행 방법

터미널에서 프로젝트 폴더로 이동한 후 다음 명령어를 실행합니다.

```bash
python main.py
```

프로그램 실행 후 다음 중 하나를 선택합니다.

```text
1. 사용자 입력 (3x3)
2. data.json 분석
3. 종료
```

---

# 4. 프로젝트 소개

## 4.1 프로젝트 개요

이 프로젝트는 AI 반도체에서 사용되는 **MAC(Multiply-Accumulate) 연산의 원리**를 직접 Python으로 구현하고 실험해 보는 NPU 시뮬레이터입니다.

컴퓨터는 사람처럼 그림을 보고 직관적으로 `"십자가"` 또는 `"X"`라고 판단하지 않습니다.

대신 그림을 **0과 1로 이루어진 숫자 배열**로 표현하고, 미리 정의된 필터와 비교하여 어떤 패턴과 더 비슷한지 계산합니다.

예를 들어 십자가와 X를 다음과 같이 숫자로 표현할 수 있습니다.

### Cross

```text
0 1 0
1 1 1
0 1 0
```

### X

```text
1 0 1
0 1 0
1 0 1
```

입력 패턴과 필터를 같은 위치에 겹쳐 놓고 각각의 숫자를 곱한 뒤 모두 더하면 **MAC 점수**를 얻을 수 있습니다.

```text
입력 패턴
   ×
필터
   ↓
위치별 곱셈
   ↓
모든 값을 누적해서 더함
   ↓
MAC 점수
```

일반적으로 점수가 높을수록 입력 패턴과 해당 필터가 더 유사하다고 판단할 수 있습니다.

---

# 5. MAC 연산이란?

MAC은 다음 연산을 의미합니다.

* **Multiply**: 곱한다.
* **Accumulate**: 곱한 결과를 누적해서 더한다.

즉, MAC 연산은 다음과 같은 형태입니다.

```text
MAC = Σ(input[i][j] × filter[i][j])
```

예를 들어 Cross 패턴과 Cross 필터를 비교하면 다음과 같습니다.

### 입력 패턴

```text
0 1 0
1 1 1
0 1 0
```

### Cross 필터

```text
1 1 1
1 1 1
1 1 1
```

위치를 서로 곱하면:

```text
0×1  1×1  0×1
1×1  1×1  1×1
0×1  1×1  0×1
```

결과는:

```text
0 1 0
1 1 1
0 1 0
```

모든 값을 더하면:

```text
0 + 1 + 0
+ 1 + 1 + 1
+ 0 + 1 + 0
= 5
```

따라서 MAC 점수는 다음과 같습니다.

```text
Cross 필터 점수 = 5
```

반대로 X 필터를 적용하면 Cross 패턴과 일치하는 위치가 적기 때문에 상대적으로 낮은 점수가 나옵니다.

따라서 두 필터의 점수를 비교하여 입력 패턴을 판별할 수 있습니다.

```text
Cross 점수 > X 점수
→ Cross

X 점수 > Cross 점수
→ X
```

---

# 6. 왜 NPU가 필요한가?

실제 AI에서는 이러한 MAC 연산이 한두 번만 수행되는 것이 아닙니다.

이미지의 크기가 커지고 필터의 수가 많아지면 **수많은 곱셈과 덧셈을 반복**해야 합니다.

예를 들어 25×25 크기의 필터 하나를 계산하면:

```text
25 × 25 = 625
```

개의 위치에서 MAC 연산을 수행해야 합니다.

실제 AI 모델에서는 이러한 연산이 수천 개 이상의 필터와 여러 입력 데이터에 대해 반복될 수 있기 때문에 매우 많은 계산량이 발생합니다.

CPU는 다양한 종류의 복잡한 작업을 처리하는 데 강점이 있지만, AI에서는 동일한 형태의 곱셈과 덧셈을 매우 많이 수행해야 합니다.

그래서 이러한 AI 연산을 빠르게 처리하기 위한 전용 프로세서인 **NPU(Neural Processing Unit)** 등이 사용됩니다.

이 프로젝트에서는 실제 NPU 하드웨어를 구현하는 것이 아니라, NPU가 처리하는 핵심 연산 중 하나인 **MAC 연산을 Python 코드로 직접 구현**하여 원리를 이해하는 것을 목표로 합니다.

---

# 7. 프로젝트에서 무엇을 하는가?

프로젝트는 크게 두 가지 실행 모드로 구성됩니다.

## 7.1 사용자 입력 모드

사용자가 직접 **3×3 크기의 필터 A, 필터 B와 패턴**을 입력합니다.

예:

### 필터 A

```text
0 1 0
1 1 1
0 1 0
```

### 필터 B

```text
1 0 1
0 1 0
1 0 1
```

### 패턴

```text
0 1 0
1 1 1
0 1 0
```

프로그램은 각각의 필터와 패턴의 MAC 점수를 계산합니다.

```text
A 점수: 5
B 점수: 1

판정: A
```

즉, 입력 패턴이 A와 B 중 어느 필터와 더 유사한지 확인합니다.

---

## 7.2 data.json 분석 모드

미리 준비된 `data.json` 파일에서 여러 크기의 필터와 패턴을 읽어옵니다.

지원하는 주요 크기는 다음과 같습니다.

```text
3×3
5×5
13×13
25×25
```

각 패턴에 대해 다음 과정을 수행합니다.

1. 패턴 크기 확인
2. 해당 크기의 필터 선택
3. Cross 필터와 MAC 연산
4. X 필터와 MAC 연산
5. 두 점수 비교
6. `Cross`, `X`, `UNDECIDED` 중 하나로 판정
7. `expected` 값과 비교
8. `PASS` 또는 `FAIL` 출력

예:

```text
size_5_1

Cross 점수: 1.0
X 점수: 5.0

판정: X
expected: X
결과: PASS
```

---

# 8. 데이터 크기 지원

본 프로젝트에서는 다음과 같은 N×N 패턴을 사용합니다.

|    크기 | MAC 연산 횟수 |
| ----: | --------: |
|   3×3 |         9 |
|   5×5 |        25 |
| 13×13 |       169 |
| 25×25 |       625 |

N×N 배열에서 모든 위치를 한 번씩 계산하기 때문에 기본적인 MAC 연산량은 다음과 같습니다.

```text
N × N = N²
```

따라서 시간 복잡도는:

```text
O(N²)
```

입니다.

---

# 9. 라벨 정규화

## 9.1 라벨 정규화가 필요한 이유

`data.json`에서는 같은 의미의 라벨이 서로 다른 형태로 저장될 수 있습니다.

예를 들어 Cross는 다음과 같이 표현될 수 있습니다.

```text
Cross
cross
+
```

X 역시 다음과 같이 표현될 수 있습니다.

```text
X
x
```

따라서 프로그램 내부에서는 비교하기 전에 라벨을 하나의 표준 형태로 통일합니다.

### 표준 라벨

| 입력 라벨   | 정규화 결과  |
| ------- | ------- |
| `Cross` | `Cross` |
| `cross` | `Cross` |
| `+`     | `Cross` |
| `X`     | `X`     |
| `x`     | `X`     |

이 과정을 **라벨 정규화(Label Normalization)** 라고 합니다.

라벨을 정규화하면 데이터에 표현 방식이 조금 다르더라도 프로그램이 동일한 의미로 처리할 수 있습니다.

---

# 10. UNDECIDED와 epsilon

컴퓨터에서 실수를 계산하면 예상과 달리 아주 작은 오차가 발생할 수 있습니다.

예를 들어 사람이 보기에는 다음 두 값이 거의 같아도:

```text
A = 0.9000000000000000
B = 0.8999999999999999
```

컴퓨터에서는 두 값이 완전히 동일하지 않을 수 있습니다.

따라서 단순히 다음과 같이 비교하는 것은 적절하지 않을 수 있습니다.

```python
score_a == score_b
```

이 프로젝트에서는 **epsilon(허용오차)** 을 사용합니다.

```text
epsilon = 1e-9
```

두 점수의 차이가 다음 조건을 만족하면 동점으로 처리합니다.

```python
abs(score_a - score_b) < 1e-9
```

동점이면 어느 쪽이 더 높은지 결정할 수 없으므로:

```text
UNDECIDED
```

로 판정합니다.

판정 규칙은 다음과 같습니다.

```text
Cross > X
→ Cross

X > Cross
→ X

|Cross - X| < epsilon
→ UNDECIDED
```

이러한 정책을 사용하면 부동소수점 계산으로 인한 잘못된 판정을 줄일 수 있습니다.

---

# 11. 실행 결과

## 11.1 전체 실행 결과

현재 `data.json`을 분석한 결과는 다음과 같습니다.

```text
=======================================
NPU Simulator
=======================================

[모드 선택]

1. 사용자 입력 (3x3)
2. data.json 분석
3. 종료

선택: 2

data.json 경로 (기본값: data.json):
```

---

## 11.2 JSON 데이터 로드

```text
#---------------------------------------
# [1] JSON 데이터 로드
#---------------------------------------
```

---

## 11.3 필터 로드

```text
#---------------------------------------
# [2] 필터 로드
#---------------------------------------

✓ size_5 필터 로드 완료 (Cross, X)
✓ size_13 필터 로드 완료 (Cross, X)
✓ size_25 필터 로드 완료 (Cross, X)
```

---

## 11.4 패턴 분석 결과

```text
#---------------------------------------
# [3] 패턴 분석 (라벨 정규화 적용)
#---------------------------------------
```

### `size_5_1`

```text
Cross 점수: 0.9000000000
X 점수:     0.9000000000

판정: UNDECIDED
expected: X
FAIL

원인:
동점 규칙: |Cross-X| < 1e-09
```

### `size_5_2`

```text
Cross 점수: 8.9000000000
X 점수:     0.1000000000

판정: Cross
expected: Cross
PASS
```

### `size_13_1`

```text
Cross 점수: 0.3000000000
X 점수:     14.7000000000

판정: X
expected: X
PASS
```

### `size_13_2`

```text
Cross 점수: 7.5000000000
X 점수:     7.5000000000

판정: UNDECIDED
expected: Cross
FAIL

원인:
동점 규칙: |Cross-X| < 1e-09
```

### `size_25_1`

```text
Cross 점수: 4.9000000000
X 점수:     4.9000000000

판정: UNDECIDED
expected: X
FAIL

원인:
동점 규칙: |Cross-X| < 1e-09
```

### `size_25_2`

```text
Cross 점수: 52.9000000000
X 점수:     0.1000000000

판정: Cross
expected: Cross
PASS
```

---

# 12. 결과 요약

전체 테스트 결과는 다음과 같습니다.

| 항목    | 결과 |
| ----- | -: |
| 총 테스트 |  6 |
| 통과    |  3 |
| 실패    |  3 |

### 실패 케이스

| 테스트         | 결과   | 원인       |         |          |
| ----------- | ---- | -------- | ------- | -------- |
| `size_5_1`  | FAIL | 동점 규칙: ` | Cross-X | < 1e-09` |
| `size_13_2` | FAIL | 동점 규칙: ` | Cross-X | < 1e-09` |
| `size_25_1` | FAIL | 동점 규칙: ` | Cross-X | < 1e-09` |

현재 발생한 FAIL은 **MAC 계산 자체의 오류라기보다는 동점 처리 정책에 의해 발생한 결과**입니다.

즉, Cross와 X의 점수가 정확하게 동일하기 때문에 프로그램이 임의로 한쪽을 선택하지 않고 `UNDECIDED`를 반환하도록 설계되어 있습니다.

---

# 13. 성능 분석

이 프로젝트에서는 패턴 크기가 커질수록 MAC 연산 시간이 어떻게 증가하는지도 측정합니다.

각 크기에 대해 최소 10회 MAC 연산을 수행하고 평균 시간을 계산합니다.

측정 대상은 다음과 같습니다.

```text
3×3
5×5
13×13
25×25
```

## 13.1 성능 측정 결과

|    크기 | 평균 시간(ms) | 연산 횟수(N²) |
| ----: | --------: | --------: |
|   3×3 |  0.001591 |         9 |
|   5×5 |  0.002745 |        25 |
| 13×13 |  0.012842 |       169 |
| 25×25 |  0.043404 |       625 |

> 측정 시간은 실행 환경, CPU 성능, 운영체제, Python 실행 상태 등에 따라 달라질 수 있습니다.

---

# 14. 시간 복잡도

N×N 크기의 패턴에서는 각 위치에 대해 한 번씩 곱셈과 누적 연산을 수행합니다.

따라서 MAC 연산 횟수는:

```text
N × N = N²
```

입니다.

따라서 시간 복잡도는:

```text
O(N²)
```

입니다.

크기가 증가하면 다음과 같이 연산량이 증가합니다.

```text
3×3
→ 9회

5×5
→ 25회

13×13
→ 169회

25×25
→ 625회
```

즉, 패턴 크기가 커질수록 계산량이 빠르게 증가하는 것을 확인할 수 있습니다.

---

# 15. 프로젝트 핵심 동작 과정

전체적인 처리 과정은 다음과 같습니다.

```text
숫자 배열
   ↓
필터와 패턴 비교
   ↓
Multiply
   ↓
Accumulate
   ↓
MAC 점수
   ↓
Cross / X 점수 비교
   ↓
최종 판정
```

data.json 분석 모드에서는 다음과 같은 흐름으로 동작합니다.

```text
data.json 로드
   ↓
필터 검증
   ↓
패턴 로드
   ↓
패턴 크기 확인
   ↓
Cross / X 필터 선택
   ↓
MAC 연산
   ↓
Cross / X 점수 비교
   ↓
판정
   ↓
expected와 비교
   ↓
PASS / FAIL
   ↓
성능 분석
   ↓
전체 결과 요약
```

---

# 16. 오류가 발생했을 때 분석하는 방법

프로그램에서는 단순히 `FAIL`이라는 결과만 확인하는 것이 아니라, **왜 실패했는지 원인을 구분하는 것**도 중요합니다.

실패 원인은 크게 세 가지로 나눌 수 있습니다.

## 16.1 데이터 / 스키마 문제

예:

* 필터 크기와 패턴 크기가 다름
* 필수 키가 없음
* 잘못된 JSON 구조
* 잘못된 데이터 타입

이 경우 프로그램이 전체 실행을 중단하지 않고 해당 케이스를 `FAIL` 처리하도록 구성할 수 있습니다.

---

## 16.2 로직 문제

예:

* 잘못된 필터를 선택함
* MAC 계산 방식이 잘못됨
* Cross와 X의 판정 기준이 반대로 구현됨
* 패턴 크기에 맞지 않는 필터를 사용함

이 경우 프로그램의 알고리즘과 필터 선택 로직을 확인해야 합니다.

---

## 16.3 수치 비교 문제

예:

```text
0.9000000000000000
0.8999999999999999
```

처럼 매우 작은 부동소수점 차이 때문에 결과가 달라질 수 있습니다.

이 경우 epsilon을 이용한 허용오차 비교 정책을 적용할 수 있습니다.

---

# 17. 프로젝트의 핵심 목표

이 프로젝트의 핵심은 단순히 프로그램을 만드는 것이 아닙니다.

다음 과정을 직접 구현하고 실험하면서 **AI 연산의 기본 원리와 문제 해결 방법**을 이해하는 것이 목표입니다.

```text
숫자 배열
   ↓
패턴과 필터 비교
   ↓
Multiply
   ↓
Accumulate
   ↓
MAC 점수
   ↓
Cross / X 비교
   ↓
최종 판정
```

그리고 데이터 크기가 증가하면:

```text
작은 패턴
   ↓
적은 MAC 연산
   ↓
빠른 처리

큰 패턴
   ↓
많은 MAC 연산
   ↓
더 많은 처리 시간
```

이라는 관계를 직접 확인합니다.

---

# 18. 최종 프로그램 실행 흐름

프로그램을 실행하면 먼저 모드를 선택합니다.

```text
=== NPU Simulator ===

1. 사용자 입력 (3x3)
2. data.json 분석
3. 종료

선택:
```

## 모드 1: 사용자 입력

```text
필터 A 입력
   ↓
필터 B 입력
   ↓
패턴 입력
   ↓
MAC 연산
   ↓
A / B 점수 비교
   ↓
판정
   ↓
3×3 성능 측정
```

## 모드 2: data.json 분석

```text
data.json 로드
   ↓
필터 검증
   ↓
패턴 로드
   ↓
패턴 크기 확인
   ↓
Cross / X 필터 선택
   ↓
MAC 연산
   ↓
판정
   ↓
expected와 비교
   ↓
PASS / FAIL
   ↓
성능 분석
   ↓
전체 결과 요약
```

---

# 19. 최종적으로 확인하는 것

프로그램 실행이 끝나면 다음과 같은 결과를 확인할 수 있습니다.

```text
=== 결과 요약 ===

총 테스트: 6개
통과: 3개
실패: 3개
```

실패 케이스:

```text
- size_5_1
  → 동점 규칙에 의해 UNDECIDED

- size_13_2
  → 동점 규칙에 의해 UNDECIDED

- size_25_1
  → 동점 규칙에 의해 UNDECIDED
```

또한 성능 분석을 통해 패턴 크기가 커질수록 연산량과 처리 시간이 증가하는 현상을 확인할 수 있습니다.

|    크기 | 연산 횟수 |
| ----: | ----: |
|   3×3 |     9 |
|   5×5 |    25 |
| 13×13 |   169 |
| 25×25 |   625 |

---

# 20. 이 프로젝트를 통해 배우는 것

이 프로젝트를 완료하면 다음 내용을 설명할 수 있는 것을 목표로 합니다.

* MAC(Multiply-Accumulate) 연산이 무엇인지 이해한다.
* AI에서 MAC 연산이 왜 중요한지 이해한다.
* 숫자 배열을 이용해 패턴과 필터의 유사도를 계산하는 원리를 이해한다.
* Cross와 X 패턴을 MAC 점수로 구분하는 방법을 이해한다.
* `data.json`의 구조와 키 규칙을 해석할 수 있다.
* 서로 다른 라벨을 `Cross`, `X`로 정규화하는 이유를 이해한다.
* 부동소수점 오차가 판정에 미치는 영향을 이해한다.
* epsilon을 이용한 동점 처리 방법을 이해한다.
* N×N MAC 연산의 시간 복잡도가 `O(N²)`임을 이해한다.
* PASS/FAIL 결과가 발생했을 때 데이터 문제, 로직 문제, 수치 비교 문제를 구분하여 분석할 수 있다.
* CPU와 NPU가 왜 서로 다른 방식으로 설계되는지 기본적인 이유를 이해한다.

---

# 21. 프로젝트 구성

```text
E1-3/
├── main.py
├── data.json
└── README.md
```

### `main.py`

NPU Simulator의 메인 프로그램입니다.

### `data.json`

패턴과 필터를 저장한 테스트 데이터입니다.

### `README.md`

프로젝트 설명, 실행 방법, 결과 분석 및 시간 복잡도 등을 설명합니다.

---

# 22. 개발 환경 및 구현 조건

* Python 3.8 이상
* 외부 라이브러리 사용 금지
* Python 표준 라이브러리 사용
* `json`, `time`, `os`, `re` 등의 표준 라이브러리 사용
* NumPy 등 외부 수치 계산 라이브러리 사용 금지
* MAC 연산은 Python 반복문으로 직접 구현

예를 들어 MAC 연산은 개념적으로 다음과 같이 구현할 수 있습니다.

```python
score = 0

for i in range(size):
    for j in range(size):
        score += pattern[i][j] * filter[i][j]
```

이렇게 구현하면 NPU에서 수행하는 기본적인 MAC 연산의 원리를 직접 확인할 수 있습니다.

---

# 23. 한 문장으로 정리

> **NPU Simulator는 숫자 배열로 표현된 패턴과 필터를 MAC 연산으로 비교하여 Cross와 X를 판별하고, 데이터 크기가 증가할 때 AI 연산량과 처리 시간이 어떻게 증가하는지를 직접 구현하고 실험해 보는 교육용 NPU 시뮬레이터이다.**

---

# 24. 결론

이 프로젝트는 실제 NPU의 모든 기능을 구현하는 프로젝트가 아니라, NPU가 빠르게 처리하는 AI 연산의 가장 기본적인 단위인 **MAC 연산을 직접 구현해 보는 교육용 시뮬레이터**입니다.

작은 3×3 패턴에서 시작하여 5×5, 13×13, 25×25로 크기를 증가시키면서 연산 횟수와 실행 시간을 측정합니다.

이를 통해 단순한 숫자 곱셈과 덧셈이 실제 AI에서는 엄청난 규모로 반복된다는 사실을 확인할 수 있습니다.

또한 이러한 대규모 연산을 효율적으로 처리하기 위해 NPU와 같은 AI 전용 하드웨어가 사용되는 이유를 이해할 수 있습니다.

최종적으로 이 프로젝트를 통해 다음과 같은 흐름을 직접 구현하고 확인하는 것을 목표로 합니다.

```text
패턴 / 필터
     ↓
Multiply
     ↓
Accumulate
     ↓
MAC Score
     ↓
Cross / X 비교
     ↓
패턴 판정
     ↓
성능 측정
     ↓
연산량 증가 분석
```

즉, **작은 숫자 배열의 MAC 연산에서 시작하여 AI 반도체와 NPU의 기본적인 연산 구조까지 이해하는 것**이 이 프로젝트의 최종 목적입니다.




## 25. 설명

# NPU Simulator

## 1. 프로젝트 소개

이 프로젝트는 AI 연산에서 중요한 기본 연산 중 하나인 **MAC(Multiply-Accumulate)** 연산을 Python으로 직접 구현한 **NPU Simulator**이다.

NPU(Neural Processing Unit)는 인공지능 연산을 빠르게 처리하기 위한 프로세서이며, 특히 행렬 연산과 MAC 연산을 매우 빠르게 수행하도록 설계된다.

이 프로젝트에서는 실제 NPU 하드웨어를 구현하는 대신, Python의 반복문을 이용하여 MAC 연산을 직접 구현하고 다음 과정을 시뮬레이션한다.

```text
패턴 입력
   ↓
Cross 필터와 MAC 계산
   ↓
X 필터와 MAC 계산
   ↓
두 점수 비교
   ↓
Cross / X / UNDECIDED 판정
```

또한 `data.json`을 이용하면 여러 크기의 패턴을 자동으로 분석할 수 있으며, `3x3`, `5x5`, `13x13`, `25x25` 크기의 MAC 연산 성능도 측정한다.

---

# 2. 프로젝트의 핵심 개념

## 2.1 MAC이란?

MAC은 다음 두 단어의 앞 글자를 합친 것이다.

- **Multiply**: 곱하기
- **Accumulate**: 누적해서 더하기

즉, 두 행렬의 같은 위치에 있는 값을 곱하고 그 결과를 모두 더한다.

예를 들어 다음과 같은 두 행렬이 있다고 하자.

### Pattern

```text
0 1 0
1 1 1
0 1 0
```

### Filter

```text
0 1 0
1 1 1
0 1 0
```

같은 위치의 값을 곱한다.

```text
0×0 + 1×1 + 0×0
+ 1×1 + 1×1 + 1×1
+ 0×0 + 1×1 + 0×0
```

결과는 다음과 같다.

```text
0 + 1 + 0
+ 1 + 1 + 1
+ 0 + 1 + 0
= 5
```

이 값이 MAC 점수이다.

따라서 패턴과 필터가 비슷한 모양을 가지고 있을수록 높은 MAC 점수를 얻을 수 있다.

---

# 3. 프로그램 전체 구조

프로그램은 크게 두 가지 실행 모드를 제공한다.

```text
main()
 │
 ├─ 1번 선택
 │    ↓
 │  run_user_mode()
 │    ├─ 필터 A 입력
 │    ├─ 필터 B 입력
 │    ├─ 패턴 입력
 │    ├─ mac_score()
 │    ├─ classify_scores()
 │    └─ run_performance_analysis()
 │
 └─ 2번 선택
      ↓
    run_json_mode()
      ├─ load_json_file()
      ├─ validate_filter_group()
      ├─ analyze_pattern_case()
      │    ├─ normalize_label()
      │    ├─ validate_square_matrix()
      │    ├─ to_float_matrix()
      │    ├─ mac_score()
      │    └─ classify_scores()
      │
      ├─ PASS / FAIL 집계
      └─ run_performance_analysis()
```

---

# 4. 파일 구성

기본적으로 다음과 같은 구조로 사용할 수 있다.

```text
E1-3/
├── main.py
├── data.json
└── README.md
```

- `main.py`: NPU Simulator 프로그램
- `data.json`: 자동 분석에 사용할 패턴과 필터 데이터
- `README.md`: 프로젝트 설명 및 실행 방법

---

# 5. Python 기본 라이브러리 import

코드에서는 외부 라이브러리를 사용하지 않고 Python 기본 라이브러리만 사용한다.

```python
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
```

각 라이브러리의 역할은 다음과 같다.

---

## 5.1 `json`

```python
import json
```

JSON 파일을 읽고 Python 객체로 변환할 때 사용한다.

예를 들어 다음 JSON이 있다고 하자.

```json
{
    "name": "Cross",
    "size": 5
}
```

Python에서는 다음과 같이 읽을 수 있다.

```python
data = json.load(file)
```

그러면 Python의 dictionary 형태로 변환된다.

```python
{
    "name": "Cross",
    "size": 5
}
```

이 프로젝트에서는 `data.json`을 읽기 위해 사용한다.

---

## 5.2 `os`

```python
import os
```

운영체제와 관련된 기능을 제공한다.

이 프로젝트에서는 파일이 존재하는지 확인하는 데 사용한다.

```python
os.path.exists(json_path)
```

예를 들어:

```python
if not os.path.exists(json_path):
    print("파일을 찾을 수 없습니다.")
```

와 같이 사용할 수 있다.

---

## 5.3 `re`

```python
import re
```

정규표현식(Regular Expression)을 처리하는 라이브러리이다.

이 프로젝트에서는 패턴 이름이 다음 형식인지 검사한다.

```text
size_5_1
size_13_2
size_25_1
```

예를 들어:

```python
r"size_(\d+)_(\d+)"
```

라는 정규표현식을 이용한다.

---

## 5.4 `time`

```python
import time
```

시간을 측정할 때 사용한다.

이 프로젝트에서는 MAC 연산에 걸리는 시간을 측정한다.

```python
time.perf_counter()
```

를 사용하여 짧은 실행 시간을 비교적 정밀하게 측정한다.

---

## 5.5 `typing`

```python
from typing import Any, Dict, List, Optional, Tuple
```

Python 코드의 타입 정보를 명확하게 표현하기 위해 사용한다.

예:

```python
def normalize_label(label: Any) -> Optional[str]:
```

이 의미는 다음과 같다.

```text
label은 어떤 타입이든 받을 수 있고
반환값은 문자열이거나 None일 수 있다.
```

또 다른 예:

```python
def matrix_size(matrix: Any) -> Optional[Tuple[int, int]]:
```

이는 다음과 같은 의미이다.

```text
반환값:
    (행 개수, 열 개수)
또는
    None
```

---

# 6. 프로그램 설정 상수

코드에는 여러 설정값이 상수 형태로 정의되어 있다.

---

## 6.1 `EPSILON`

```python
EPSILON = 1e-9
```

부동소수점 숫자의 비교에서 사용하는 허용 오차이다.

컴퓨터에서는 다음 두 값이 미세하게 다를 수 있다.

```text
0.9
0.8999999999999999
```

사람이 보기에는 거의 같지만 컴퓨터 내부에서는 정확하게 같지 않을 수 있다.

따라서 다음과 같이 점수 차이가 매우 작으면 동점으로 처리한다.

```text
|Cross 점수 - X 점수| < EPSILON
```

이 경우 결과는 다음과 같다.

```text
UNDECIDED
```

---

## 6.2 `MIN_SIZE`

```python
MIN_SIZE = 3
```

프로그램에서 사용하는 최소 행렬 크기를 의미하기 위해 정의된 상수이다.

다만 현재 코드에서는 실제 계산에 직접 사용되지는 않는다.

즉, 현재 코드에서 실질적인 역할은 없다.

향후 입력 검증에서 최소 크기를 제한하고 싶다면 사용할 수 있다.

예:

```python
if size < MIN_SIZE:
    ...
```

---

## 6.3 `SUPPORTED_JSON_SIZES`

```python
SUPPORTED_JSON_SIZES = (5, 13, 25)
```

`data.json` 모드에서 지원하는 행렬 크기이다.

현재는 다음 세 가지 크기를 사용한다.

```text
5x5
13x13
25x25
```

---

## 6.4 `PERFORMANCE_REPEATS`

```python
PERFORMANCE_REPEATS = 10
```

성능 측정을 몇 번 반복할지 결정한다.

현재 값은 10이므로 MAC 연산을 10회 측정한 후 평균을 계산한다.

한 번만 측정하면 CPU 상태나 운영체제의 다른 작업에 영향을 받을 수 있기 때문에 여러 번 반복한다.

---

# 7. `normalize_label()`

```python
def normalize_label(label: Any) -> Optional[str]:
```

## 역할

입력된 라벨을 프로그램에서 사용할 표준 라벨로 변환한다.

프로그램 내부에서는 다음 두 가지 라벨만 사용한다.

```text
Cross
X
```

하지만 JSON에서는 여러 표현을 사용할 수 있다.

예:

```text
+
cross
Cross
x
X
```

이를 하나의 형식으로 통일한다.

---

## 변환 규칙

```text
"+"       → "Cross"
"cross"   → "Cross"
"Cross"   → "Cross"

"x"       → "X"
"X"       → "X"
```

알 수 없는 값은:

```text
None
```

을 반환한다.

---

## 내부 동작

```python
if not isinstance(label, str):
    return None
```

먼저 문자열인지 검사한다.

그 다음:

```python
value = label.strip().lower()
```

앞뒤 공백을 제거하고 소문자로 변환한다.

예:

```text
" Cross "
```

↓

```text
"cross"
```

이후:

```python
if value in ("+", "cross"):
    return "Cross"
```

Cross 계열의 라벨이면 `Cross`로 변환한다.

그리고:

```python
if value in ("x",):
    return "X"
```

X 계열이면 `X`로 변환한다.

---

# 8. `is_matrix()`

```python
def is_matrix(value: Any) -> bool:
```

## 역할

입력 데이터가 2차원 리스트인지 확인한다.

정상적인 예:

```python
[
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
]
```

잘못된 예:

```python
[0, 1, 0]
```

또는:

```python
"hello"
```

---

## 검사 과정

먼저 전체 값이 리스트인지 확인한다.

```python
if not isinstance(value, list) or len(value) == 0:
    return False
```

그 다음 내부의 모든 값이 리스트인지 검사한다.

```python
if not all(isinstance(row, list) for row in value):
    return False
```

모든 조건을 통과하면:

```python
return True
```

를 반환한다.

---

# 9. `matrix_size()`

```python
def matrix_size(matrix: Any) -> Optional[Tuple[int, int]]:
```

## 역할

행렬의 행과 열 개수를 계산한다.

예:

```python
[
    [1, 2, 3],
    [4, 5, 6]
]
```

결과:

```python
(2, 3)
```

즉:

```text
행 = 2
열 = 3
```

---

## 중요한 검증

모든 행의 열 개수가 같은지도 확인한다.

잘못된 예:

```python
[
    [1, 2, 3],
    [4, 5]
]
```

첫 번째 행은 3개이고 두 번째 행은 2개이므로 정상적인 행렬이 아니다.

이 경우:

```python
None
```

을 반환한다.

---

# 10. `validate_square_matrix()`

```python
def validate_square_matrix(
    matrix: Any,
    expected_size: Optional[int] = None
) -> Tuple[bool, str]:
```

## 역할

행렬이 정상적인 `N × N` 정사각형 행렬인지 검사한다.

예:

```text
3x3 → 정상
5x5 → 정상
3x4 → 실패
```

---

## 반환값

정상이면:

```python
(True, "")
```

문제가 있으면:

```python
(False, "실패 원인")
```

형태로 반환한다.

---

## 검사 내용

### 1. 행렬 구조 검사

```python
size = matrix_size(matrix)
```

행렬의 크기를 확인한다.

---

### 2. 정사각형인지 확인

```python
if rows != cols:
```

행과 열이 다르면 실패한다.

예:

```text
3x4
```

는 정사각형이 아니다.

---

### 3. 특정 크기인지 확인

`expected_size`가 전달된 경우 정확한 크기를 확인한다.

예:

```python
validate_square_matrix(matrix, expected_size=5)
```

그러면 반드시:

```text
5x5
```

여야 한다.

---

### 4. 숫자인지 확인

각 요소를:

```python
float(value)
```

로 변환해 본다.

예:

```text
1
0
0.5
"3"
```

등은 변환 가능하다.

반면:

```text
"hello"
```

는 숫자로 변환할 수 없기 때문에 실패한다.

---

# 11. `to_float_matrix()`

```python
def to_float_matrix(matrix: List[List[Any]]) -> List[List[float]]:
```

## 역할

행렬의 모든 값을 `float` 타입으로 변환한다.

예:

```python
[
    [1, 0],
    [0, 1]
]
```

↓

```python
[
    [1.0, 0.0],
    [0.0, 1.0]
]
```

JSON 데이터에 정수와 실수가 섞여 있을 수 있기 때문에 MAC 계산 전에 하나의 숫자 타입으로 통일한다.

---

# 12. `mac_score()`

```python
def mac_score(
    pattern: List[List[float]],
    filter_matrix: List[List[float]]
) -> float:
```

## 역할

이 프로그램의 **가장 핵심적인 함수**이다.

패턴과 필터의 같은 위치에 있는 값을 곱하고 모두 더한다.

---

## 계산 구조

```text
Pattern
    ↓
같은 위치의 Filter와 곱하기
    ↓
결과를 계속 누적
    ↓
최종 MAC Score
```

코드:

```python
score = 0.0

for r in range(rows):
    for c in range(rows):
        score += pattern[r][c] * filter_matrix[r][c]
```

---

## 예제

Pattern:

```text
1 0 1
0 1 0
1 0 1
```

Filter:

```text
1 1 0
0 1 0
1 0 1
```

계산:

```text
1×1
+ 0×1
+ 1×0
+ 0×0
+ 1×1
+ 0×0
+ 1×1
+ 0×0
+ 1×1
```

결과:

```text
4
```

---

## 시간 복잡도

`N × N` 행렬의 모든 위치를 한 번씩 검사한다.

따라서:

```text
N × N = N²
```

이고 시간 복잡도는:

```text
O(N²)
```

이다.

예:

| 크기 | MAC 연산 위치 수 |
|---|---:|
| 3x3 | 9 |
| 5x5 | 25 |
| 13x13 | 169 |
| 25x25 | 625 |

행렬 크기가 커질수록 계산량이 제곱 형태로 증가한다.

---

# 13. `classify_scores()`

```python
def classify_scores(
    cross_score: float,
    x_score: float,
    epsilon: float = EPSILON
) -> str:
```

## 역할

Cross 필터와 X 필터의 MAC 점수를 비교하여 최종 결과를 결정한다.

결과는 세 가지 중 하나이다.

```text
Cross
X
UNDECIDED
```

---

## 판정 규칙

### Cross 점수가 큰 경우

```text
Cross > X
```

결과:

```text
Cross
```

---

### X 점수가 큰 경우

```text
Cross < X
```

결과:

```text
X
```

---

### 두 점수가 거의 같은 경우

```text
|Cross - X| < EPSILON
```

결과:

```text
UNDECIDED
```

---

## 왜 `UNDECIDED`가 필요한가?

부동소수점 계산에서는 아주 작은 오차가 발생할 수 있다.

따라서 단순히:

```python
cross_score == x_score
```

로 비교하는 것보다 허용 오차를 사용하는 것이 안전하다.

---

# 14. `measure_mac()`

```python
def measure_mac(
    pattern: List[List[float]],
    filter_matrix: List[List[float]],
    repeats: int = PERFORMANCE_REPEATS
) -> Tuple[float, float]:
```

## 역할

MAC 연산에 걸리는 시간을 측정한다.

반환값:

```text
평균 실행 시간(ms)
마지막 MAC 결과
```

---

## `time.perf_counter()`

시간 측정에는:

```python
time.perf_counter()
```

를 사용한다.

실행 구조:

```python
start = time.perf_counter()

last_score = mac_score(
    pattern,
    filter_matrix
)

end = time.perf_counter()
```

그리고:

```python
(end - start) * 1000.0
```

를 통해 초 단위 시간을 밀리초(ms)로 변환한다.

---

## Warm-up

실제 측정 전에 MAC을 한 번 실행한다.

```python
last_score = mac_score(pattern, filter_matrix)
```

이 결과는 성능 측정값에는 포함하지 않는다.

첫 번째 실행에서 발생할 수 있는 초기화 영향을 줄이기 위한 것이다.

---

## 반복 측정

기본적으로:

```python
PERFORMANCE_REPEATS = 10
```

이므로 10회 측정한다.

예:

```text
0.010 ms
0.011 ms
0.009 ms
...
```

그리고 평균을 계산한다.

---

# 15. `create_performance_matrix()`

```python
def create_performance_matrix(size: int) -> List[List[float]]:
```

## 역할

성능 측정에 사용할 `N × N` 행렬을 생성한다.

모든 값을:

```text
1.0
```

으로 만든다.

예를 들어:

```python
create_performance_matrix(3)
```

결과:

```python
[
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0]
]
```

성능 측정에서는 실제 패턴의 정답 여부보다 동일한 크기의 MAC 연산량을 비교하는 것이 중요하기 때문에 단순한 행렬을 사용한다.

---

# 16. `run_performance_analysis()`

```python
def run_performance_analysis(
    sizes: Tuple[int, ...],
    repeats: int = PERFORMANCE_REPEATS
) -> None:
```

## 역할

여러 행렬 크기의 MAC 성능을 측정한다.

예:

```python
run_performance_analysis(
    (3, 5, 13, 25),
    10
)
```

이면:

```text
3x3
5x5
13x13
25x25
```

의 성능을 측정한다.

---

## 출력 정보

다음과 같은 정보를 출력한다.

```text
크기
평균 시간(ms)
연산 횟수(N²)
```

예:

```text
크기              평균 시간(ms)      연산 횟수(N²)
------------------------------------------------
3x3                  0.010000                    9
5x5                  0.020000                   25
13x13                0.100000                  169
25x25                0.400000                  625
```

실제 시간은 컴퓨터의 CPU와 Python 실행 환경에 따라 달라진다.

---

# 17. `read_matrix_from_console()`

```python
def read_matrix_from_console(
    size: int,
    matrix_name: str
) -> List[List[float]]:
```

## 역할

사용자로부터 행렬을 직접 입력받는다.

예를 들어 `3x3`이면:

```text
0 1 0
1 1 1
0 1 0
```

처럼 3개의 숫자를 3줄 입력한다.

---

## 입력 검증

각 행마다 정확하게 `size`개의 값이 입력되었는지 확인한다.

예:

```text
0 1
```

은 3x3 행렬에서 숫자가 2개뿐이므로 오류이다.

다음과 같은 입력도 오류이다.

```text
0 1 hello
```

`hello`를 숫자로 변환할 수 없기 때문이다.

---

## 잘못된 입력 처리

잘못된 입력이 들어오면:

```text
입력 형식 오류
```

를 출력하고 다시 입력받는다.

---

# 18. `run_user_mode()`

```python
def run_user_mode() -> None:
```

## 역할

프로그램의 **모드 1**을 담당한다.

사용자가 직접:

```text
필터 A
필터 B
패턴
```

을 입력한다.

---

## 실행 순서

```text
필터 A 입력
    ↓
필터 B 입력
    ↓
패턴 입력
    ↓
A와 패턴 MAC 계산
    ↓
B와 패턴 MAC 계산
    ↓
A/B 점수 비교
    ↓
Cross/X/UNDECIDED 판정
    ↓
성능 측정
```

---

## 핵심 코드

```python
score_a = mac_score(
    pattern,
    filter_a
)

score_b = mac_score(
    pattern,
    filter_b
)
```

두 필터와 패턴의 MAC 점수를 계산한다.

그리고:

```python
result = classify_scores(
    score_a,
    score_b,
    EPSILON
)
```

으로 최종 판정을 한다.

---

# 19. `load_json_file()`

```python
def load_json_file(path: str) -> Dict[str, Any]:
```

## 역할

JSON 파일을 읽어 Python dictionary로 변환한다.

코드:

```python
with open(path, "r", encoding="utf-8") as file:
    return json.load(file)
```

---

## `with open()`을 사용하는 이유

파일을 열고 사용한 후 자동으로 닫아준다.

따라서 다음과 같은 자원 관리 문제를 줄일 수 있다.

```text
파일 열기
 ↓
파일 사용
 ↓
파일 닫기
```

---

# 20. `extract_size_from_pattern_key()`

```python
def extract_size_from_pattern_key(
    key: str
) -> Optional[int]:
```

## 역할

패턴 이름에서 행렬 크기를 추출한다.

예:

```text
size_5_1
```

에서:

```text
5
```

를 추출한다.

---

## 사용하는 정규표현식

```python
r"size_(\d+)_(\d+)"
```

의 의미:

```text
size_
숫자
_
숫자
```

형태를 찾는다.

예:

```text
size_5_1
size_13_2
size_25_10
```

모두 허용된다.

---

## 반환값

```text
size_5_1
    ↓
5

size_13_2
    ↓
13

size_25_1
    ↓
25
```

잘못된 형식이면:

```python
None
```

을 반환한다.

---

# 21. `validate_filter_group()`

```python
def validate_filter_group(
    filters: Any,
    size: int
) -> Tuple[
    bool,
    str,
    Optional[Dict[str, List[List[float]]]]
]:
```

## 역할

`data.json`의 특정 크기 필터가 정상적으로 구성되어 있는지 검사한다.

예:

```text
size_5
size_13
size_25
```

각각에 다음 필터가 존재해야 한다.

```text
cross
x
```

---

## 기대 구조

```json
{
    "size_5": {
        "cross": [...],
        "x": [...]
    }
}
```

---

## 검사 과정

### 1. filters가 dictionary인지 확인

```python
if not isinstance(filters, dict):
```

---

### 2. `size_N`이 존재하는지 확인

예를 들어 size가 5이면:

```python
size_key = "size_5"
```

가 된다.

그리고:

```python
if size_key not in filters:
```

로 존재 여부를 확인한다.

---

### 3. `cross` 필터 확인

```python
if "cross" not in group:
```

필터가 없으면 오류이다.

---

### 4. `x` 필터 확인

```python
if "x" not in group:
```

필터가 없으면 오류이다.

---

### 5. 행렬 크기 확인

```python
validate_square_matrix(
    matrix,
    expected_size=size
)
```

을 사용하여 정확히 `N × N`인지 확인한다.

---

### 6. float 변환

정상적인 행렬이면:

```python
to_float_matrix(matrix)
```

를 이용해 숫자 타입을 통일한다.

---

## 최종 반환 형태

정상적으로 처리되면:

```python
{
    "Cross": [...],
    "X": [...]
}
```

형태로 반환한다.

---

# 22. `analyze_pattern_case()`

```python
def analyze_pattern_case(
    case_id: str,
    case_data: Any,
    filters: Dict[str, Any]
) -> Dict[str, Any]:
```

## 역할

`data.json`의 **패턴 하나를 분석하는 핵심 함수**이다.

예:

```text
size_5_1
```

이라는 하나의 테스트 케이스를 받아 다음을 모두 수행한다.

```text
키 형식 확인
    ↓
행렬 크기 확인
    ↓
expected 확인
    ↓
패턴 확인
    ↓
필터 확인
    ↓
Cross MAC
    ↓
X MAC
    ↓
판정
    ↓
expected와 비교
    ↓
PASS / FAIL
```

---

## 결과 dictionary

처음에는 다음과 같은 형태로 만든다.

```python
result = {
    "case_id": case_id,
    "status": "FAIL",
    "reason": "",
    "cross_score": None,
    "x_score": None,
    "prediction": None,
    "expected": None,
}
```

모든 검증이 정상적으로 끝나면:

```text
status = PASS
```

가 된다.

---

## expected 정규화

예를 들어:

```json
"expected": "+"
```

이면:

```text
Cross
```

로 변환된다.

```python
expected = normalize_label(
    case_data["expected"]
)
```

---

## 패턴 검증

```python
pattern = case_data["input"]
```

을 가져온 후:

```python
validate_square_matrix(
    pattern,
    expected_size=size
)
```

를 통해 크기를 확인한다.

---

## Cross MAC

```python
cross_score = mac_score(
    pattern,
    cross_filter
)
```

---

## X MAC

```python
x_score = mac_score(
    pattern,
    x_filter
)
```

---

## 최종 판정

```python
prediction = classify_scores(
    cross_score,
    x_score,
    EPSILON
)
```

---

## expected 비교

예를 들어:

```text
expected = Cross
prediction = Cross
```

이면:

```text
PASS
```

이다.

반대로:

```text
expected = Cross
prediction = X
```

이면:

```text
FAIL
```

이다.

---

# 23. `run_json_mode()`

```python
def run_json_mode(
    json_path: str = "data.json"
) -> None:
```

## 역할

프로그램의 **모드 2 전체를 담당한다.**

`data.json`을 읽고 모든 패턴을 자동으로 분석한다.

---

## 전체 실행 과정

```text
data.json 존재 여부 확인
        ↓
JSON 파일 읽기
        ↓
JSON 파싱
        ↓
최상위 구조 검사
        ↓
filters 검사
        ↓
patterns 검사
        ↓
필터 크기 검사
        ↓
모든 패턴 분석
        ↓
PASS / FAIL 집계
        ↓
성능 분석
        ↓
최종 결과 출력
```

---

## 파일 존재 여부

```python
if not os.path.exists(json_path):
```

파일이 존재하지 않으면 오류 메시지를 출력하고 종료한다.

---

## JSON 파싱 오류

```python
except json.JSONDecodeError as error:
```

JSON 문법이 잘못되었을 경우 처리한다.

예:

```json
{
    "name": "test",
}
```

처럼 마지막 쉼표 등의 문제로 JSON 문법이 잘못된 경우이다.

---

## 파일 읽기 오류

```python
except OSError as error:
```

파일 접근 권한이나 파일 시스템 문제 등을 처리한다.

---

# 24. JSON 최상위 구조 검사

`data.json`은 기본적으로 다음과 같은 구조를 가져야 한다.

```json
{
    "filters": {},
    "patterns": {}
}
```

코드에서는:

```python
filters = data.get("filters")
patterns = data.get("patterns")
```

으로 가져온다.

---

## `filters`

필터 데이터를 저장한다.

예:

```json
"filters": {
    "size_5": {
        "cross": [],
        "x": []
    }
}
```

---

## `patterns`

분석할 패턴 데이터를 저장한다.

예:

```json
"patterns": {
    "size_5_1": {
        "input": [],
        "expected": "+"
    }
}
```

---

# 25. PASS / FAIL 집계

`run_json_mode()`에서는 다음 변수를 사용한다.

```python
total = 0
passed = 0
failed = 0
```

각각의 의미는 다음과 같다.

| 변수 | 의미 |
|---|---|
| `total` | 전체 테스트 수 |
| `passed` | 성공한 테스트 수 |
| `failed` | 실패한 테스트 수 |

---

## 예

테스트가 10개이고:

```text
PASS = 8
FAIL = 2
```

라면:

```text
총 테스트: 10개
통과: 8개
실패: 2개
```

로 출력된다.

---

# 26. `failures`

```python
failures = []
```

실패한 테스트 케이스의 정보를 저장한다.

예:

```python
failures.append(result)
```

나중에:

```text
실패 케이스:
- size_5_2: 판정 불일치
- size_25_1: 패턴 크기 오류
```

처럼 출력할 수 있다.

---

# 27. `print_title()`

```python
def print_title() -> None:
```

## 역할

프로그램 실행 시 제목을 출력한다.

출력 예:

```text
=======================================
        NPU Simulator
=======================================
```

기능적으로 중요한 계산을 수행하는 함수는 아니며 사용자에게 프로그램의 시작 화면을 보여주는 역할을 한다.

---

# 28. `main()`

```python
def main() -> None:
```

## 역할

프로그램의 **최상위 실행 흐름을 관리하는 함수**이다.

사용자에게 메뉴를 보여주고 선택에 따라 알맞은 기능을 실행한다.

---

## 메뉴

```text
[모드 선택]
1. 사용자 입력 (3x3)
2. data.json 분석
0. 종료
```

---

## 1번 선택

```python
if choice == "1":
    run_user_mode()
    break
```

사용자 입력 모드를 실행한다.

---

## 2번 선택

```python
elif choice == "2":
```

JSON 분석 모드를 실행한다.

먼저:

```text
data.json 경로
```

를 입력받는다.

아무것도 입력하지 않으면:

```text
data.json
```

을 기본값으로 사용한다.

---

## 0번 선택

```python
elif choice == "0":
```

프로그램을 종료한다.

---

## 잘못된 입력

예:

```text
3
```

을 입력하면:

```text
입력 오류: 1, 2 또는 0을 선택하세요.
```

를 출력한다.

---

# 29. `if __name__ == "__main__":`

프로그램의 실제 시작점이다.

```python
if __name__ == "__main__":
    main()
```

Python 파일을 직접 실행하면:

```python
__name__ == "__main__"
```

이 된다.

따라서:

```python
main()
```

이 실행된다.

---

## 직접 실행하는 경우

```bash
python main.py
```

실행하면:

```text
main()
```

이 실행된다.

---

## 다른 파일에서 import하는 경우

예:

```python
import main
```

이 경우에는 `__name__`이 `"main"`이므로:

```python
if __name__ == "__main__":
```

조건이 거짓이 된다.

따라서 `main()`이 자동 실행되지 않는다.

이 구조를 사용하는 이유는 **파일을 직접 실행할 때와 다른 코드에서 모듈로 가져올 때의 동작을 구분하기 위해서이다.**

---

# 30. 프로그램에는 실제 `class`가 없다

이 코드에서 중요한 점은 **Python의 `class`가 사용되지 않았다는 것**이다.

즉:

```python
class NPU:
    ...
```

와 같은 클래스는 존재하지 않는다.

대신 여러 개의 함수가 각각 하나의 역할을 담당한다.

이를 **함수 중심 구조** 또는 **절차적 구조**라고 볼 수 있다.

---

# 31. 전체 함수 목록

현재 코드에 정의된 주요 함수는 다음과 같다.

| 함수 | 역할 |
|---|---|
| `normalize_label()` | 라벨을 `Cross` / `X`로 통일 |
| `is_matrix()` | 2차원 리스트인지 검사 |
| `matrix_size()` | 행렬 크기 확인 |
| `validate_square_matrix()` | N×N 정사각형 행렬인지 검증 |
| `to_float_matrix()` | 행렬 값을 float으로 변환 |
| `mac_score()` | MAC 연산 수행 |
| `classify_scores()` | Cross/X 점수 비교 |
| `measure_mac()` | MAC 실행 시간 측정 |
| `create_performance_matrix()` | 성능 측정용 행렬 생성 |
| `run_performance_analysis()` | 여러 크기의 성능 측정 |
| `read_matrix_from_console()` | 콘솔에서 행렬 입력 |
| `run_user_mode()` | 사용자 입력 모드 실행 |
| `load_json_file()` | JSON 파일 읽기 |
| `extract_size_from_pattern_key()` | 패턴 이름에서 크기 추출 |
| `validate_filter_group()` | JSON 필터 검증 |
| `analyze_pattern_case()` | 패턴 하나 분석 |
| `run_json_mode()` | JSON 전체 분석 |
| `print_title()` | 프로그램 제목 출력 |
| `main()` | 프로그램 전체 흐름 관리 |

---

# 32. 함수 간 관계

가장 중요한 함수 관계를 보면 다음과 같다.

```text
main()
 │
 ├── run_user_mode()
 │      │
 │      ├── read_matrix_from_console()
 │      │
 │      ├── mac_score()
 │      │
 │      ├── classify_scores()
 │      │
 │      ├── measure_mac()
 │      │
 │      └── run_performance_analysis()
 │
 └── run_json_mode()
        │
        ├── load_json_file()
        │
        ├── validate_filter_group()
        │
        ├── analyze_pattern_case()
        │      │
        │      ├── extract_size_from_pattern_key()
        │      ├── normalize_label()
        │      ├── validate_square_matrix()
        │      ├── to_float_matrix()
        │      ├── mac_score()
        │      └── classify_scores()
        │
        └── run_performance_analysis()
               │
               ├── create_performance_matrix()
               └── measure_mac()
```

---

# 33. 모드 1 전체 동작

사용자가 메뉴에서:

```text
1
```

을 선택하면 다음과 같이 동작한다.

```text
main()
 ↓
run_user_mode()
 ↓
필터 A 입력
 ↓
필터 B 입력
 ↓
패턴 입력
 ↓
mac_score(pattern, filter_a)
 ↓
A 점수
 ↓
mac_score(pattern, filter_b)
 ↓
B 점수
 ↓
classify_scores()
 ↓
Cross / X / UNDECIDED
 ↓
성능 측정
```

---

# 34. 모드 2 전체 동작

사용자가:

```text
2
```

를 선택하면:

```text
main()
 ↓
run_json_mode()
 ↓
data.json 읽기
 ↓
filters 확인
 ↓
patterns 확인
 ↓
각 패턴 반복
 ↓
analyze_pattern_case()
 ↓
패턴 크기 확인
 ↓
expected 정규화
 ↓
필터 확인
 ↓
Cross MAC
 ↓
X MAC
 ↓
판정
 ↓
expected 비교
 ↓
PASS / FAIL
```

---

# 35. `data.json`의 개념적인 구조

이 프로그램은 다음과 같은 형태의 JSON을 기대한다.

```json
{
    "filters": {
        "size_5": {
            "cross": [],
            "x": []
        },
        "size_13": {
            "cross": [],
            "x": []
        },
        "size_25": {
            "cross": [],
            "x": []
        }
    },
    "patterns": {
        "size_5_1": {
            "input": [],
            "expected": "+"
        },
        "size_13_1": {
            "input": [],
            "expected": "x"
        }
    }
}
```

실제 행렬 데이터는 각 크기에 맞게 들어가야 한다.

---

# 36. JSON 라벨 정규화

예를 들어 JSON에 다음과 같이 작성할 수 있다.

```json
"expected": "+"
```

프로그램에서는:

```text
+
 ↓
Cross
```

로 변환한다.

또한:

```json
"expected": "x"
```

는:

```text
x
 ↓
X
```

가 된다.

따라서 프로그램 내부에서는 항상 다음 두 값만 비교한다.

```text
Cross
X
```

---

# 37. 오류 처리 구조

이 프로그램의 중요한 특징 중 하나는 **잘못된 데이터가 있어도 가능한 경우 전체 프로그램을 종료하지 않는 것**이다.

예를 들어 10개의 테스트 케이스 중 하나의 데이터가 잘못되었다면:

```text
size_5_1 → PASS
size_5_2 → PASS
size_5_3 → FAIL
size_5_4 → PASS
...
```

처럼 해당 케이스만 실패 처리할 수 있다.

이것은 여러 테스트 데이터를 자동 검증할 때 중요한 구조이다.

---

# 38. PASS와 FAIL의 의미

## PASS

다음 조건을 모두 만족해야 한다.

```text
JSON 구조 정상
+
패턴 정상
+
필터 정상
+
MAC 계산 성공
+
Cross/X 판정 성공
+
prediction == expected
```

---

## FAIL

다음 중 하나라도 문제가 있으면 FAIL이다.

```text
패턴 키 오류
필터 없음
행렬 크기 오류
숫자가 아닌 데이터
expected 오류
JSON 구조 오류
예상 결과와 실제 결과 불일치
UNDECIDED 발생
```

---

# 39. 성능 분석의 의미

프로그램은 다음 크기를 비교한다.

```text
3x3
5x5
13x13
25x25
```

각각의 MAC 연산량은:

```text
3 × 3   = 9
5 × 5   = 25
13 × 13 = 169
25 × 25 = 625
```

이다.

따라서 행렬 크기가 증가하면 연산량이 빠르게 증가한다.

```text
N
↓
N²
```

이러한 특성 때문에 AI 연산에서는 MAC 연산을 빠르게 수행하는 하드웨어가 중요하다.

---

# 40. CPU Python 구현과 NPU의 차이

이 프로젝트의 `mac_score()`는 다음과 같은 단순한 구조이다.

```python
for r in range(rows):
    for c in range(rows):
        score += pattern[r][c] * filter_matrix[r][c]
```

즉, Python 프로그램이 순차적으로 연산한다.

반면 실제 NPU는 대규모 MAC 연산을 병렬적으로 처리하도록 설계된다.

개념적으로:

```text
Python CPU 방식

MAC
 ↓
MAC
 ↓
MAC
 ↓
MAC
```

보다:

```text
NPU 방식

MAC ─┐
MAC ─┤
MAC ─┤
MAC ─┤
MAC ─┘
 ↓
병렬 처리
```

에 가까운 구조를 목표로 한다.

이 프로젝트는 이러한 NPU의 핵심 연산 개념을 소프트웨어 수준에서 이해하기 위한 시뮬레이터이다.

---

# 41. 시간 복잡도

`mac_score()`는 `N × N`의 모든 위치를 방문한다.

```python
for r in range(rows):
    for c in range(rows):
```

따라서 시간 복잡도는:

```text
O(N²)
```

이다.

---

## 크기별 연산량

| 행렬 크기 | 연산량 |
|---|---:|
| 3x3 | 9 |
| 5x5 | 25 |
| 13x13 | 169 |
| 25x25 | 625 |

예를 들어 25x25 행렬은 3x3 행렬보다 훨씬 많은 MAC 연산을 수행한다.

---

# 42. 공간 복잡도

행렬 자체를 저장해야 하기 때문에 `N × N`개의 데이터가 필요하다.

따라서 입력 행렬 기준 공간 복잡도는:

```text
O(N²)
```

이다.

---

# 43. 이 코드에서 중요한 설계 포인트

## 43.1 검증과 계산을 분리

다음 함수들이 역할을 나누어 가진다.

```text
validate_square_matrix()
    ↓
데이터가 정상인지 확인

mac_score()
    ↓
실제 MAC 계산
```

따라서 잘못된 데이터를 계산 함수에 바로 전달하지 않도록 구성되어 있다.

---

## 43.2 라벨 정규화

다음과 같은 다양한 표현을:

```text
+
cross
Cross
x
X
```

프로그램 내부에서:

```text
Cross
X
```

두 가지로 통일한다.

---

## 43.3 오류 원인 저장

단순히:

```text
FAIL
```

만 출력하지 않고:

```python
result["reason"]
```

에 실패 원인을 저장한다.

따라서:

```text
FAIL
```

뿐만 아니라:

```text
판정 불일치
크기 오류
JSON 구조 오류
필터 없음
```

등을 확인할 수 있다.

---

## 43.4 성능 측정과 실제 판정 분리

성능 측정에서는:

```python
measure_mac()
```

을 사용한다.

실제 판정에서는:

```python
mac_score()
```

와:

```python
classify_scores()
```

를 사용한다.

즉, 성능 측정 코드가 실제 판정 로직과 섞이지 않도록 구성되어 있다.

---

# 44. 프로그램 실행 방법

Python 3.12 이상 환경에서 실행할 수 있다.

```bash
python main.py
```

실행하면:

```text
=======================================
        NPU Simulator
=======================================

[모드 선택]
1. 사용자 입력 (3x3)
2. data.json 분석
0. 종료
```

가 표시된다.

---

# 45. 모드 1 실행 예시

```text
[모드 선택]
1. 사용자 입력 (3x3)
2. data.json 분석
0. 종료

선택: 1
```

필터 A 입력:

```text
필터 A
1/3행 > 0 1 0
2/3행 > 1 1 1
3/3행 > 0 1 0
```

필터 B 입력:

```text
필터 B
1/3행 > 1 0 1
2/3행 > 0 1 0
3/3행 > 1 0 1
```

패턴 입력:

```text
패턴
1/3행 > 0 1 0
2/3행 > 1 1 1
3/3행 > 0 1 0
```

그러면:

```text
A 점수: ...
B 점수: ...

판정: Cross
```

와 같은 결과를 확인할 수 있다.

---

# 46. 모드 2 실행 예시

```text
[모드 선택]
1. 사용자 입력 (3x3)
2. data.json 분석
0. 종료

선택: 2
```

경로를 입력한다.

```text
data.json 경로 (기본값: data.json):
```

그냥 Enter를 누르면 현재 디렉토리의:

```text
data.json
```

을 사용한다.

---

# 47. JSON 모드 결과 예시

```text
#---------------------------------------
# [3] 패턴 분석 (라벨 정규화 적용)
#---------------------------------------

--- size_5_1 ---
Cross 점수: 10.0000000000
X 점수: 2.0000000000
판정: Cross | expected: Cross | PASS

--- size_13_1 ---
Cross 점수: 20.0000000000
X 점수: 30.0000000000
판정: X | expected: X | PASS
```

마지막에는:

```text
#---------------------------------------
# [5] 결과 요약
#---------------------------------------

총 테스트: 2개
통과: 2개
실패: 0개

실패 케이스가 없습니다.
```

와 같이 표시된다.

---

# 48. 코드에서 사용하는 주요 Python 문법

이 프로그램을 이해하려면 다음 Python 문법을 알아두면 좋다.

## 함수

```python
def function_name():
    ...
```

특정 기능을 하나의 이름으로 묶는다.

---

## 리스트

```python
[1, 2, 3]
```

여러 데이터를 순서대로 저장한다.

행렬도 리스트 안에 리스트를 넣어서 표현한다.

```python
[
    [1, 0],
    [0, 1]
]
```

---

## Dictionary

```python
{
    "name": "Cross",
    "size": 5
}
```

키와 값의 형태로 데이터를 저장한다.

JSON 데이터를 Python으로 읽으면 주로 dictionary 형태가 된다.

---

## `for`

```python
for r in range(rows):
```

반복문이다.

MAC 연산에서 행과 열을 순회할 때 사용한다.

---

## `if`

```python
if score_a > score_b:
```

조건에 따라 다른 코드를 실행한다.

---

## `try / except`

```python
try:
    ...
except ValueError:
    ...
```

오류가 발생했을 때 프로그램이 갑자기 종료되지 않도록 처리한다.

---

## `with`

```python
with open(path) as file:
```

파일 같은 자원을 안전하게 사용하고 자동으로 정리할 수 있다.

---

# 49. 타입 힌트 이해하기

코드에는 다음과 같은 타입 힌트가 많이 사용된다.

```python
def mac_score(
    pattern: List[List[float]],
    filter_matrix: List[List[float]]
) -> float:
```

이것을 쉽게 해석하면:

```text
pattern
→ float이 들어 있는 2차원 리스트

filter_matrix
→ float이 들어 있는 2차원 리스트

반환값
→ float
```

이라는 뜻이다.

---

## `Optional`

예:

```python
Optional[str]
```

은:

```text
str 또는 None
```

을 의미한다.

예:

```python
def normalize_label(...) -> Optional[str]:
```

은:

```text
"Cross"
"X"
```

같은 문자열을 반환할 수도 있고:

```python
None
```

을 반환할 수도 있다는 뜻이다.

---

## `Tuple`

예:

```python
Tuple[int, int]
```

은 두 개의 정수로 이루어진 튜플이다.

예:

```python
(3, 3)
```

---

# 50. 현재 코드에서 실제로 사용되지 않는 부분

코드를 분석할 때 다음 부분도 알아둘 필요가 있다.

## `MIN_SIZE`

```python
MIN_SIZE = 3
```

현재 선언되어 있지만 실제 함수에서 사용되지는 않는다.

향후 최소 크기 검증에 사용할 수 있다.

---

## `valid_filter_sizes`

`run_json_mode()`에서:

```python
valid_filter_sizes = []
```

를 만들고 정상적인 필터 크기를 저장하지만 이후 로직에서 적극적으로 사용하지는 않는다.

현재는 정보 기록용에 가깝다.

향후:

```python
if size not in valid_filter_sizes:
```

등의 검증에 활용할 수 있다.

---

# 51. 주석과 마지막 설명 문자열

코드 마지막 부분에는 다음과 같은 문자열이 있다.

```python
"""
main()
  │
  ├─ 모드 1 → run_user_mode()
  ...
"""
```

이것은 Python의 일반적인 실행 코드라기보다 **프로그램 구조를 설명하기 위한 문자열 문서**이다.

마찬가지로:

```python
"""
핵심은 mac_score() 하나
...
"""
```

와 같은 부분도 프로그램의 동작 구조를 설명하는 문서 역할을 한다.

실제 로직에 필요한 코드는 아니다.

---

# 52. 핵심 함수 3개

이 프로그램에서 특히 중요한 함수는 다음 세 가지이다.

## 1. `mac_score()`

```text
패턴 + 필터
    ↓
같은 위치끼리 곱하기
    ↓
모두 더하기
    ↓
MAC 점수
```

---

## 2. `classify_scores()`

```text
Cross 점수
      +
X 점수
      ↓
비교
      ↓
Cross / X / UNDECIDED
```

---

## 3. `analyze_pattern_case()`

```text
JSON 데이터
    ↓
검증
    ↓
MAC
    ↓
판정
    ↓
expected 비교
    ↓
PASS / FAIL
```

즉:

```text
mac_score()
    ↓
classify_scores()
    ↓
analyze_pattern_case()
```

순서로 이해하면 프로그램의 핵심 구조를 쉽게 파악할 수 있다.

---

# 53. 전체 데이터 흐름

최종적으로 프로그램의 데이터 흐름을 정리하면 다음과 같다.

```text
                 ┌─────────────────┐
                 │   사용자 입력    │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │     Pattern     │
                 └────────┬────────┘
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
          ┌──────────────┐  ┌──────────────┐
          │ Cross Filter │  │   X Filter   │
          └──────┬───────┘  └──────┬───────┘
                 │                 │
                 ▼                 ▼
          ┌──────────────┐  ┌──────────────┐
          │ Cross MAC    │  │ X MAC        │
          └──────┬───────┘  └──────┬───────┘
                 │                 │
                 └────────┬────────┘
                          ▼
                 ┌─────────────────┐
                 │ Score 비교      │
                 └────────┬────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
           Cross          X       UNDECIDED
```

JSON 모드에서는 여기에 `expected` 비교가 추가된다.

```text
data.json
   ↓
JSON 검증
   ↓
패턴 추출
   ↓
필터 선택
   ↓
Cross MAC
   ↓
X MAC
   ↓
분류
   ↓
expected 비교
   ↓
PASS / FAIL
```

---

# 54. 최종 정리

이 프로젝트는 단순한 행렬 계산 프로그램이 아니라 다음과 같은 AI 연산의 기본 구조를 직접 구현해 보는 프로젝트이다.

```text
행렬
 ↓
Multiply
 ↓
Accumulate
 ↓
MAC Score
 ↓
Score 비교
 ↓
Classification
```

특히 핵심 함수인:

```python
mac_score()
```

는 실제 AI 하드웨어에서 매우 중요한 MAC 연산을 Python의 기본 반복문으로 구현한다.

그리고:

```python
classify_scores()
```

가 MAC 결과를 이용하여 Cross 또는 X를 판정한다.

`data.json` 모드에서는:

```python
analyze_pattern_case()
```

가 데이터 검증부터 MAC 계산, 판정, `expected` 비교까지 하나의 테스트 케이스를 처리한다.

마지막으로:

```python
run_performance_analysis()
```

를 통해 행렬 크기가 증가할 때 MAC 연산량과 실행 시간이 어떻게 변하는지 확인할 수 있다.

---

# 55. 핵심 개념 한 줄 요약

| 개념 | 설명 |
|---|---|
| MAC | 같은 위치의 값을 곱한 후 모두 더하는 연산 |
| Pattern | 분석하려는 입력 행렬 |
| Filter | 패턴과 비교하는 기준 행렬 |
| Cross | Cross 모양을 나타내는 필터 |
| X | X 모양을 나타내는 필터 |
| `mac_score()` | Pattern과 Filter의 MAC 점수 계산 |
| `classify_scores()` | Cross와 X 점수 비교 |
| `UNDECIDED` | 두 점수가 사실상 동일한 경우 |
| `normalize_label()` | 다양한 라벨 표현을 표준화 |
| `validate_square_matrix()` | N×N 행렬인지 검증 |
| `measure_mac()` | MAC 실행 시간 측정 |
| `data.json` | 자동 테스트용 데이터 |
| `expected` | 데이터가 기대하는 정답 |
| `PASS` | 실제 판정과 expected가 일치 |
| `FAIL` | 검증 실패 또는 판정 불일치 |
| `main()` | 프로그램 전체 실행 흐름 관리 |

---

# 56. 한 문장으로 이해하기

> **NPU Simulator는 Python 반복문으로 MAC 연산을 직접 구현하고, Pattern과 Cross/X Filter의 유사도를 계산하여 패턴을 분류한 뒤, JSON 기반 자동 검증과 행렬 크기별 성능 측정까지 수행하는 프로그램이다.**


## 참고
https://chatgpt.com/share/6a7beb82-3104-83ea-8649-d73d1a243639