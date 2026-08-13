import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# NPU Simulator
# ============================================================
#
# 이 프로그램은 AI에서 중요한 연산 중 하나인
# MAC(Multiply-Accumulate)을 직접 구현해 보는
# "NPU Simulator"입니다.
#
# ------------------------------------------------------------
# MAC이란?
# ------------------------------------------------------------
#
# 패턴과 필터의 같은 위치에 있는 숫자를
# 각각 곱한 뒤 모두 더하는 연산입니다.
#
# 예를 들어,
#
# 패턴                 필터
# 0 1 0                0 1 0
# 1 1 1        ×       1 1 1
# 0 1 0                0 1 0
#
# 같은 위치의 숫자를 곱합니다.
#
# 0×0 + 1×1 + 0×0
# + 1×1 + 1×1 + 1×1
# + 0×0 + 1×1 + 0×0
#
# = 5
#
# 즉, 필터와 패턴이 비슷할수록 높은 점수가 나옵니다.
#
# 이 프로그램에서는 Cross 필터와 X 필터의 점수를 각각 계산한 뒤
# 어느 쪽 점수가 더 높은지를 이용하여 패턴을 판정합니다.
#
#
# 프로그램의 전체 흐름
# ------------------------------------------------------------
#
# 1. 프로그램 시작
# 2. 사용자가 모드를 선택
#
#    [모드 1]
#    사용자가 직접 3×3 필터 A/B와 패턴 입력
#
#    [모드 2]
#    data.json에 저장된 여러 크기의 필터와 패턴을 자동 분석
#
# 3. 패턴과 필터의 MAC 점수 계산
# 4. Cross / X 중 어느 쪽인지 판정
# 5. data.json 모드에서는 expected와 비교하여 PASS / FAIL
# 6. 3×3, 5×5, 13×13, 25×25의 연산 시간을 측정
# 7. 전체 결과를 콘솔에 출력
#
# ------------------------------------------------------------
# 개발 조건
# ------------------------------------------------------------
#
# - Python 3.8+
# - 외부 라이브러리 사용 금지
# - NumPy 사용 금지
# - MAC 연산은 직접 반복문으로 구현
#
# ============================================================


# ============================================================
# 프로그램 설정값
# ============================================================

# 부동소수점 숫자를 비교할 때 사용할 허용 오차입니다.
#
# 컴퓨터에서 실수 계산을 하다 보면
#
# 0.9
# 0.8999999999999999
#
# 처럼 사람이 보기에는 같은 값이지만
# 실제 컴퓨터 내부에서는 아주 작은 차이가 발생할 수 있습니다.
#
# 따라서 두 점수의 차이가 1e-9보다 작으면
# "거의 같은 값"이라고 판단합니다.
EPSILON = 1e-9


# 프로그램에서 최소한으로 처리할 수 있는 행렬 크기입니다.
#
# 현재 JSON 분석에서는 5×5, 13×13, 25×25를 사용하지만
# 프로그램의 기본 행렬 검증 함수에서는
# 특정 크기에만 제한하지 않고 일반적인 N×N 행렬을 처리합니다.
MIN_SIZE = 3


# data.json에서 지원하는 크기입니다.
#
# 요구사항에 따라
# 5×5
# 13×13
# 25×25
# 를 처리합니다.
SUPPORTED_JSON_SIZES = (5, 13, 25)


# 성능 측정 시 같은 MAC 연산을 최소 10번 반복합니다.
#
# 한 번만 측정하면 운영체제의 다른 작업이나
# 순간적인 CPU 상태 등의 영향을 받을 수 있기 때문에
# 여러 번 측정한 뒤 평균을 계산합니다.
PERFORMANCE_REPEATS = 10


# ============================================================
# 1. 라벨 정규화(Label Normalization)
# ============================================================
#
# data.json에는 같은 의미의 라벨이 여러 형태로 들어올 수 있습니다.
#
# 예:
#
# expected:
#   "+"
#   "x"
#
# filter:
#   "cross"
#   "x"
#
# 프로그램 내부에서는 이러한 여러 표현을 그대로 사용하지 않고
# 아래 두 가지 표준 라벨만 사용합니다.
#
#   Cross
#   X
#
# 이렇게 하나의 기준으로 통일하는 것을
# "라벨 정규화"라고 합니다.
#
# 정규화하면 다음과 같은 장점이 있습니다.
#
# "+"와 "Cross"가 같은 의미인지
# "x"와 "X"가 같은 의미인지
# 여러 곳에서 따로 처리할 필요가 없습니다.
#
# 프로그램 전체에서는 항상
#
#   Cross
#   X
#
# 만 사용하면 됩니다.
# ============================================================


def normalize_label(label: Any) -> Optional[str]:
    """
    입력 라벨을 프로그램 내부의 표준 라벨로 변환한다.

    --------------------------------------------------------
    입력 예시
    --------------------------------------------------------

    "+"       -> "Cross"
    "cross"   -> "Cross"
    "Cross"   -> "Cross"

    "x"       -> "X"
    "X"       -> "X"

    --------------------------------------------------------
    알 수 없는 값
    --------------------------------------------------------

    "circle"
    "triangle"
    None
    숫자

    등의 값이 들어오면 None을 반환한다.

    --------------------------------------------------------
    왜 필요한가?
    --------------------------------------------------------

    JSON의 expected 값과 filter의 키 이름이
    서로 다른 표현을 사용할 수 있기 때문입니다.

    프로그램 내부에서는 항상
    "Cross" 또는 "X"라는 동일한 표현을 사용합니다.
    """

    # 문자열이 아니라면 정상적인 라벨로 볼 수 없습니다.
    if not isinstance(label, str):
        return None

    # 앞뒤 공백을 제거하고 소문자로 통일합니다.
    #
    # 예:
    # " Cross " -> "cross"
    # " X "     -> "x"
    value = label.strip().lower()

    # "+" 또는 "cross"는 모두 Cross를 의미합니다.
    if value in ("+", "cross"):
        return "Cross"

    # "x"는 X를 의미합니다.
    if value in ("x",):
        return "X"

    # 그 외의 값은 알 수 없는 라벨입니다.
    return None


# ============================================================
# 2. 행렬(Matrix) 검증
# ============================================================
#
# 이 프로그램의 핵심 데이터는 2차원 배열입니다.
#
# 예:
#
# [
#     [0, 1, 0],
#     [1, 1, 1],
#     [0, 1, 0]
# ]
#
# 이것은 3×3 행렬입니다.
#
# JSON 데이터가 잘못되어 있거나
# 사용자가 잘못 입력했을 경우
# MAC 연산을 수행하기 전에 먼저 검증해야 합니다.
#
# 그렇지 않으면 프로그램이 중간에 오류가 발생하면서
# 전체 실행이 중단될 수 있습니다.
# ============================================================


def is_matrix(value: Any) -> bool:
    """
    전달받은 값이 2차원 리스트 형태인지 확인한다.

    정상적인 예:
        [
            [0, 1],
            [1, 0]
        ]

    잘못된 예:
        [0, 1, 0]

           "hello"

        None
    """

    # 전체 값이 리스트인지 확인합니다.
    if not isinstance(value, list) or len(value) == 0:
        return False

    # 리스트 안의 각각의 요소도 리스트인지 확인합니다.
    #
    # 예:
    #
    # [
    #   [0, 1, 0],
    #   [1, 1, 1],
    #   [0, 1, 0]
    # ]
    #
    # 각각의 행이 리스트이므로 정상입니다.
    if not all(isinstance(row, list) for row in value):
        return False

    return True


def matrix_size(matrix: Any) -> Optional[Tuple[int, int]]:
    """
    행렬의 크기를 반환한다.

    예:
        [
            [1, 2, 3],
            [4, 5, 6]
        ]

    -> (2, 3)

    즉,
        행 2개
        열 3개
    """

    # 먼저 2차원 리스트인지 확인합니다.
    if not is_matrix(matrix):
        return None

    rows = len(matrix)

    # 첫 번째 행의 길이를 이용하여 열의 개수를 구합니다.
    cols = len(matrix[0])

    # 열이 0개라면 정상적인 행렬이 아닙니다.
    if cols == 0:
        return None

    # 모든 행의 길이가 같은지 확인합니다.
    #
    # 예:
    #
    # [
    #   [1, 2, 3],
    #   [4, 5]
    # ]
    #
    # 이런 데이터는 행마다 열의 개수가 다르므로
    # 정상적인 직사각형 행렬이 아닙니다.
    if any(len(row) != cols for row in matrix):
        return None

    return rows, cols


def validate_square_matrix(
    matrix: Any,
    expected_size: Optional[int] = None
) -> Tuple[bool, str]:
    """
    정사각형 N×N 행렬인지 검사한다.

    반환값:
        (True, "")
        또는
        (False, "실패 원인")

    예:
        3×3 행렬
        -> (True, "")

        3×4 행렬
        -> (False, "정사각형이 아닙니다...")

    expected_size가 주어지면
    정확히 해당 크기인지도 검사합니다.
    """

    # 행렬의 행/열 크기를 가져옵니다.
    size = matrix_size(matrix)

    # 행렬 자체가 잘못되었으면 실패합니다.
    if size is None:
        return False, "2차원 배열이 아니거나 행의 길이가 서로 다릅니다."

    rows, cols = size

    # MAC 필터는 N×N 형태이므로
    # 행과 열의 개수가 같아야 합니다.
    if rows != cols:
        return False, f"정사각형이 아닙니다. 현재 크기: {rows}x{cols}"

    # 특정 크기를 요구하는 경우
    # 실제 크기와 비교합니다.
    if expected_size is not None and rows != expected_size:
        return (
            False,
            f"크기 불일치: 기대 크기 {expected_size}x{expected_size}, "
            f"실제 크기 {rows}x{cols}"
        )

    # 행렬의 모든 값이 숫자로 변환 가능한지 검사합니다.
    #
    # 예:
    # 1
    # 0.5
    # "3"
    #
    # 등은 float으로 변환할 수 있으므로 허용합니다.
    #
    # 하지만
    # "hello"
    #
    # 같은 값은 숫자가 아니므로 실패합니다.
    for r, row in enumerate(matrix):
        for c, value in enumerate(row):
            try:
                float(value)
            except (TypeError, ValueError):
                return (
                    False,
                    f"숫자가 아닌 값 발견: 위치 ({r}, {c}), 값={value!r}"
                )

    return True, ""


def to_float_matrix(matrix: List[List[Any]]) -> List[List[float]]:
    """
    행렬의 모든 값을 float으로 변환한다.

    JSON에서 숫자가 정수일 수도 있고
    문자열 형태의 숫자일 수도 있기 때문에
    MAC 계산 전에 모두 float으로 통일합니다.

    예:
        [[1, 0], [0, 1]]

    ->

        [[1.0, 0.0], [0.0, 1.0]]
    """

    return [
        [float(value) for value in row]
        for row in matrix
    ]


# ============================================================
# 3. MAC 연산
# ============================================================
#
# 이 프로그램의 가장 핵심적인 부분입니다.
#
# MAC = Multiply + Accumulate
#
# 즉,
#
# 1. Multiply
#    같은 위치의 값을 곱한다.
#
# 2. Accumulate
#    곱한 결과를 계속 누적해서 더한다.
#
# 예를 들어 3×3 행렬이라면
# 총 9번의 곱셈과 덧셈이 발생합니다.
#
# 이것이 실제 AI의 Convolution/Neural Network 연산에서
# 반복적으로 등장하는 기본적인 연산 구조입니다.
#
# 실제 NPU는 이런 연산을 매우 많이,
# 그리고 동시에 병렬적으로 처리하도록 설계됩니다.
#
# 이 프로젝트에서는 NPU 자체를 만드는 것이 아니라
# NPU가 빠르게 처리하는 핵심 연산인 MAC을
# CPU의 Python 반복문으로 직접 시뮬레이션합니다.
# ============================================================


def mac_score(
    pattern: List[List[float]],
    filter_matrix: List[List[float]]
) -> float:
    """
    패턴과 필터의 MAC 점수를 계산한다.

    계산식:

        score =
            pattern[0][0] * filter[0][0]
          + pattern[0][1] * filter[0][1]
          + ...

    즉, 같은 위치의 숫자끼리 곱하고
    그 결과를 모두 더합니다.

    --------------------------------------------------------
    예시
    --------------------------------------------------------

    pattern:

        0 1 0
        1 1 1
        0 1 0

    filter:

        0 1 0
        1 1 1
        0 1 0

    계산:

        0×0 + 1×1 + 0×0
      + 1×1 + 1×1 + 1×1
      + 0×0 + 1×1 + 0×0

      = 5

    --------------------------------------------------------
    시간 복잡도
    --------------------------------------------------------

    N×N 행렬의 모든 위치를 한 번씩 확인하므로

        N × N = N²

    입니다.

    따라서 시간 복잡도는

        O(N²)

    입니다.

    NumPy 등의 외부 라이브러리를 사용하지 않고
    직접 반복문으로 구현하는 것이 이 프로젝트의 핵심 조건입니다.
    """

    # 정사각형 행렬이므로 행의 개수를 크기로 사용합니다.
    rows = len(pattern)

    # MAC 결과를 저장할 변수입니다.
    #
    # 처음에는 아무것도 더하지 않았으므로 0입니다.
    score = 0.0

    # 모든 행을 순회합니다.
    for r in range(rows):

        # 각 행의 모든 열을 순회합니다.
        for c in range(rows):

            # 같은 위치의 값을 곱하고
            # 기존 score에 계속 더합니다.
            #
            # 이것이 바로
            #
            # Multiply + Accumulate
            #
            # 입니다.
            score += pattern[r][c] * filter_matrix[r][c]

    # 최종 MAC 점수를 반환합니다.
    return score


# ============================================================
# 4. 판정(Classification)
# ============================================================
#
# Cross 필터와 X 필터의 점수를 비교하여
# 어떤 패턴인지 결정합니다.
#
# 예:
#
# Cross 점수 = 8
# X 점수     = 2
#
# -> Cross
#
# 반대로
#
# Cross 점수 = 2
# X 점수     = 8
#
# -> X
#
# 그런데 두 점수가 거의 같다면?
#
# Cross = 0.9000000000000000
# X     = 0.8999999999999999
#
# 사람이 보기에는 거의 같은 값입니다.
#
# 이런 경우 단순히
#
#     cross_score == x_score
#
# 를 사용하면 원하는 결과가 나오지 않을 수 있습니다.
#
# 따라서 EPSILON을 사용하여
# 두 점수의 차이가 충분히 작은 경우
# "UNDECIDED"로 처리합니다.
# ============================================================


def classify_scores(
    cross_score: float,
    x_score: float,
    epsilon: float = EPSILON
) -> str:
    """
    Cross 점수와 X 점수를 비교하여
    Cross / X / UNDECIDED 중 하나를 반환한다.

    규칙:

        |Cross - X| < epsilon
            -> UNDECIDED

        Cross > X
            -> Cross

        Cross < X
            -> X
    """

    # 두 점수의 절대적인 차이를 계산합니다.
    difference = abs(cross_score - x_score)

    # 차이가 epsilon보다 작다면
    # 사실상 같은 점수로 간주합니다.
    if difference < epsilon:
        return "UNDECIDED"

    # Cross 점수가 더 높으면 Cross입니다.
    if cross_score > x_score:
        return "Cross"

    # 그 외에는 X 점수가 더 높다는 의미입니다.
    return "X"


# ============================================================
# 5. 성능 측정
# ============================================================
#
# 이 프로젝트에서는 단순히 "판정이 된다"에서 끝나지 않고
# 행렬 크기가 커질수록 연산 시간이 어떻게 변하는지도 확인합니다.
#
# 측정 대상:
#
#   3×3
#   5×5
#   13×13
#   25×25
#
# 각 크기에 대해 MAC 연산을 10회 반복하고
# 평균 시간을 계산합니다.
#
# 중요한 점은
# 파일 읽기, 콘솔 출력, 사용자 입력 등의 시간은 제외하고
# 실제 mac_score() 함수 실행 시간만 측정하는 것입니다.
# ============================================================


def measure_mac(
    pattern: List[List[float]],
    filter_matrix: List[List[float]],
    repeats: int = PERFORMANCE_REPEATS
) -> Tuple[float, float]:
    """
    MAC 연산만 반복 측정한다.

    반환값:

        (평균 시간(ms), 마지막 MAC 결과)

    --------------------------------------------------------
    왜 여러 번 반복하는가?
    --------------------------------------------------------

    한 번의 측정값은 운영체제나 CPU의 순간적인 상태 때문에
    약간 흔들릴 수 있습니다.

    따라서 같은 연산을 여러 번 수행하고
    평균값을 사용합니다.

    --------------------------------------------------------
    왜 perf_counter()를 사용하는가?
    --------------------------------------------------------

    time.perf_counter()는 짧은 실행 시간을 측정하기 위한
    고해상도 타이머입니다.
    """

    # 각 반복에서 측정된 시간을 저장합니다.
    elapsed_times = []

    # 마지막 MAC 결과를 저장합니다.
    last_score = 0.0

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------
    #
    # 실제 측정 전에 한 번 실행합니다.
    #
    # 첫 번째 실행에는 함수 호출이나 인터프리터 등의
    # 초기 영향이 포함될 수 있으므로
    # 측정 대상에서 제외합니다.
    last_score = mac_score(pattern, filter_matrix)

    # --------------------------------------------------------
    # 실제 성능 측정
    # --------------------------------------------------------

    for _ in range(repeats):

        # MAC 연산 직전에 시간 측정을 시작합니다.
        start = time.perf_counter()

        # 실제로 측정하고 싶은 부분입니다.
        last_score = mac_score(pattern, filter_matrix)

        # MAC 연산이 끝난 직후 시간 측정을 종료합니다.
        end = time.perf_counter()

        # 초 단위를 밀리초(ms) 단위로 변환합니다.
        #
        # 1초 = 1000ms
        elapsed_times.append(
            (end - start) * 1000.0
        )

    # 모든 측정값의 평균을 계산합니다.
    average_ms = sum(elapsed_times) / len(elapsed_times)

    return average_ms, last_score


def create_performance_matrix(size: int) -> List[List[float]]:
    """
    성능 측정용 N×N 행렬을 생성한다.

    모든 값을 1.0으로 설정합니다.

    예:
        size = 3

        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ]

    성능 측정에서는 특정 패턴의 정답 여부가 중요한 것이 아니라
    "N×N 크기의 MAC 연산이 얼마나 걸리는가"가 중요합니다.
    """

    return [
        [1.0 for _ in range(size)]
        for _ in range(size)
    ]


def run_performance_analysis(
    sizes: Tuple[int, ...],
    repeats: int = PERFORMANCE_REPEATS
) -> None:
    """
    지정된 행렬 크기들에 대해 MAC 성능을 측정하고 출력한다.

    출력 예:

        크기       평균 시간(ms)    연산 횟수(N²)
        ----------------------------------------
        3x3              0.010                 9
        5x5              0.020                25
        13x13            0.150               169
        25x25            0.600               625

    여기서 연산 횟수는

        N × N = N²

    입니다.
    """

    print("\n#---------------------------------------")
    print(f"# [성능 분석] 평균/{repeats}회")
    print("#---------------------------------------")

    # 표의 헤더를 출력합니다.
    print(
        f"{'크기':<12}"
        f"{'평균 시간(ms)':>18}"
        f"{'연산 횟수(N²)':>18}"
    )

    print("-" * 48)

    # 요청된 모든 크기를 하나씩 처리합니다.
    for size in sizes:

        # 성능 측정을 위한 패턴을 생성합니다.
        pattern = create_performance_matrix(size)

        # 성능 측정을 위한 필터를 생성합니다.
        filter_matrix = create_performance_matrix(size)

        # MAC 실행 시간을 측정합니다.
        average_ms, _ = measure_mac(
            pattern,
            filter_matrix,
            repeats
        )

        # N×N이므로 MAC 기본 연산 횟수는 N²입니다.
        operation_count = size * size

        # 결과를 표 형태로 출력합니다.
        print(
            f"{size}x{size:<8}"
            f"{average_ms:>18.6f}"
            f"{operation_count:>18}"
        )


# ============================================================
# 6. 사용자 입력 모드
# ============================================================
#
# 모드 1에서는 사용자가 직접 다음 데이터를 입력합니다.
#
# 필터 A
# 필터 B
# 패턴
#
# 모두 3×3 크기입니다.
#
# 입력된 데이터를 이용하여
#
# 1. A와 패턴의 MAC 점수
# 2. B와 패턴의 MAC 점수
# 3. 두 점수 비교
# 4. Cross / X / UNDECIDED 판정
# 5. 3×3 성능 측정
#
# 을 수행합니다.
# ============================================================


def read_matrix_from_console(
    size: int,
    matrix_name: str
) -> List[List[float]]:
    """
    콘솔에서 N×N 행렬을 한 줄씩 입력받는다.

    잘못된 입력이 들어오면 다시 입력하도록 합니다.

    예를 들어 3×3이면 한 줄에 숫자 3개를 입력해야 합니다.

        0 1 0
        1 1 1
        0 1 0

    잘못된 입력:

        0 1

    -> 숫자가 2개뿐이므로 재입력

    잘못된 입력:

        0 1 hello

    -> 숫자가 아니므로 재입력
    """

    # 행렬 전체가 정상적으로 입력될 때까지 반복합니다.
    while True:

        print(
            f"\n{matrix_name} "
            f"({size}줄 입력, 공백 구분)"
        )

        matrix = []

        # 현재 행렬 입력이 정상인지 기록하는 변수입니다.
        valid = True

        # size개의 행을 입력받습니다.
        for row_index in range(size):

            # 현재 행이 올바르게 입력될 때까지 반복합니다.
            while True:

                text = input(
                    f"{row_index + 1}/{size}행 > "
                ).strip()

                # 공백을 기준으로 문자열을 나눕니다.
                #
                # "0 1 0"
                #
                # -> ["0", "1", "0"]
                parts = text.split()

                # 정확히 size개의 숫자가 들어왔는지 검사합니다.
                if len(parts) != size:
                    print(
                        f"입력 형식 오류: "
                        f"각 줄에 {size}개의 숫자를 "
                        f"공백으로 구분해 입력하세요."
                    )
                    continue

                # 각 값을 float으로 변환합니다.
                #
                # 숫자가 아닌 값이 들어오면
                # ValueError가 발생합니다.
                try:
                    row = [
                        float(value)
                        for value in parts
                    ]

                except ValueError:
                    print(
                        "입력 형식 오류: 숫자만 입력하세요."
                    )
                    continue

                # 정상적으로 변환된 행을 저장합니다.
                matrix.append(row)

                break

        # 모든 행 입력이 끝난 후
        # 최종적으로 행렬 구조를 다시 검사합니다.
        ok, reason = validate_square_matrix(
            matrix,
            expected_size=size
        )

        # 정상적인 행렬이면 반환합니다.
        if ok:
            return matrix

        # 문제가 있으면 원인을 출력합니다.
        print(f"입력 오류: {reason}")
        print("행렬을 처음부터 다시 입력해주세요.")


def run_user_mode() -> None:
    """
    모드 1을 실행한다.

    전체 흐름:

        필터 A 입력
            ↓
        필터 B 입력
            ↓
        패턴 입력
            ↓
        MAC 점수 계산
            ↓
        Cross/X 판정
            ↓
        3×3 성능 분석
    """

    # 사용자 모드는 요구사항상 3×3만 사용합니다.
    size = 3

    # --------------------------------------------------------
    # 필터 입력
    # --------------------------------------------------------

    print("\n#---------------------------------------")
    print("# [1] 필터 입력")
    print("#---------------------------------------")

    # 필터 A를 입력받습니다.
    filter_a = read_matrix_from_console(
        size,
        "필터 A"
    )

    # 필터 B를 입력받습니다.
    filter_b = read_matrix_from_console(
        size,
        "필터 B"
    )

    print("\n필터 A 저장 완료.")
    print("필터 B 저장 완료.")

    # --------------------------------------------------------
    # 패턴 입력
    # --------------------------------------------------------

    print("\n#---------------------------------------")
    print("# [2] 패턴 입력")
    print("#---------------------------------------")

    # 비교할 실제 패턴을 입력받습니다.
    pattern = read_matrix_from_console(
        size,
        "패턴"
    )

    print("\n패턴 저장 완료.")

    # --------------------------------------------------------
    # MAC 결과
    # --------------------------------------------------------

    print("\n#---------------------------------------")
    print("# [3] MAC 결과")
    print("#---------------------------------------")

    # 패턴과 필터 A의 MAC 점수를 계산합니다.
    score_a = mac_score(
        pattern,
        filter_a
    )

    # 패턴과 필터 B의 MAC 점수를 계산합니다.
    score_b = mac_score(
        pattern,
        filter_b
    )

    # --------------------------------------------------------
    # 연산 시간 측정
    # --------------------------------------------------------
    #
    # A 필터에 대한 MAC 시간을 측정하고
    # B 필터에 대한 MAC 시간도 측정합니다.
    #
    # 두 결과의 평균을 사용자에게 보여줍니다.
    # --------------------------------------------------------

    average_a_ms, _ = measure_mac(
        pattern,
        filter_a,
        PERFORMANCE_REPEATS
    )

    average_b_ms, _ = measure_mac(
        pattern,
        filter_b,
        PERFORMANCE_REPEATS
    )

    # A와 B의 측정 시간을 평균냅니다.
    average_ms = (
        average_a_ms + average_b_ms
    ) / 2.0

    # --------------------------------------------------------
    # 최종 판정
    # --------------------------------------------------------

    result = classify_scores(
        score_a,
        score_b,
        EPSILON
    )

    # 점수를 출력합니다.
    print(f"A 점수: {score_a:.10f}")
    print(f"B 점수: {score_b:.10f}")

    # 평균 연산 시간을 출력합니다.
    print(
        f"연산 시간(평균/{PERFORMANCE_REPEATS}회): "
        f"{average_ms:.6f} ms"
    )

    # 동점이면 별도의 안내 문구를 출력합니다.
    if result == "UNDECIDED":

        print(
            f"판정: 판정 불가 "
            f"(|A-B| < {EPSILON})"
        )

    else:

        # Cross 또는 X 결과를 출력합니다.
        print(f"판정: {result}")

    # --------------------------------------------------------
    # 3×3 성능 분석
    # --------------------------------------------------------

    print("\n#---------------------------------------")
    print("# [4] 성능 분석")
    print("#---------------------------------------")

    # 사용자 입력 모드에서는 3×3 성능만 측정합니다.
    run_performance_analysis(
        (3,),
        PERFORMANCE_REPEATS
    )


# ============================================================
# 7. JSON 파일 읽기
# ============================================================
#
# 모드 2에서는 사용자가 직접 데이터를 입력하지 않습니다.
#
# 대신 data.json이라는 파일에서
#
# filters
# patterns
#
# 데이터를 읽습니다.
#
# 예를 들어 JSON의 개념적인 구조는 다음과 같습니다.
#
# {
#     "filters": {
#         "size_5": {
#             "cross": [...],
#             "x": [...]
#         },
#         ...
#     },
#
#     "patterns": {
#         "size_5_1": {
#             "input": [...],
#             "expected": "+"
#         },
#         ...
#     }
# }
#
# 프로그램은 이 데이터를 읽어서
# 각각의 패턴을 자동으로 판정합니다.
# ============================================================


def load_json_file(path: str) -> Dict[str, Any]:
    """
    JSON 파일을 읽어 Python 객체로 변환한다.

    json.load()를 사용하면
    JSON의 객체는 Python의 dict,
    JSON의 배열은 Python의 list로 변환됩니다.
    """

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_size_from_pattern_key(
    key: str
) -> Optional[int]:
    """
    패턴 키에서 행렬 크기 N을 추출한다.

    요구되는 키 형식:

        size_{N}_{idx}

    예:

        size_5_1
            -> 5

        size_13_2
            -> 13

        size_25_1
            -> 25

    여기서 추출한 숫자를 이용하여
    해당 크기의 필터를 선택합니다.
    """

    # 정규표현식으로
    #
    # size_숫자_숫자
    #
    # 형태인지 검사합니다.
    match = re.fullmatch(
        r"size_(\d+)_(\d+)",
        key
    )

    # 형식이 맞지 않으면 None을 반환합니다.
    if not match:
        return None

    # 첫 번째 숫자 그룹이 N입니다.
    return int(match.group(1))


# ============================================================
# 8. JSON 필터 검증
# ============================================================
#
# data.json의 filters에는
#
# size_5
# size_13
# size_25
#
# 각각의 그룹이 존재해야 합니다.
#
# 그리고 각 그룹에는
#
# cross
# x
#
# 필터가 있어야 합니다.
#
# 예:
#
# "size_5": {
#     "cross": [...],
#     "x": [...]
# }
#
# 이 함수는 이런 구조가 정상인지 확인합니다.
# ============================================================


def validate_filter_group(
    filters: Any,
    size: int
) -> Tuple[
    bool,
    str,
    Optional[Dict[str, List[List[float]]]]
]:
    """
    size_N 필터 그룹의 스키마와 행렬 크기를 검사한다.

    성공하면:

        True,
        "",
        {
            "Cross": Cross 필터,
            "X": X 필터
        }

    형태를 반환합니다.

    실패하면:

        False,
        실패 원인,
        None

    을 반환합니다.
    """

    # filters 전체가 객체(dict)인지 확인합니다.
    if not isinstance(filters, dict):
        return (
            False,
            "filters가 객체가 아닙니다.",
            None
        )

    # 예:
    #
    # size = 5
    #
    # -> "size_5"
    size_key = f"size_{size}"

    # 해당 크기의 필터가 존재하는지 확인합니다.
    if size_key not in filters:
        return (
            False,
            f"{size_key} 필터가 존재하지 않습니다.",
            None
        )

    # 해당 크기의 필터 그룹을 가져옵니다.
    group = filters[size_key]

    # 필터 그룹도 객체(dict)여야 합니다.
    if not isinstance(group, dict):
        return (
            False,
            f"{size_key} 값이 객체가 아닙니다.",
            None
        )

    # 정규화된 필터를 저장할 공간입니다.
    normalized = {}

    # JSON에서는 cross / x라는 키를 사용한다고 가정합니다.
    for raw_label in ("cross", "x"):

        # 필요한 필터 키가 없으면 실패합니다.
        if raw_label not in group:
            return (
                False,
                f"{size_key}에 '{raw_label}' 필터가 없습니다.",
                None
            )

        # raw_label을 프로그램 표준 라벨로 변환합니다.
        #
        # cross -> Cross
        # x     -> X
        label = normalize_label(raw_label)

        # 실제 필터 행렬을 가져옵니다.
        matrix = group[raw_label]

        # 행렬이 올바른 N×N인지 검사합니다.
        ok, reason = validate_square_matrix(
            matrix,
            expected_size=size
        )

        if not ok:
            return (
                False,
                f"{size_key}/{raw_label}: {reason}",
                None
            )

        # 숫자를 float으로 변환한 뒤 저장합니다.
        normalized[label] = to_float_matrix(matrix)

    # 최종적으로
    #
    # {
    #     "Cross": [...],
    #     "X": [...]
    # }
    #
    # 형태를 반환합니다.
    return True, "", normalized


# ============================================================
# 9. JSON의 개별 패턴 분석
# ============================================================
#
# data.json의 patterns에는 여러 테스트 케이스가 있습니다.
#
# 예:
#
# size_5_1
# size_5_2
# size_13_1
# size_25_1
#
# 각각을 하나의 테스트 케이스라고 생각하면 됩니다.
#
# 이 함수는 테스트 케이스 하나를 받아서
#
# 1. 키 형식 검사
# 2. 크기 검사
# 3. expected 검사
# 4. 패턴 크기 검사
# 5. 해당 크기의 필터 선택
# 6. Cross MAC 계산
# 7. X MAC 계산
# 8. 판정
# 9. expected와 비교
#
# 를 수행합니다.
#
# 중요한 점:
#
# 데이터 하나가 잘못되었다고 해서
# 프로그램 전체가 종료되면 안 됩니다.
#
# 따라서 오류가 발생해도
# 해당 케이스만 FAIL 처리하고
# 다음 케이스로 넘어갈 수 있도록 설계합니다.
# ============================================================


def analyze_pattern_case(
    case_id: str,
    case_data: Any,
    filters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    하나의 JSON 패턴 케이스를 분석한다.

    반환되는 결과에는
    점수, 판정, expected, PASS/FAIL, 실패 원인 등이 포함됩니다.
    """

    # --------------------------------------------------------
    # 결과 객체의 기본값
    # --------------------------------------------------------
    #
    # 처음에는 일단 FAIL 상태로 시작합니다.
    #
    # 모든 검증과 계산이 정상적으로 끝났을 때만
    # PASS로 변경합니다.
    result = {
        "case_id": case_id,
        "status": "FAIL",
        "reason": "",
        "cross_score": None,
        "x_score": None,
        "prediction": None,
        "expected": None,
    }

    # --------------------------------------------------------
    # 1. 패턴 키에서 크기 추출
    # --------------------------------------------------------

    size = extract_size_from_pattern_key(case_id)

    # 키 형식이 잘못되었다면 해당 케이스만 FAIL입니다.
    if size is None:
        result["reason"] = (
            "패턴 키 형식 오류: "
            "size_{N}_{idx} 형식이 필요합니다."
        )
        return result

    # --------------------------------------------------------
    # 2. 지원하는 크기인지 확인
    # --------------------------------------------------------

    if size not in SUPPORTED_JSON_SIZES:
        result["reason"] = (
            f"지원하지 않는 크기입니다: {size}x{size}"
        )
        return result

    # --------------------------------------------------------
    # 3. 패턴 데이터 구조 확인
    # --------------------------------------------------------

    if not isinstance(case_data, dict):
        result["reason"] = (
            "패턴 데이터가 객체가 아닙니다."
        )
        return result

    # input 필드가 반드시 필요합니다.
    if "input" not in case_data:
        result["reason"] = (
            "'input' 필드가 없습니다."
        )
        return result

    # expected 필드도 반드시 필요합니다.
    if "expected" not in case_data:
        result["reason"] = (
            "'expected' 필드가 없습니다."
        )
        return result

    # --------------------------------------------------------
    # 4. expected 라벨 정규화
    # --------------------------------------------------------
    #
    # 예:
    #
    # "+"
    # -> Cross
    #
    # "x"
    # -> X
    expected = normalize_label(
        case_data["expected"]
    )

    # 알 수 없는 라벨이면 FAIL입니다.
    if expected is None:
        result["reason"] = (
            f"알 수 없는 expected 라벨: "
            f"{case_data['expected']!r}"
        )
        return result

    # 표준화된 expected 값을 저장합니다.
    result["expected"] = expected

    # --------------------------------------------------------
    # 5. 패턴 행렬 검증
    # --------------------------------------------------------

    pattern = case_data["input"]

    # 패턴이 N×N인지 확인합니다.
    ok, reason = validate_square_matrix(
        pattern,
        expected_size=size
    )

    if not ok:
        result["reason"] = (
            f"패턴 크기/데이터 오류: {reason}"
        )
        return result

    # 패턴의 모든 값을 float으로 변환합니다.
    pattern = to_float_matrix(pattern)

    # --------------------------------------------------------
    # 6. 해당 크기의 필터 가져오기
    # --------------------------------------------------------
    #
    # 예:
    #
    # case_id = size_13_2
    #
    # -> size = 13
    #
    # -> filters["size_13"]
    #
    # 에서 Cross/X 필터를 가져옵니다.
    ok, reason, normalized_filters = (
        validate_filter_group(
            filters,
            size
        )
    )

    # 필터에 문제가 있으면 해당 케이스만 FAIL입니다.
    if not ok:
        result["reason"] = (
            f"필터 오류: {reason}"
        )
        return result

    # 표준 라벨을 이용해 필터를 가져옵니다.
    cross_filter = normalized_filters["Cross"]
    x_filter = normalized_filters["X"]

    # --------------------------------------------------------
    # 7. Cross 필터 MAC 계산
    # --------------------------------------------------------

    cross_score = mac_score(
        pattern,
        cross_filter
    )

    # --------------------------------------------------------
    # 8. X 필터 MAC 계산
    # --------------------------------------------------------

    x_score = mac_score(
        pattern,
        x_filter
    )

    # --------------------------------------------------------
    # 9. 두 점수 비교
    # --------------------------------------------------------

    prediction = classify_scores(
        cross_score,
        x_score,
        EPSILON
    )

    # 계산 결과를 result에 저장합니다.
    result["cross_score"] = cross_score
    result["x_score"] = x_score
    result["prediction"] = prediction

    # --------------------------------------------------------
    # 10. expected와 실제 판정 비교
    # --------------------------------------------------------

    if prediction == expected:

        # 예측 결과와 정답이 같다면 PASS입니다.
        result["status"] = "PASS"
        result["reason"] = "정상 판정"

    else:

        # 결과가 다르면 FAIL입니다.
        result["status"] = "FAIL"

        # 특히 UNDECIDED인 경우는
        # epsilon 동점 규칙 때문에 발생한 것인지 알려줍니다.
        if prediction == "UNDECIDED":

            result["reason"] = (
                f"동점 규칙: "
                f"|Cross-X| < {EPSILON}"
            )

        else:

            # Cross/X 중 하나가 나왔지만
            # expected와 다른 경우입니다.
            result["reason"] = (
                f"판정 불일치: "
                f"expected={expected}, "
                f"prediction={prediction}"
            )

    # 최종 결과 반환
    return result


# ============================================================
# 10. JSON 분석 모드
# ============================================================
#
# 이 함수는 모드 2 전체를 담당합니다.
#
# 전체 흐름:
#
# data.json 읽기
#       ↓
# JSON 구조 검사
#       ↓
# 필터 검사
#       ↓
# 모든 패턴 케이스 분석
#       ↓
# PASS / FAIL 집계
#       ↓
# 성능 분석
#       ↓
# 최종 결과 요약
# ============================================================


def run_json_mode(
    json_path: str = "data.json"
) -> None:
    """
    data.json을 읽어 모든 패턴 케이스를 분석한다.
    """

    # --------------------------------------------------------
    # JSON 데이터 로드
    # --------------------------------------------------------

    print("\n#---------------------------------------")
    print("# [1] JSON 데이터 로드")
    print("#---------------------------------------")

    # 파일이 존재하는지 먼저 확인합니다.
    if not os.path.exists(json_path):
        print(
            f"파일을 찾을 수 없습니다: {json_path}"
        )
        return

    # JSON 파일을 읽습니다.
    try:

        data = load_json_file(json_path)

    except json.JSONDecodeError as error:

        # JSON 문법 자체가 잘못된 경우입니다.
        print("JSON 파싱 오류:")
        print(error)
        return

    except OSError as error:

        # 파일 접근/읽기 과정에서 문제가 발생한 경우입니다.
        print("파일 읽기 오류:")
        print(error)
        return

    # --------------------------------------------------------
    # 최상위 JSON 구조 검사
    # --------------------------------------------------------

    if not isinstance(data, dict):
        print(
            "스키마 오류: "
            "최상위 JSON은 객체여야 합니다."
        )
        return

    # filters와 patterns를 가져옵니다.
    filters = data.get("filters")
    patterns = data.get("patterns")

    # filters가 없거나 잘못된 형태인지 확인합니다.
    if not isinstance(filters, dict):
        print(
            "스키마 오류: 'filters'가 없습니다."
        )
        return

    # patterns가 없거나 잘못된 형태인지 확인합니다.
    if not isinstance(patterns, dict):
        print(
            "스키마 오류: 'patterns'가 없습니다."
        )
        return

    # --------------------------------------------------------
    # 필터 로드 결과 출력
    # --------------------------------------------------------

    print("\n#---------------------------------------")
    print("# [2] 필터 로드")
    print("#---------------------------------------")

    # 정상적으로 로드된 필터 크기를 기록합니다.
    #
    # 현재 코드에서는 실제 분석 여부 판단에 직접 사용하지 않지만
    # 어떤 크기의 필터가 정상인지 확인하기 위한 정보입니다.
    valid_filter_sizes = []

    # 5, 13, 25 크기를 차례대로 확인합니다.
    for size in SUPPORTED_JSON_SIZES:

        ok, reason, normalized_filters = (
            validate_filter_group(
                filters,
                size
            )
        )

        if ok:

            valid_filter_sizes.append(size)

            print(
                f"✓ size_{size:<2} "
                f"필터 로드 완료 (Cross, X)"
            )

        else:

            print(
                f"✗ size_{size:<2} "
                f"필터 로드 실패: {reason}"
            )

    # --------------------------------------------------------
    # 패턴 분석
    # --------------------------------------------------------

    print("\n#---------------------------------------")
    print("# [3] 패턴 분석 (라벨 정규화 적용)")
    print("#---------------------------------------")

    # 전체 테스트 수
    total = 0

    # PASS 수
    passed = 0

    # FAIL 수
    failed = 0

    # 실패 케이스 상세 정보를 저장합니다.
    failures = []

    # patterns의 모든 케이스를 하나씩 처리합니다.
    for case_id, case_data in patterns.items():

        # 테스트 수 증가
        total += 1

        # 현재 케이스 분석
        result = analyze_pattern_case(
            case_id,
            case_data,
            filters
        )

        # PASS / FAIL 집계
        if result["status"] == "PASS":

            passed += 1

        else:

            failed += 1

            # 나중에 최종 실패 목록을 출력하기 위해 저장합니다.
            failures.append(result)

        # ----------------------------------------------------
        # 개별 케이스 출력
        # ----------------------------------------------------

        print(f"\n--- {case_id} ---")

        # MAC 점수까지 정상적으로 계산된 경우
        # 점수와 판정 결과를 출력합니다.
        if result["cross_score"] is not None:

            print(
                f"Cross 점수: "
                f"{result['cross_score']:.10f}"
            )

            print(
                f"X 점수: "
                f"{result['x_score']:.10f}"
            )

            print(
                f"판정: {result['prediction']} | "
                f"expected: {result['expected']} | "
                f"{result['status']}"
            )

            # FAIL이면 실패 원인도 출력합니다.
            if result["status"] == "FAIL":

                print(
                    f"원인: {result['reason']}"
                )

        else:

            # 행렬 오류나 스키마 오류 등으로
            # 점수 계산까지 가지 못한 경우입니다.
            print(
                f"판정: FAIL | "
                f"원인: {result['reason']}"
            )

    # --------------------------------------------------------
    # 성능 분석
    # --------------------------------------------------------

    print("\n#---------------------------------------")
    print("# [4] 성능 분석")
    print("#---------------------------------------")

    # 요구사항에 있는 모든 크기를 측정합니다.
    #
    # 3×3
    # 5×5
    # 13×13
    # 25×25
    run_performance_analysis(
        (3, 5, 13, 25),
        PERFORMANCE_REPEATS
    )

    # --------------------------------------------------------
    # 최종 결과 요약
    # --------------------------------------------------------

    print("\n#---------------------------------------")
    print("# [5] 결과 요약")
    print("#---------------------------------------")

    print(f"총 테스트: {total}개")
    print(f"통과: {passed}개")
    print(f"실패: {failed}개")

    # 실패한 케이스가 하나라도 있다면
    # 목록을 출력합니다.
    if failures:

        print("\n실패 케이스:")

        for failure in failures:

            print(
                f"- {failure['case_id']}: "
                f"{failure['reason']}"
            )

    else:

        print("\n실패 케이스가 없습니다.")


# ============================================================
# 11. 메인 메뉴
# ============================================================
#
# 지금까지 만든 모든 기능을 실제로 실행시키는 부분입니다.
#
# 프로그램을 실행하면
#
#     Mini NPU Simulator
#
#     [모드 선택]
#     1. 사용자 입력 (3x3)
#     2. data.json 분석
#     0. 종료
#
# 가 나타납니다.
#
# 사용자가 선택한 번호에 따라
# 적절한 함수를 호출합니다.
# ============================================================


def print_title() -> None:
    """
    프로그램 제목을 출력한다.
    """

    print("\n=======================================")
    print("        Mini NPU Simulator")
    print("=======================================")


def main() -> None:
    """
    프로그램의 시작점입니다.

    전체 실행 흐름을 관리합니다.
    """

    # 프로그램 제목 출력
    print_title()

    # 메뉴를 반복해서 보여줍니다.
    while True:

        print("\n[모드 선택]")
        print("1. 사용자 입력 (3x3)")
        print("2. data.json 분석")
        print("0. 종료")

        # 사용자의 메뉴 선택을 받습니다.
        choice = input("선택: ").strip()

        # ----------------------------------------------------
        # 모드 1
        # ----------------------------------------------------

        if choice == "1":

            # 사용자 입력 모드 실행
            run_user_mode()

            # 모드가 끝나면 프로그램 종료
            break

        # ----------------------------------------------------
        # 모드 2
        # ----------------------------------------------------

        elif choice == "2":

            # data.json의 경로를 입력받습니다.
            #
            # 아무것도 입력하지 않으면
            # 기본값인 data.json을 사용합니다.
            json_path = input(
                "data.json 경로 "
                "(기본값: data.json): "
            ).strip()

            if not json_path:
                json_path = "data.json"

            # JSON 분석 모드 실행
            run_json_mode(json_path)

            # 분석이 끝나면 프로그램 종료
            break

        # ----------------------------------------------------
        # 종료
        # ----------------------------------------------------

        elif choice == "0":

            print("프로그램을 종료합니다.")
            break

        # ----------------------------------------------------
        # 잘못된 메뉴 입력
        # ----------------------------------------------------

        else:

            print(
                "입력 오류: 1, 2 또는 0을 선택하세요."
            )


# ============================================================
# 프로그램 시작
# ============================================================
#
# Python 파일을 직접 실행했을 때만 main()을 실행합니다.
#
# 예:
#
#     python main.py
#
# 이때 __name__은 "__main__"이 됩니다.
#
# 반대로 다른 Python 파일에서
#
#     import main
#
# 처럼 가져오는 경우에는
# main()이 자동으로 실행되지 않습니다.
#
# 따라서 프로그램의 시작점을 명확하게 관리할 수 있습니다.
# ============================================================


if __name__ == "__main__":
    main()




"""
main()
  │
  ├─ 모드 1 → run_user_mode()
  │              │
  │              ├─ read_matrix_from_console()
  │              ├─ mac_score()
  │              ├─ classify_scores()
  │              └─ run_performance_analysis()
  │
  └─ 모드 2 → run_json_mode()
                 │
                 ├─ load_json_file()
                 ├─ validate_filter_group()
                 ├─ analyze_pattern_case()
                 │      │
                 │      ├─ normalize_label()
                 │      ├─ validate_square_matrix()
                 │      ├─ mac_score()
                 │      └─ classify_scores()
                 │
                 ├─ PASS / FAIL 집계
                 └─ run_performance_analysis()
"""


"""
핵심은 mac_score() 하나

패턴
 ↓
[0 1 0]
[1 1 1]  ×  필터
[0 1 0]
 ↓
같은 위치끼리 곱하기
 ↓
모두 더하기
 ↓
MAC 점수
 ↓
Cross 점수와 X 점수 비교
 ↓
Cross / X / UNDECIDED
"""


"""
data.json 모드에서는 여기에 검증과 expected 비교

data.json
   ↓
JSON 구조 확인
   ↓
라벨 정규화
   ↓
패턴 크기 확인
   ↓
해당 크기의 Cross/X 필터 선택
   ↓
MAC 계산
   ↓
Cross/X 비교
   ↓
expected와 비교
   ↓
PASS / FAIL
"""
