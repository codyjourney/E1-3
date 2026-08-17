import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# NPU Simulator
# ============================================================
#
# 이 프로그램은 NPU(Neural Processing Unit)의 기본 연산을
# 아주 간단하게 흉내 내는 시뮬레이터입니다.
#
# 핵심 연산은 MAC입니다.
#
# MAC = Multiply-Accumulate
#     = 각각의 값을 곱하고 모두 더하는 연산
#
# 예를 들어,
#
# 패턴:
#   1 2
#   3 4
#
# 필터:
#   1 0
#   0 1
#
# MAC 결과:
#   1×1 + 2×0 + 3×0 + 4×1
#   = 1 + 4
#   = 5
#
# 이 프로그램에서는
#   Cross(+) 모양 필터
#   X 모양 필터
#
# 두 개를 사용하여 어떤 모양에 더 가까운지 판정합니다.
#
# 또한 JSON에 저장된 실제 패턴을 사용하여
# MAC 연산에 걸리는 시간도 측정합니다.
#
# ============================================================


# ============================================================
# 프로그램 기본 설정
# ============================================================

# 두 점수가 이 값보다 작게 차이나면
# 사실상 같은 점수라고 판단합니다.
#
# 예:
#   Cross = 10.000000000
#   X     = 10.0000000005
#
# 차이가 EPSILON보다 작으면 "UNDECIDED"가 됩니다.
EPSILON = 1e-9


# 필터의 최소 크기입니다.
#
# Cross나 X 모양을 만들려면 가운데가 필요하기 때문에
# 최소 3×3 크기를 사용합니다.
MIN_SIZE = 3


# JSON 모드에서 허용하는 행렬 크기입니다.
#
# 예:
#   size_5_0   → 5×5
#   size_13_0  → 13×13
#   size_25_0  → 25×25
SUPPORTED_JSON_SIZES = (5, 13, 25)


# 성능 측정을 몇 번 반복할지 결정합니다.
#
# 한 번만 측정하면 컴퓨터 상태에 따라 시간이 크게 달라질 수 있으므로
# 여러 번 측정한 뒤 평균을 사용합니다.
PERFORMANCE_REPEATS = 10


# ============================================================
# 1. 라벨 정규화
# ============================================================

def normalize_label(label: Any) -> Optional[str]:
    """
    JSON 등에 저장된 다양한 형태의 라벨을
    프로그램에서 사용할 하나의 표준 이름으로 바꿉니다.

    예:
        "+"     → "Cross"
        "cross" → "Cross"
        "Cross" → "Cross"

        "x"     → "X"
        "X"     → "X"

    알 수 없는 라벨이면 None을 반환합니다.
    """

    # 문자열이 아니면 라벨로 사용할 수 없습니다.
    if not isinstance(label, str):
        return None

    # 앞뒤 공백을 제거하고 소문자로 변경합니다.
    value = label.strip().lower()

    # + 또는 cross는 Cross로 통일합니다.
    if value in ("+", "cross"):
        return "Cross"

    # x는 X로 통일합니다.
    if value in ("x",):
        return "X"

    # 그 외에는 알 수 없는 라벨입니다.
    return None


# ============================================================
# 2. 행렬(Matrix) 확인
# ============================================================

def is_matrix(value: Any) -> bool:
    """
    입력값이 2차원 리스트인지 확인합니다.

    정상적인 행렬 예:
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]

    잘못된 예:
        [1, 2, 3]

    반환값:
        True  → 2차원 리스트
        False → 행렬이 아님
    """

    # 전체가 리스트이고,
    # 비어 있지 않아야 합니다.
    if not isinstance(value, list) or len(value) == 0:
        return False

    # 모든 요소가 리스트인지 확인합니다.
    if not all(isinstance(row, list) for row in value):
        return False

    return True


def matrix_size(matrix: Any) -> Optional[Tuple[int, int]]:
    """
    행렬의 크기를 확인합니다.

    예:
        3×3 행렬 → (3, 3)
        5×5 행렬 → (5, 5)

    행마다 길이가 다르면 올바른 행렬이 아니므로 None을 반환합니다.
    """

    # 먼저 2차원 리스트인지 확인합니다.
    if not is_matrix(matrix):
        return None

    # 행의 개수를 구합니다.
    rows = len(matrix)

    # 첫 번째 행의 열 개수를 기준으로 사용합니다.
    cols = len(matrix[0])

    # 열이 하나도 없으면 잘못된 행렬입니다.
    if cols == 0:
        return None

    # 모든 행의 열 개수가 같은지 확인합니다.
    if any(len(row) != cols for row in matrix):
        return None

    return rows, cols


def validate_square_matrix(
    matrix: Any,
    expected_size: Optional[int] = None
) -> Tuple[bool, str]:
    """
    행렬이 올바른 정사각형인지 검사합니다.

    검사하는 내용:
        1. 2차원 리스트인가?
        2. 각 행의 길이가 같은가?
        3. 행과 열의 크기가 같은가?
        4. 원하는 크기와 같은가?
        5. 모든 값이 숫자로 변환 가능한가?

    반환:
        (True, "")             → 정상
        (False, "오류 설명")   → 오류
    """

    size = matrix_size(matrix)

    # 기본적인 행렬 구조가 잘못된 경우
    if size is None:
        return False, "2차원 배열이 아니거나 행의 길이가 서로 다릅니다."

    rows, cols = size

    # 행과 열의 크기가 같아야 정사각형입니다.
    if rows != cols:
        return False, f"정사각형이 아닙니다. 현재 크기: {rows}x{cols}"

    # 특정 크기를 요구한 경우 크기도 확인합니다.
    if expected_size is not None and rows != expected_size:
        return (
            False,
            f"크기 불일치: 기대 크기 {expected_size}x{expected_size}, "
            f"실제 크기 {rows}x{cols}"
        )

    # 모든 값을 하나씩 확인합니다.
    for r, row in enumerate(matrix):
        for c, value in enumerate(row):

            try:
                # 숫자로 변환 가능한지 확인합니다.
                float(value)

            except (TypeError, ValueError):

                return (
                    False,
                    f"숫자가 아닌 값 발견: 위치 ({r}, {c}), 값={value!r}"
                )

    return True, ""


def to_float_matrix(
    matrix: List[List[Any]]
) -> List[List[float]]:
    """
    행렬 안의 모든 값을 float 타입으로 변환합니다.

    예:
        [["1", "2"], ["3", "4"]]
        ↓
        [[1.0, 2.0], [3.0, 4.0]]
    """

    return [
        [float(value) for value in row]
        for row in matrix
    ]


# ============================================================
# 3. MAC 연산
# ============================================================

def mac_score(
    pattern: List[List[float]],
    filter_matrix: List[List[float]]
) -> float:
    """
    패턴과 필터를 같은 위치끼리 곱한 후
    모든 값을 더합니다.

    이것이 이 프로그램에서 가장 중요한 연산입니다.

    예:
        pattern:
            1 2
            3 4

        filter:
            1 0
            0 1

        계산:
            1×1 + 2×0 + 3×0 + 4×1
            = 5

    시간 복잡도:
        N×N의 모든 위치를 확인하므로 O(N²)
    """

    # 행렬의 크기
    rows = len(pattern)

    # MAC 결과를 저장할 변수
    score = 0.0

    # 모든 행을 순회합니다.
    for r in range(rows):

        # 모든 열을 순회합니다.
        for c in range(rows):

            # 같은 위치의 패턴과 필터를 곱해서
            # 기존 점수에 더합니다.
            score += (
                pattern[r][c]
                * filter_matrix[r][c]
            )

    return score


# ============================================================
# 4. 판정
# ============================================================

def classify_scores(
    score_a: float,
    score_b: float,
    label_a: str = "A",
    label_b: str = "B",
    epsilon: float = EPSILON
) -> str:
    """
    두 MAC 점수를 비교하여 더 높은 쪽을 선택합니다.

    예:
        A = 10
        B = 5
        → "A"

        A = 5
        B = 10
        → "B"

    두 점수의 차이가 너무 작으면
    확실하게 구분할 수 없다고 판단하여
    "UNDECIDED"를 반환합니다.
    """

    # 두 점수의 차이를 계산합니다.
    difference = abs(score_a - score_b)

    # 거의 같은 점수라면 판정하지 않습니다.
    if difference < epsilon:
        return "UNDECIDED"

    # A가 더 높으면 A를 반환합니다.
    if score_a > score_b:
        return label_a

    # 그렇지 않으면 B가 더 높습니다.
    return label_b


# ============================================================
# 5. Cross 필터 자동 생성
# ============================================================

def create_cross_filter(size: int) -> List[List[float]]:
    """
    '+' 모양의 Cross 필터를 자동으로 만듭니다.

    예: 3×3

        0 1 0
        1 1 1
        0 1 0

    가운데 행과 가운데 열을 1로 만들고
    나머지는 0으로 만듭니다.
    """

    # 너무 작은 크기는 허용하지 않습니다.
    if size < MIN_SIZE:
        raise ValueError(
            f"필터 크기는 최소 {MIN_SIZE} 이상이어야 합니다."
        )

    # 가운데 위치가 필요하기 때문에 홀수 크기만 사용합니다.
    if size % 2 == 0:
        raise ValueError(
            "Cross 필터는 중앙 위치가 필요하므로 홀수 크기만 지원합니다."
        )

    # 예: size=3이면 center=1
    center = size // 2

    # 각 위치를 검사하여
    # 가운데 행 또는 가운데 열이면 1,
    # 아니면 0을 넣습니다.
    return [
        [
            1.0
            if r == center or c == center
            else 0.0
            for c in range(size)
        ]
        for r in range(size)
    ]


# ============================================================
# 6. X 필터 자동 생성
# ============================================================

def create_x_filter(size: int) -> List[List[float]]:
    """
    'X' 모양의 필터를 자동으로 만듭니다.

    예: 3×3

        1 0 1
        0 1 0
        1 0 1

    두 대각선 위치를 1로 만들고
    나머지는 0으로 만듭니다.
    """

    if size < MIN_SIZE:
        raise ValueError(
            f"필터 크기는 최소 {MIN_SIZE} 이상이어야 합니다."
        )

    if size % 2 == 0:
        raise ValueError(
            "X 필터는 중앙 위치가 필요하므로 홀수 크기만 지원합니다."
        )

    return [
        [
            # r == c
            #     → 왼쪽 위에서 오른쪽 아래로 가는 대각선
            #
            # r + c == size - 1
            #     → 오른쪽 위에서 왼쪽 아래로 가는 대각선
            #
            # 둘 중 하나라도 맞으면 X 모양이므로 1
            1.0
            if r == c or r + c == size - 1
            else 0.0
            for c in range(size)
        ]
        for r in range(size)
    ]


# ============================================================
# 7. 행렬 출력
# ============================================================

def print_matrix(
    matrix: List[List[float]],
    title: str
) -> None:
    """
    행렬을 사람이 보기 편한 형태로 출력합니다.
    """

    print(f"\n[{title}]")

    for row in matrix:

        print(
            " ".join(
                f"{value:g}"
                for value in row
            )
        )


# ============================================================
# 8. MAC 성능 측정
# ============================================================

def measure_mac(
    pattern: List[List[float]],
    filter_matrix: List[List[float]],
    repeats: int = PERFORMANCE_REPEATS
) -> Tuple[float, float]:
    """
    같은 MAC 연산을 여러 번 실행하여 평균 시간을 측정합니다.

    반환값:
        (
            평균 실행 시간(ms),
            마지막 MAC 결과
        )

    중요한 점:
        여기서는 새로운 테스트 패턴을 만들지 않습니다.

        전달받은 pattern을 그대로 사용합니다.

    즉,

        실제 입력 패턴
              ↓
        MAC 성능 측정

    방식입니다.
    """

    # 반복 횟수는 최소 1회 이상이어야 합니다.
    if repeats <= 0:
        raise ValueError(
            "repeats는 1 이상이어야 합니다."
        )

    # 각 반복의 실행 시간을 저장합니다.
    elapsed_times = []

    last_score = 0.0

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------
    #
    # 실제 측정 전에 한 번 실행합니다.
    #
    # 첫 실행에는 Python이나 시스템의 여러 초기 작업 때문에
    # 평소보다 시간이 오래 걸릴 수 있기 때문입니다.
    last_score = mac_score(
        pattern,
        filter_matrix
    )

    # --------------------------------------------------------
    # 실제 성능 측정
    # --------------------------------------------------------

    for _ in range(repeats):

        # 시작 시간을 기록합니다.
        start = time.perf_counter()

        # 실제 MAC 연산
        last_score = mac_score(
            pattern,
            filter_matrix
        )

        # 종료 시간을 기록합니다.
        end = time.perf_counter()

        # 초 단위를 ms(밀리초)로 변환합니다.
        elapsed_times.append(
            (end - start) * 1000.0
        )

    # 모든 측정 시간의 평균을 계산합니다.
    average_ms = (
        sum(elapsed_times)
        / len(elapsed_times)
    )

    return average_ms, last_score


# ============================================================
# 9. 성능 분석
# ============================================================

def run_performance_analysis(
    pattern: List[List[float]],
    filter_a: List[List[float]],
    filter_b: List[List[float]],
    label_a: str = "Cross",
    label_b: str = "X",
    repeats: int = PERFORMANCE_REPEATS
) -> None:
    """
    하나의 실제 패턴을 가지고
    두 필터의 MAC 성능을 비교합니다.

    측정하는 것은:

        pattern + filter_a
        pattern + filter_b

    입니다.

    별도의 테스트 패턴을 만들지 않고
    실제 입력 패턴을 그대로 사용합니다.
    """

    # --------------------------------------------------------
    # 패턴 크기 확인
    # --------------------------------------------------------

    size = matrix_size(pattern)

    if size is None:

        print(
            "성능 분석 오류: "
            "패턴 행렬이 올바르지 않습니다."
        )

        return

    rows, cols = size

    # 패턴은 정사각형이어야 합니다.
    if rows != cols:

        print(
            "성능 분석 오류: "
            "패턴은 정사각형이어야 합니다."
        )

        return

    # --------------------------------------------------------
    # 필터 크기 확인
    # --------------------------------------------------------

    filter_a_size = matrix_size(filter_a)
    filter_b_size = matrix_size(filter_b)

    # A 필터와 패턴 크기가 같은지 확인합니다.
    if filter_a_size != (rows, cols):

        print(
            "성능 분석 오류: "
            f"{label_a} 필터 크기가 패턴과 다릅니다."
        )

        return

    # B 필터와 패턴 크기가 같은지 확인합니다.
    if filter_b_size != (rows, cols):

        print(
            "성능 분석 오류: "
            f"{label_b} 필터 크기가 패턴과 다릅니다."
        )

        return

    # --------------------------------------------------------
    # 성능 분석 시작
    # --------------------------------------------------------

    print("\n#---------------------------------------")
    print(f"# [성능 분석] 실제 입력 패턴 / 평균 {repeats}회")
    print("#---------------------------------------")

    print(
        f"패턴 크기: {rows}x{cols}"
    )

    print(
        "성능 측정 대상: "
        "현재 분석에 사용한 실제 패턴"
    )

    print()

    print(
        f"{'필터':<15}"
        f"{'평균 시간(ms)':>20}"
        f"{'MAC 결과':>20}"
        f"{'MAC 위치 수(N²)':>20}"
    )

    print("-" * 80)

    # A 필터의 성능 측정
    time_a, score_a = measure_mac(
        pattern,
        filter_a,
        repeats
    )

    # B 필터의 성능 측정
    time_b, score_b = measure_mac(
        pattern,
        filter_b,
        repeats
    )

    # N×N 행렬이면 총 N²개의 위치에서
    # 곱셈과 덧셈을 수행합니다.
    operation_count = rows * cols

    # --------------------------------------------------------
    # 측정 결과 출력
    # --------------------------------------------------------

    print(
        f"{label_a:<15}"
        f"{time_a:>20.6f}"
        f"{score_a:>20.10f}"
        f"{operation_count:>20}"
    )

    print(
        f"{label_b:<15}"
        f"{time_b:>20.6f}"
        f"{score_b:>20.10f}"
        f"{operation_count:>20}"
    )

    # 두 필터의 평균 실행 시간을 계산합니다.
    average_time = (
        time_a + time_b
    ) / 2.0

    print()

    print(
        f"전체 평균 MAC 시간: "
        f"{average_time:.6f} ms"
    )


# ============================================================
# 10. 사용자 입력 모드
# ============================================================

def read_matrix_from_console(
    size: int,
    matrix_name: str
) -> List[List[float]]:
    """
    사용자가 콘솔에서 N×N 행렬을 직접 입력합니다.

    예를 들어 3×3이면:

        1 0 1
        0 1 0
        1 0 1

    처럼 3줄을 입력합니다.

    잘못 입력하면 다시 입력하도록 합니다.
    """

    while True:

        print(
            f"\n{matrix_name} "
            f"({size}줄 입력, 공백 구분)"
        )

        matrix = []

        # 행을 하나씩 입력받습니다.
        for row_index in range(size):

            while True:

                text = input(
                    f"{row_index + 1}/{size}행 > "
                ).strip()

                # 공백을 기준으로 숫자를 나눕니다.
                parts = text.split()

                # 한 행에 정확히 size개의 값이 있어야 합니다.
                if len(parts) != size:

                    print(
                        f"입력 형식 오류: "
                        f"각 줄에 {size}개의 숫자를 "
                        f"공백으로 구분해 입력하세요."
                    )

                    continue

                try:

                    # 문자열을 실수로 변환합니다.
                    row = [
                        float(value)
                        for value in parts
                    ]

                except ValueError:

                    print(
                        "입력 형식 오류: 숫자만 입력하세요."
                    )

                    continue

                # 정상적으로 입력된 행을 저장합니다.
                matrix.append(row)

                break

        # 전체 행렬이 정상인지 마지막으로 확인합니다.
        ok, reason = validate_square_matrix(
            matrix,
            expected_size=size
        )

        if ok:
            return matrix

        print(f"입력 오류: {reason}")
        print("행렬을 처음부터 다시 입력해주세요.")


def run_user_mode() -> None:
    """
    사용자 입력 모드입니다.

    사용자가 직접:

        필터 A
        필터 B
        패턴

    을 입력합니다.

    이후:

        1. MAC 점수 계산
        2. A/B 판정
        3. 실행 시간 측정
        4. 성능 분석

    을 수행합니다.
    """

    # 현재 사용자 모드는 3×3만 사용합니다.
    size = 3

    # --------------------------------------------------------
    # [1] 필터 입력
    # --------------------------------------------------------

    print("\n#---------------------------------------")
    print("# [1] 필터 입력")
    print("#---------------------------------------")

    filter_a = read_matrix_from_console(
        size,
        "필터 A"
    )

    filter_b = read_matrix_from_console(
        size,
        "필터 B"
    )

    print("\n필터 A 저장 완료.")
    print("필터 B 저장 완료.")

    # --------------------------------------------------------
    # [2] 패턴 입력
    # --------------------------------------------------------

    print("\n#---------------------------------------")
    print("# [2] 패턴 입력")
    print("#---------------------------------------")

    pattern = read_matrix_from_console(
        size,
        "패턴"
    )

    print("\n패턴 저장 완료.")

    # --------------------------------------------------------
    # [3] MAC 계산
    # --------------------------------------------------------

    print("\n#---------------------------------------")
    print("# [3] MAC 결과")
    print("#---------------------------------------")

    # 패턴과 A 필터의 MAC 점수
    score_a = mac_score(
        pattern,
        filter_a
    )

    # 패턴과 B 필터의 MAC 점수
    score_b = mac_score(
        pattern,
        filter_b
    )

    # A 필터의 평균 실행 시간
    average_a_ms, _ = measure_mac(
        pattern,
        filter_a,
        PERFORMANCE_REPEATS
    )

    # B 필터의 평균 실행 시간
    average_b_ms, _ = measure_mac(
        pattern,
        filter_b,
        PERFORMANCE_REPEATS
    )

    # A와 B의 평균 실행 시간
    average_ms = (
        average_a_ms + average_b_ms
    ) / 2.0

    # --------------------------------------------------------
    # 판정
    # --------------------------------------------------------

    # 점수가 높은 필터를 선택합니다.
    result = classify_scores(
        score_a,
        score_b,
        label_a="A",
        label_b="B",
        epsilon=EPSILON
    )

    print(
        f"A 점수: {score_a:.10f}"
    )

    print(
        f"B 점수: {score_b:.10f}"
    )

    print(
        f"연산 시간(평균/{PERFORMANCE_REPEATS}회): "
        f"{average_ms:.6f} ms"
    )

    if result == "UNDECIDED":

        print(
            f"판정: 판정 불가 "
            f"(|A-B| < {EPSILON})"
        )

    else:

        print(
            f"판정: {result}"
        )

    # --------------------------------------------------------
    # [4] 추가 성능 분석
    # --------------------------------------------------------

    print("\n#---------------------------------------")
    print("# [4] 성능 분석")
    print("#---------------------------------------")

    # 여기서 중요한 점:
    #
    # 성능 측정을 위해 새로운 패턴을 만들지 않습니다.
    #
    # 사용자가 실제로 입력한
    #
    #     pattern
    #
    # 을 그대로 성능 측정에 사용합니다.

    run_performance_analysis(
        pattern=pattern,
        filter_a=filter_a,
        filter_b=filter_b,
        label_a="A",
        label_b="B",
        repeats=PERFORMANCE_REPEATS
    )


# ============================================================
# 11. JSON 파일 읽기
# ============================================================

def load_json_file(
    path: str
) -> Dict[str, Any]:
    """
    JSON 파일을 열어서 Python 객체로 변환합니다.
    """

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def extract_size_from_pattern_key(
    key: str
) -> Optional[int]:
    """
    JSON의 패턴 이름에서 행렬 크기를 추출합니다.

    예:
        size_5_0   → 5
        size_13_3  → 13
        size_25_10 → 25

    여기서 뒤의 숫자는 패턴 번호이고
    앞의 숫자가 행렬 크기입니다.
    """

    match = re.fullmatch(
        r"size_(\d+)_(\d+)",
        key
    )

    # 패턴 이름 형식이 맞지 않으면 None
    if not match:
        return None

    # 첫 번째 숫자만 크기로 사용합니다.
    return int(match.group(1))


# ============================================================
# 12. JSON 필터 검증
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
    JSON 안에 있는 특정 크기의 Cross/X 필터가
    올바르게 저장되어 있는지 검사합니다.

    예:
        size = 5

    그러면 JSON에서

        filters
          └── size_5
                ├── cross
                └── x

    구조를 확인합니다.
    """

    # filters가 객체인지 확인합니다.
    if not isinstance(filters, dict):

        return (
            False,
            "filters가 객체가 아닙니다.",
            None
        )

    # 예: size_5
    size_key = f"size_{size}"

    # 해당 크기의 필터가 존재하는지 확인합니다.
    if size_key not in filters:

        return (
            False,
            f"{size_key} 필터가 존재하지 않습니다.",
            None
        )

    group = filters[size_key]

    # size_N의 값도 객체여야 합니다.
    if not isinstance(group, dict):

        return (
            False,
            f"{size_key} 값이 객체가 아닙니다.",
            None
        )

    normalized = {}

    # Cross와 X 두 필터가 모두 필요한지 확인합니다.
    for raw_label in ("cross", "x"):

        if raw_label not in group:

            return (
                False,
                f"{size_key}에 '{raw_label}' 필터가 없습니다.",
                None
            )

        # "cross" → "Cross"
        # "x"     → "X"
        label = normalize_label(
            raw_label
        )

        matrix = group[raw_label]

        # 실제 필터 행렬의 크기와 데이터도 검사합니다.
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

        # 모든 값을 float으로 변환하여 저장합니다.
        normalized[label] = to_float_matrix(
            matrix
        )

    return True, "", normalized


# ============================================================
# 13. JSON의 패턴 하나 분석
# ============================================================

def analyze_pattern_case(
    case_id: str,
    case_data: Any,
    filters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    JSON에 들어 있는 패턴 하나를 분석합니다.

    한 케이스에 대해 다음을 수행합니다.

        1. 패턴 크기 확인
        2. expected 정답 확인
        3. 입력 패턴 확인
        4. Cross 필터 확인
        5. X 필터 확인
        6. Cross MAC 계산
        7. X MAC 계산
        8. 판정
        9. 정답과 비교
       10. 실제 패턴의 MAC 실행 시간 측정
    """

    # 분석 결과를 저장할 기본 구조입니다.
    result = {
        "case_id": case_id,
        "status": "FAIL",
        "reason": "",
        "cross_score": None,
        "x_score": None,
        "prediction": None,
        "expected": None,

        # 성능 측정 결과
        "cross_time_ms": None,
        "x_time_ms": None,
        "average_time_ms": None,
    }

    # --------------------------------------------------------
    # 1. 패턴 이름에서 크기 추출
    # --------------------------------------------------------

    size = extract_size_from_pattern_key(
        case_id
    )

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

    # input이 있어야 합니다.
    if "input" not in case_data:

        result["reason"] = (
            "'input' 필드가 없습니다."
        )

        return result

    # expected도 있어야 합니다.
    if "expected" not in case_data:

        result["reason"] = (
            "'expected' 필드가 없습니다."
        )

        return result

    # --------------------------------------------------------
    # 4. 정답 라벨 확인
    # --------------------------------------------------------

    expected = normalize_label(
        case_data["expected"]
    )

    if expected is None:

        result["reason"] = (
            f"알 수 없는 expected 라벨: "
            f"{case_data['expected']!r}"
        )

        return result

    result["expected"] = expected

    # --------------------------------------------------------
    # 5. 입력 패턴 확인
    # --------------------------------------------------------

    pattern = case_data["input"]

    # 패턴이 올바른 N×N 행렬인지 확인합니다.
    ok, reason = validate_square_matrix(
        pattern,
        expected_size=size
    )

    if not ok:

        result["reason"] = (
            f"패턴 크기/데이터 오류: {reason}"
        )

        return result

    # 계산하기 편하도록 모든 값을 float으로 변환합니다.
    pattern = to_float_matrix(
        pattern
    )

    # --------------------------------------------------------
    # 6. Cross / X 필터 확인
    # --------------------------------------------------------

    ok, reason, normalized_filters = (
        validate_filter_group(
            filters,
            size
        )
    )

    if not ok:

        result["reason"] = (
            f"필터 오류: {reason}"
        )

        return result

    # 검증이 끝난 필터를 꺼냅니다.
    cross_filter = normalized_filters["Cross"]
    x_filter = normalized_filters["X"]

    # --------------------------------------------------------
    # 7. Cross MAC 계산
    # --------------------------------------------------------

    cross_score = mac_score(
        pattern,
        cross_filter
    )

    # --------------------------------------------------------
    # 8. X MAC 계산
    # --------------------------------------------------------

    x_score = mac_score(
        pattern,
        x_filter
    )

    # --------------------------------------------------------
    # 9. 두 점수를 비교하여 판정
    # --------------------------------------------------------

    prediction = classify_scores(
        cross_score,
        x_score,
        label_a="Cross",
        label_b="X",
        epsilon=EPSILON
    )

    result["cross_score"] = cross_score
    result["x_score"] = x_score
    result["prediction"] = prediction

    # --------------------------------------------------------
    # 10. 정답과 비교하여 PASS / FAIL 결정
    # --------------------------------------------------------

    if prediction == expected:

        # 실제 판정과 정답이 같으면 성공입니다.
        result["status"] = "PASS"
        result["reason"] = "정상 판정"

    else:

        # 정답과 다르면 실패입니다.
        result["status"] = "FAIL"

        if prediction == "UNDECIDED":

            result["reason"] = (
                f"동점 규칙: "
                f"|Cross-X| < {EPSILON}"
            )

        else:

            result["reason"] = (
                f"판정 불일치: "
                f"expected={expected}, "
                f"prediction={prediction}"
            )

    # --------------------------------------------------------
    # 11. 실제 입력 패턴으로 성능 측정
    # --------------------------------------------------------
    #
    # 여기서 매우 중요한 부분입니다.
    #
    # 성능 측정용으로 새로운 패턴을 만들지 않습니다.
    #
    # JSON의 실제 input:
    #
    #     case_data["input"]
    #
    # 을 그대로 사용합니다.
    #
    # 따라서
    #
    #     실제 분류에 사용한 패턴
    #              =
    #     성능 측정에 사용한 패턴
    #
    # 입니다.

    cross_time_ms, _ = measure_mac(
        pattern,
        cross_filter,
        PERFORMANCE_REPEATS
    )

    x_time_ms, _ = measure_mac(
        pattern,
        x_filter,
        PERFORMANCE_REPEATS
    )

    # 두 필터의 평균 실행 시간을 계산합니다.
    average_time_ms = (
        cross_time_ms
        + x_time_ms
    ) / 2.0

    # 결과에 성능 정보를 저장합니다.
    result["cross_time_ms"] = cross_time_ms
    result["x_time_ms"] = x_time_ms
    result["average_time_ms"] = average_time_ms

    return result


# ============================================================
# 14. JSON 분석 모드
# ============================================================

def run_json_mode(
    json_path: str = "data.json"
) -> None:
    """
    data.json을 읽어서 모든 패턴을 차례대로 분석합니다.

    각 패턴에 대해:

        JSON 읽기
          ↓
        패턴 검증
          ↓
        필터 검증
          ↓
        Cross MAC
          ↓
        X MAC
          ↓
        판정
          ↓
        정답 비교
          ↓
        실제 패턴 성능 측정

    순서로 진행됩니다.
    """

    # --------------------------------------------------------
    # [1] JSON 파일 읽기
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

    try:

        # JSON 파일을 읽습니다.
        data = load_json_file(
            json_path
        )

    except json.JSONDecodeError as error:

        # JSON 문법이 잘못된 경우
        print("JSON 파싱 오류:")
        print(error)

        return

    except OSError as error:

        # 파일을 읽을 수 없는 경우
        print("파일 읽기 오류:")
        print(error)

        return

    # JSON 최상위는 객체여야 합니다.
    if not isinstance(data, dict):

        print(
            "스키마 오류: "
            "최상위 JSON은 객체여야 합니다."
        )

        return

    # JSON에서 filters와 patterns를 가져옵니다.
    filters = data.get(
        "filters"
    )

    patterns = data.get(
        "patterns"
    )

    # filters가 있는지 확인합니다.
    if not isinstance(filters, dict):

        print(
            "스키마 오류: 'filters'가 없습니다."
        )

        return

    # patterns가 있는지 확인합니다.
    if not isinstance(patterns, dict):

        print(
            "스키마 오류: 'patterns'가 없습니다."
        )

        return

    # --------------------------------------------------------
    # [2] JSON 필터 확인
    # --------------------------------------------------------

    print("\n#---------------------------------------")
    print("# [2] 필터 로드")
    print("#---------------------------------------")

    valid_filter_sizes = []

    # 지원하는 크기를 하나씩 확인합니다.
    for size in SUPPORTED_JSON_SIZES:

        ok, reason, normalized_filters = (
            validate_filter_group(
                filters,
                size
            )
        )

        if ok:

            # 정상적으로 사용할 수 있는 크기를 저장합니다.
            valid_filter_sizes.append(
                size
            )

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
    # [3] 모든 패턴 분석
    # --------------------------------------------------------

    print("\n#---------------------------------------")
    print("# [3] 패턴 분석 + 실제 패턴 성능 측정")
    print("#---------------------------------------")

    # 전체 패턴 개수
    total = 0

    # 정답을 맞힌 개수
    passed = 0

    # 정답을 틀린 개수
    failed = 0

    # 실패한 케이스를 저장합니다.
    failures = []

    # 성능 측정이 완료된 결과를 저장합니다.
    performance_results = []

    # JSON에 들어 있는 모든 패턴을 하나씩 처리합니다.
    for case_id, case_data in patterns.items():

        total += 1

        # 패턴 하나를 분석합니다.
        result = analyze_pattern_case(
            case_id,
            case_data,
            filters
        )

        # PASS인지 FAIL인지 확인합니다.
        if result["status"] == "PASS":

            passed += 1

        else:

            failed += 1

            failures.append(
                result
            )

        # ----------------------------------------------------
        # 현재 패턴의 결과 출력
        # ----------------------------------------------------

        print(
            f"\n--- {case_id} ---"
        )

        # MAC 계산까지 정상적으로 끝난 경우
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

            # 실패한 경우 왜 실패했는지 출력합니다.
            if result["status"] == "FAIL":

                print(
                    f"원인: {result['reason']}"
                )

            # ------------------------------------------------
            # 성능 측정 결과 출력
            # ------------------------------------------------

            if result["cross_time_ms"] is not None:

                print(
                    f"Cross MAC 시간: "
                    f"{result['cross_time_ms']:.6f} ms"
                )

                print(
                    f"X MAC 시간: "
                    f"{result['x_time_ms']:.6f} ms"
                )

                print(
                    f"평균 MAC 시간: "
                    f"{result['average_time_ms']:.6f} ms"
                )

                # 나중에 전체 성능 결과를 출력하기 위해 저장합니다.
                performance_results.append(
                    result
                )

        else:

            # MAC 계산 자체가 불가능했던 경우
            print(
                f"판정: FAIL | "
                f"원인: {result['reason']}"
            )

    # --------------------------------------------------------
    # [4] 전체 성능 분석 결과
    # --------------------------------------------------------

    print("\n#---------------------------------------")
    print("# [4] 실제 입력 패턴 성능 분석")
    print("#---------------------------------------")

    print(
        "※ 별도의 1.0 테스트 패턴을 생성하지 않습니다."
    )

    print(
        "※ 각 JSON 케이스의 실제 input 패턴을 사용합니다."
    )

    print()

    # 성능 측정이 성공한 패턴이 있는 경우
    if performance_results:

        print(
            f"{'패턴':<18}"
            f"{'크기':<10}"
            f"{'Cross(ms)':>15}"
            f"{'X(ms)':>15}"
            f"{'평균(ms)':>15}"
        )

        print("-" * 75)

        # 각 패턴의 성능을 출력합니다.
        for result in performance_results:

            case_id = result["case_id"]

            # 패턴 이름에서 크기를 가져옵니다.
            size = extract_size_from_pattern_key(
                case_id
            )

            print(
                f"{case_id:<18}"
                f"{str(size) + 'x' + str(size):<10}"
                f"{result['cross_time_ms']:>15.6f}"
                f"{result['x_time_ms']:>15.6f}"
                f"{result['average_time_ms']:>15.6f}"
            )

    else:

        print(
            "성능 측정을 완료할 수 있는 "
            "정상 패턴이 없습니다."
        )

    # --------------------------------------------------------
    # [5] 최종 결과 요약
    # --------------------------------------------------------

    print("\n#---------------------------------------")
    print("# [5] 결과 요약")
    print("#---------------------------------------")

    print(
        f"총 테스트: {total}개"
    )

    print(
        f"통과: {passed}개"
    )

    print(
        f"실패: {failed}개"
    )

    # 실패한 테스트가 있다면 목록을 출력합니다.
    if failures:

        print("\n실패 케이스:")

        for failure in failures:

            print(
                f"- {failure['case_id']}: "
                f"{failure['reason']}"
            )

    else:

        print(
            "\n실패 케이스가 없습니다."
        )


# ============================================================
# 15. 메인 메뉴
# ============================================================

def print_title() -> None:
    """
    프로그램 시작 시 제목을 출력합니다.
    """

    print(
        "\n======================================="
    )

    print(
        "        NPU Simulator"
    )

    print(
        "=======================================")


def main() -> None:
    """
    프로그램의 시작점입니다.

    사용자에게 다음 메뉴를 보여줍니다.

        1 → 직접 행렬 입력
        2 → data.json 분석
        0 → 프로그램 종료
    """

    # 프로그램 제목 출력
    print_title()

    # 사용자가 종료할 때까지 메뉴를 반복합니다.
    while True:

        print("\n[모드 선택]")
        print("1. 사용자 입력 (3x3)")
        print("2. data.json 분석")
        print("0. 종료")

        choice = input(
            "선택: "
        ).strip()

        # ----------------------------------------------------
        # 1번: 사용자 입력 모드
        # ----------------------------------------------------

        if choice == "1":

            run_user_mode()

            # 작업이 끝나면 프로그램 종료
            break

        # ----------------------------------------------------
        # 2번: JSON 분석 모드
        # ----------------------------------------------------

        elif choice == "2":

            # JSON 파일 경로를 입력받습니다.
            json_path = input(
                "data.json 경로 "
                "(기본값: data.json): "
            ).strip()

            # 아무것도 입력하지 않으면
            # 기본 파일 이름을 사용합니다.
            if not json_path:

                json_path = "data.json"

            # JSON 분석 시작
            run_json_mode(
                json_path
            )

            # 작업이 끝나면 프로그램 종료
            break

        # ----------------------------------------------------
        # 0번: 종료
        # ----------------------------------------------------

        elif choice == "0":

            print(
                "프로그램을 종료합니다."
            )

            break

        # ----------------------------------------------------
        # 잘못된 메뉴 입력
        # ----------------------------------------------------

        else:

            print(
                "입력 오류: "
                "1, 2 또는 0을 선택하세요."
            )


# ============================================================
# 프로그램 실행
# ============================================================
#
# 이 파일을 직접 실행했을 때만 main()을 실행합니다.
#
# 다른 Python 파일에서 이 파일을 import하면
# main()이 자동으로 실행되지 않습니다.
# ============================================================

if __name__ == "__main__":
    main()
