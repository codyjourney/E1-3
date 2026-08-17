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
#    -> A 점수와 B 점수를 비교
#    -> A / B 중 하나로 판정
#
#    [모드 2]
#    data.json에 저장된 여러 크기의 필터와 패턴을 자동 분석
#    -> Cross 점수와 X 점수를 비교
#    -> Cross / X 중 하나로 판정
#
# 3. 패턴과 필터의 MAC 점수 계산
# 4. 두 점수를 비교하여 판정
# 5. data.json 모드에서는 expected와 비교하여 PASS / FAIL
# 6. 3×3, 5×5, 13×13, 25×25의 연산 시간을 측정
# 7. 전체 결과를 콘솔에 출력
#
# ------------------------------------------------------------
# 개발 조건
# ------------------------------------------------------------
#
# - Python 3.12+
# - 외부 라이브러리 사용 금지
# - NumPy 사용 금지
# - MAC 연산은 직접 반복문으로 구현
#
# ============================================================


# ============================================================
# 프로그램 설정값
# ============================================================

EPSILON = 1e-9

MIN_SIZE = 3

SUPPORTED_JSON_SIZES = (5, 13, 25)

PERFORMANCE_REPEATS = 10


# ============================================================
# 1. 라벨 정규화(Label Normalization)
# ============================================================

def normalize_label(label: Any) -> Optional[str]:
    """
    입력 라벨을 프로그램 내부의 표준 라벨로 변환한다.

    "+"       -> "Cross"
    "cross"   -> "Cross"
    "Cross"   -> "Cross"

    "x"       -> "X"
    "X"       -> "X"

    알 수 없는 값은 None을 반환한다.
    """

    if not isinstance(label, str):
        return None

    value = label.strip().lower()

    if value in ("+", "cross"):
        return "Cross"

    if value in ("x",):
        return "X"

    return None


# ============================================================
# 2. 행렬(Matrix) 검증
# ============================================================

def is_matrix(value: Any) -> bool:
    """
    전달받은 값이 2차원 리스트 형태인지 확인한다.
    """

    if not isinstance(value, list) or len(value) == 0:
        return False

    if not all(isinstance(row, list) for row in value):
        return False

    return True


def matrix_size(matrix: Any) -> Optional[Tuple[int, int]]:
    """
    행렬의 크기를 반환한다.

    예:
        [[1, 2, 3],
         [4, 5, 6]]

        -> (2, 3)
    """

    if not is_matrix(matrix):
        return None

    rows = len(matrix)
    cols = len(matrix[0])

    if cols == 0:
        return None

    if any(len(row) != cols for row in matrix):
        return None

    return rows, cols


def validate_square_matrix(
    matrix: Any,
    expected_size: Optional[int] = None
) -> Tuple[bool, str]:
    """
    정사각형 N×N 행렬인지 검사한다.
    """

    size = matrix_size(matrix)

    if size is None:
        return False, "2차원 배열이 아니거나 행의 길이가 서로 다릅니다."

    rows, cols = size

    if rows != cols:
        return False, f"정사각형이 아닙니다. 현재 크기: {rows}x{cols}"

    if expected_size is not None and rows != expected_size:
        return (
            False,
            f"크기 불일치: 기대 크기 {expected_size}x{expected_size}, "
            f"실제 크기 {rows}x{cols}"
        )

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
    패턴과 필터의 MAC 점수를 계산한다.

    시간 복잡도:
        O(N²)
    """

    rows = len(pattern)

    score = 0.0

    for r in range(rows):
        for c in range(rows):
            score += pattern[r][c] * filter_matrix[r][c]

    return score


# ============================================================
# 4. 판정(Classification)
# ============================================================
#
# 중요:
#
# 이 함수는 더 이상 Cross/X에 종속되지 않습니다.
#
# 사용자 입력 모드:
#
#     classify_scores(
#         score_a,
#         score_b,
#         "A",
#         "B"
#     )
#
#     -> "A" 또는 "B"
#
# JSON 모드:
#
#     classify_scores(
#         cross_score,
#         x_score,
#         "Cross",
#         "X"
#     )
#
#     -> "Cross" 또는 "X"
#
# 따라서 하나의 판정 함수를 두 모드에서
# 재사용할 수 있습니다.
# ============================================================

def classify_scores(
    score_a: float,
    score_b: float,
    label_a: str = "A",
    label_b: str = "B",
    epsilon: float = EPSILON
) -> str:
    """
    두 점수를 비교하여 label_a / label_b / UNDECIDED 중
    하나를 반환한다.

    score_a > score_b
        -> label_a

    score_b > score_a
        -> label_b

    두 점수의 차이가 epsilon보다 작으면
        -> UNDECIDED
    """

    difference = abs(score_a - score_b)

    if difference < epsilon:
        return "UNDECIDED"

    if score_a > score_b:
        return label_a

    return label_b


# ============================================================
# 5. 성능 측정용 패턴 생성
# ============================================================

def create_performance_pattern(size: int) -> List[List[float]]:
    """
    성능 측정용 패턴을 생성한다.

    모든 값을 1.0으로 설정한다.

    이 함수는 패턴만 생성한다.
    Cross/X 필터는 별도의 함수에서 생성한다.
    """

    return [
        [1.0 for _ in range(size)]
        for _ in range(size)
    ]


# ============================================================
# 6. 성능 측정용 Cross 필터 생성
# ============================================================

def create_cross_filter(size: int) -> List[List[float]]:
    """
    성능 측정용 Cross(+) 필터를 생성한다.
    """

    if size < MIN_SIZE:
        raise ValueError(
            f"필터 크기는 최소 {MIN_SIZE} 이상이어야 합니다."
        )

    if size % 2 == 0:
        raise ValueError(
            "Cross 필터는 중앙 위치가 필요하므로 홀수 크기만 지원합니다."
        )

    center = size // 2

    return [
        [
            1.0 if r == center or c == center else 0.0
            for c in range(size)
        ]
        for r in range(size)
    ]


# ============================================================
# 7. 성능 측정용 X 필터 생성
# ============================================================

def create_x_filter(size: int) -> List[List[float]]:
    """
    성능 측정용 X 필터를 생성한다.
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
            1.0
            if r == c or r + c == size - 1
            else 0.0
            for c in range(size)
        ]
        for r in range(size)
    ]


# ============================================================
# 8. 성능 측정용 필터 출력
# ============================================================

def print_matrix(
    matrix: List[List[float]],
    title: str
) -> None:
    """
    행렬을 콘솔에 출력한다.
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
# 9. MAC 성능 측정
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
    """

    if repeats <= 0:
        raise ValueError("repeats는 1 이상이어야 합니다.")

    elapsed_times = []

    last_score = 0.0

    # Warm-up
    last_score = mac_score(
        pattern,
        filter_matrix
    )

    # 실제 측정
    for _ in range(repeats):

        start = time.perf_counter()

        last_score = mac_score(
            pattern,
            filter_matrix
        )

        end = time.perf_counter()

        elapsed_times.append(
            (end - start) * 1000.0
        )

    average_ms = (
        sum(elapsed_times)
        / len(elapsed_times)
    )

    return average_ms, last_score


# ============================================================
# 10. 성능 분석
# ============================================================

def run_performance_analysis(
    sizes: Tuple[int, ...],
    repeats: int = PERFORMANCE_REPEATS
) -> None:
    """
    지정된 행렬 크기에 대해
    Cross/X 필터의 MAC 성능을 각각 측정한다.
    """

    print("\n#---------------------------------------")
    print(f"# [성능 분석] 평균/{repeats}회")
    print("#---------------------------------------")

    print(
        f"{'크기':<12}"
        f"{'Cross(ms)':>15}"
        f"{'X(ms)':>15}"
        f"{'MAC 횟수(N²)':>18}"
    )

    print("-" * 60)

    for size in sizes:

        # 1. 성능 측정용 패턴 생성
        pattern = create_performance_pattern(size)

        # 2. Cross 필터 생성
        cross_filter = create_cross_filter(size)

        # 3. X 필터 생성
        x_filter = create_x_filter(size)

        # 4. Cross MAC 성능 측정
        cross_ms, cross_score = measure_mac(
            pattern,
            cross_filter,
            repeats
        )

        # 5. X MAC 성능 측정
        x_ms, x_score = measure_mac(
            pattern,
            x_filter,
            repeats
        )

        # 6. MAC 기본 연산 횟수
        operation_count = size * size

        print(
            f"{size}x{size:<8}"
            f"{cross_ms:>15.6f}"
            f"{x_ms:>15.6f}"
            f"{operation_count:>18}"
        )


# ============================================================
# 11. 사용자 입력 모드
# ============================================================

def read_matrix_from_console(
    size: int,
    matrix_name: str
) -> List[List[float]]:
    """
    콘솔에서 N×N 행렬을 한 줄씩 입력받는다.
    """

    while True:

        print(
            f"\n{matrix_name} "
            f"({size}줄 입력, 공백 구분)"
        )

        matrix = []

        for row_index in range(size):

            while True:

                text = input(
                    f"{row_index + 1}/{size}행 > "
                ).strip()

                parts = text.split()

                if len(parts) != size:
                    print(
                        f"입력 형식 오류: "
                        f"각 줄에 {size}개의 숫자를 "
                        f"공백으로 구분해 입력하세요."
                    )
                    continue

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

                matrix.append(row)

                break

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
    모드 1을 실행한다.

    사용자 입력 모드에서는
    필터 A와 필터 B를 비교하므로
    최종 판정은 A / B로 표시한다.
    """

    size = 3

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

    print("\n#---------------------------------------")
    print("# [2] 패턴 입력")
    print("#---------------------------------------")

    pattern = read_matrix_from_console(
        size,
        "패턴"
    )

    print("\n패턴 저장 완료.")

    print("\n#---------------------------------------")
    print("# [3] MAC 결과")
    print("#---------------------------------------")

    score_a = mac_score(
        pattern,
        filter_a
    )

    score_b = mac_score(
        pattern,
        filter_b
    )

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

    average_ms = (
        average_a_ms + average_b_ms
    ) / 2.0

    # --------------------------------------------------------
    # 사용자 입력 모드에서는 A/B로 판정
    # --------------------------------------------------------

    result = classify_scores(
        score_a,
        score_b,
        label_a="A",
        label_b="B",
        epsilon=EPSILON
    )

    print(f"A 점수: {score_a:.10f}")
    print(f"B 점수: {score_b:.10f}")

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

        print(f"판정: {result}")

    print("\n#---------------------------------------")
    print("# [4] 성능 분석")
    print("#---------------------------------------")

    # 사용자 입력 모드에서는 3×3만 측정합니다.
    #
    # 성능 측정에서는 실제 Cross/X 형태의
    # 필터를 각각 생성합니다.

    run_performance_analysis(
        (3,),
        PERFORMANCE_REPEATS
    )


# ============================================================
# 12. JSON 파일 읽기
# ============================================================

def load_json_file(path: str) -> Dict[str, Any]:
    """
    JSON 파일을 읽어 Python 객체로 변환한다.
    """

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_size_from_pattern_key(
    key: str
) -> Optional[int]:
    """
    패턴 키에서 행렬 크기 N을 추출한다.

    size_{N}_{idx}
    """

    match = re.fullmatch(
        r"size_(\d+)_(\d+)",
        key
    )

    if not match:
        return None

    return int(match.group(1))


# ============================================================
# 13. JSON 필터 검증
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
    """

    if not isinstance(filters, dict):
        return (
            False,
            "filters가 객체가 아닙니다.",
            None
        )

    size_key = f"size_{size}"

    if size_key not in filters:
        return (
            False,
            f"{size_key} 필터가 존재하지 않습니다.",
            None
        )

    group = filters[size_key]

    if not isinstance(group, dict):
        return (
            False,
            f"{size_key} 값이 객체가 아닙니다.",
            None
        )

    normalized = {}

    for raw_label in ("cross", "x"):

        if raw_label not in group:
            return (
                False,
                f"{size_key}에 '{raw_label}' 필터가 없습니다.",
                None
            )

        label = normalize_label(raw_label)

        matrix = group[raw_label]

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

        normalized[label] = to_float_matrix(matrix)

    return True, "", normalized


# ============================================================
# 14. JSON 개별 패턴 분석
# ============================================================

def analyze_pattern_case(
    case_id: str,
    case_data: Any,
    filters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    하나의 JSON 패턴 케이스를 분석한다.

    JSON 모드에서는 Cross/X가 실제 분류 라벨이므로
    classify_scores()에
        label_a="Cross"
        label_b="X"
    를 전달한다.
    """

    result = {
        "case_id": case_id,
        "status": "FAIL",
        "reason": "",
        "cross_score": None,
        "x_score": None,
        "prediction": None,
        "expected": None,
    }

    size = extract_size_from_pattern_key(case_id)

    if size is None:
        result["reason"] = (
            "패턴 키 형식 오류: "
            "size_{N}_{idx} 형식이 필요합니다."
        )
        return result

    if size not in SUPPORTED_JSON_SIZES:
        result["reason"] = (
            f"지원하지 않는 크기입니다: {size}x{size}"
        )
        return result

    if not isinstance(case_data, dict):
        result["reason"] = (
            "패턴 데이터가 객체가 아닙니다."
        )
        return result

    if "input" not in case_data:
        result["reason"] = (
            "'input' 필드가 없습니다."
        )
        return result

    if "expected" not in case_data:
        result["reason"] = (
            "'expected' 필드가 없습니다."
        )
        return result

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

    pattern = case_data["input"]

    ok, reason = validate_square_matrix(
        pattern,
        expected_size=size
    )

    if not ok:
        result["reason"] = (
            f"패턴 크기/데이터 오류: {reason}"
        )
        return result

    pattern = to_float_matrix(pattern)

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

    cross_filter = normalized_filters["Cross"]
    x_filter = normalized_filters["X"]

    cross_score = mac_score(
        pattern,
        cross_filter
    )

    x_score = mac_score(
        pattern,
        x_filter
    )

    # --------------------------------------------------------
    # JSON 모드에서는 Cross/X로 판정
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

    if prediction == expected:

        result["status"] = "PASS"
        result["reason"] = "정상 판정"

    else:

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

    return result


# ============================================================
# 15. JSON 분석 모드
# ============================================================

def run_json_mode(
    json_path: str = "data.json"
) -> None:
    """
    data.json을 읽어 모든 패턴 케이스를 분석한다.
    """

    print("\n#---------------------------------------")
    print("# [1] JSON 데이터 로드")
    print("#---------------------------------------")

    if not os.path.exists(json_path):
        print(
            f"파일을 찾을 수 없습니다: {json_path}"
        )
        return

    try:

        data = load_json_file(json_path)

    except json.JSONDecodeError as error:

        print("JSON 파싱 오류:")
        print(error)
        return

    except OSError as error:

        print("파일 읽기 오류:")
        print(error)
        return

    if not isinstance(data, dict):
        print(
            "스키마 오류: "
            "최상위 JSON은 객체여야 합니다."
        )
        return

    filters = data.get("filters")
    patterns = data.get("patterns")

    if not isinstance(filters, dict):
        print(
            "스키마 오류: 'filters'가 없습니다."
        )
        return

    if not isinstance(patterns, dict):
        print(
            "스키마 오류: 'patterns'가 없습니다."
        )
        return

    print("\n#---------------------------------------")
    print("# [2] 필터 로드")
    print("#---------------------------------------")

    valid_filter_sizes = []

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

    print("\n#---------------------------------------")
    print("# [3] 패턴 분석 (라벨 정규화 적용)")
    print("#---------------------------------------")

    total = 0
    passed = 0
    failed = 0

    failures = []

    for case_id, case_data in patterns.items():

        total += 1

        result = analyze_pattern_case(
            case_id,
            case_data,
            filters
        )

        if result["status"] == "PASS":

            passed += 1

        else:

            failed += 1

            failures.append(result)

        print(f"\n--- {case_id} ---")

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

            if result["status"] == "FAIL":

                print(
                    f"원인: {result['reason']}"
                )

        else:

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

    run_performance_analysis(
        (3, 5, 13, 25),
        PERFORMANCE_REPEATS
    )

    # --------------------------------------------------------
    # 결과 요약
    # --------------------------------------------------------

    print("\n#---------------------------------------")
    print("# [5] 결과 요약")
    print("#---------------------------------------")

    print(f"총 테스트: {total}개")
    print(f"통과: {passed}개")
    print(f"실패: {failed}개")

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
# 16. 메인 메뉴
# ============================================================

def print_title() -> None:
    """
    프로그램 제목을 출력한다.
    """

    print("\n=======================================")
    print("        NPU Simulator")
    print("=======================================")


def main() -> None:
    """
    프로그램의 시작점이다.
    """

    print_title()

    while True:

        print("\n[모드 선택]")
        print("1. 사용자 입력 (3x3)")
        print("2. data.json 분석")
        print("0. 종료")

        choice = input("선택: ").strip()

        if choice == "1":

            run_user_mode()

            break

        elif choice == "2":

            json_path = input(
                "data.json 경로 "
                "(기본값: data.json): "
            ).strip()

            if not json_path:
                json_path = "data.json"

            run_json_mode(json_path)

            break

        elif choice == "0":

            print("프로그램을 종료합니다.")

            break

        else:

            print(
                "입력 오류: 1, 2 또는 0을 선택하세요."
            )


# ============================================================
# 프로그램 시작
# ============================================================

if __name__ == "__main__":
    main()


# ============================================================
# 프로그램 구조
# ============================================================
#
# main()
#   │
#   ├─ 모드 1 → run_user_mode()
#   │              │
#   │              ├─ 필터 A 입력
#   │              ├─ 필터 B 입력
#   │              ├─ 패턴 입력
#   │              ├─ mac_score()
#   │              ├─ classify_scores(
#   │              │      score_a,
#   │              │      score_b,
#   │              │      "A",
#   │              │      "B"
#   │              │  )
#   │              └─ A / B 판정
#   │
#   └─ 모드 2 → run_json_mode()
#                  │
#                  ├─ load_json_file()
#                  ├─ validate_filter_group()
#                  ├─ analyze_pattern_case()
#                  │      │
#                  │      ├─ normalize_label()
#                  │      ├─ validate_square_matrix()
#                  │      ├─ mac_score()
#                  │      ├─ classify_scores(
#                  │      │      cross_score,
#                  │      │      x_score,
#                  │      │      "Cross",
#                  │      │      "X"
#                  │      │  )
#                  │      └─ Cross / X 판정
#                  │
#                  ├─ PASS / FAIL 집계
#                  └─ run_performance_analysis()
#
#
# ============================================================
# 핵심 구조
# ============================================================
#
# 사용자 입력 모드
#
#                  패턴
#                    │
#          ┌─────────┴─────────┐
#          │                   │
#       필터 A               필터 B
#          │                   │
#          ▼                   ▼
#       A 점수                B 점수
#          │                   │
#          └─────────┬─────────┘
#                    │
#             classify_scores()
#                    │
#              A / B / UNDECIDED
#
#
# JSON 분석 모드
#
#                  패턴
#                    │
#          ┌─────────┴─────────┐
#          │                   │
#     Cross 필터             X 필터
#          │                   │
#          ▼                   ▼
#    Cross 점수              X 점수
#          │                   │
#          └─────────┬─────────┘
#                    │
#             classify_scores()
#                    │
#          Cross / X / UNDECIDED
#
#
# ============================================================
# 성능 측정 구조
# ============================================================
#
# create_performance_pattern()
#             │
#             ├───────────────┐
#             │               │
#             ▼               ▼
# create_cross_filter()   create_x_filter()
#             │               │
#             ▼               ▼
#        measure_mac()    measure_mac()
#             │               │
#             ▼               ▼
#        Cross 시간        X 시간
#
# ============================================================