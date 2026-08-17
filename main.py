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
# 성능 측정에서는 별도의 테스트 패턴을 생성하지 않고,
# 사용자가 입력한 패턴 또는 JSON에 저장된 실제 패턴을
# 그대로 사용합니다.
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
    입력받은 라벨을 프로그램 내부에서 사용하는
    표준 라벨로 변환한다.

    "+" / "cross" / "Cross" -> "Cross"
    "x" / "X"               -> "X"

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

    정상:
        (행 개수, 열 개수)

    오류:
        None
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

    검사:
    1. 2차원 리스트인가?
    2. 행 길이가 모두 같은가?
    3. 정사각형인가?
    4. expected_size와 같은가?
    5. 모든 값이 숫자로 변환 가능한가?
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


def to_float_matrix(
    matrix: List[List[Any]]
) -> List[List[float]]:
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

    같은 위치의 값을 곱한 뒤 모두 더한다.

    시간 복잡도:
        O(N²)
    """

    rows = len(pattern)

    score = 0.0

    for r in range(rows):

        for c in range(rows):

            score += (
                pattern[r][c]
                * filter_matrix[r][c]
            )

    return score


# ============================================================
# 4. 판정(Classification)
# ============================================================

def classify_scores(
    score_a: float,
    score_b: float,
    label_a: str = "A",
    label_b: str = "B",
    epsilon: float = EPSILON
) -> str:
    """
    두 점수를 비교하여
    label_a / label_b / UNDECIDED 중 하나를 반환한다.
    """

    difference = abs(score_a - score_b)

    if difference < epsilon:
        return "UNDECIDED"

    if score_a > score_b:
        return label_a

    return label_b


# ============================================================
# 5. 성능 측정용 Cross 필터 생성
# ============================================================

def create_cross_filter(size: int) -> List[List[float]]:
    """
    성능 측정 또는 테스트에 사용할 Cross 필터를 생성한다.

    단, 실제 성능 평가에서는
    JSON에 저장된 Cross 필터 또는
    사용자가 입력한 필터를 우선 사용한다.
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
            1.0
            if r == center or c == center
            else 0.0
            for c in range(size)
        ]
        for r in range(size)
    ]


# ============================================================
# 6. 성능 측정용 X 필터 생성
# ============================================================

def create_x_filter(size: int) -> List[List[float]]:
    """
    성능 측정 또는 테스트에 사용할 X 필터를 생성한다.

    실제 성능 평가에서는
    JSON에 저장된 X 필터 또는
    사용자가 입력한 필터를 우선 사용한다.
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
# 7. 행렬 출력
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
# 8. MAC 성능 측정
# ============================================================

def measure_mac(
    pattern: List[List[float]],
    filter_matrix: List[List[float]],
    repeats: int = PERFORMANCE_REPEATS
) -> Tuple[float, float]:
    """
    주어진 pattern과 filter_matrix를 사용하여
    MAC 연산 시간을 측정한다.

    반환:
        (
            평균 시간(ms),
            마지막 MAC 결과
        )

    중요:
        여기서는 별도의 패턴을 생성하지 않는다.

        전달받은 pattern을 그대로 사용한다.
    """

    if repeats <= 0:
        raise ValueError(
            "repeats는 1 이상이어야 합니다."
        )

    elapsed_times = []

    last_score = 0.0

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    last_score = mac_score(
        pattern,
        filter_matrix
    )

    # --------------------------------------------------------
    # 실제 측정
    # --------------------------------------------------------

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
# 9. 입력받은 패턴을 이용한 성능 분석
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
    입력받은 실제 패턴을 이용하여 성능을 측정한다.

    --------------------------------------------------------
    기존 방식
    --------------------------------------------------------

    기존에는:

        create_performance_pattern(size)

    을 사용해서 모든 값이 1.0인
    별도의 테스트 패턴을 만들었다.

    --------------------------------------------------------
    변경된 방식
    --------------------------------------------------------

    이제는 호출한 곳에서 전달한

        pattern

    을 그대로 사용한다.

    따라서 실제 분류에 사용한 패턴과
    성능 측정에 사용하는 패턴이 동일하다.

    --------------------------------------------------------
    측정 대상
    --------------------------------------------------------

    pattern + filter_a
    pattern + filter_b

    각각의 MAC 시간을 측정한다.
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

    if filter_a_size != (rows, cols):

        print(
            "성능 분석 오류: "
            f"{label_a} 필터 크기가 패턴과 다릅니다."
        )

        return

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

    # --------------------------------------------------------
    # filter_a 성능 측정
    # --------------------------------------------------------

    time_a, score_a = measure_mac(
        pattern,
        filter_a,
        repeats
    )

    # --------------------------------------------------------
    # filter_b 성능 측정
    # --------------------------------------------------------

    time_b, score_b = measure_mac(
        pattern,
        filter_b,
        repeats
    )

    # N x N이므로 MAC 위치 수는 N²입니다.
    operation_count = rows * cols

    # --------------------------------------------------------
    # 결과 출력
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

    # --------------------------------------------------------
    # 두 필터의 평균 시간
    # --------------------------------------------------------

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

    사용자가 입력한 필터 A/B와 패턴을 사용한다.

    특히 성능 측정에서도
    사용자가 입력한 pattern을 그대로 사용한다.
    """

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
    # [3] MAC 결과
    # --------------------------------------------------------

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

    # MAC 시간 측정
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
    # 판정
    # --------------------------------------------------------

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
    # [4] 성능 분석
    # --------------------------------------------------------

    print("\n#---------------------------------------")
    print("# [4] 성능 분석")
    print("#---------------------------------------")

    # ★ 변경된 부분
    #
    # 기존:
    #
    # run_performance_analysis(
    #     (3,),
    #     PERFORMANCE_REPEATS
    # )
    #
    # 이제:
    #
    # 실제 사용자가 입력한 pattern과
    # 실제 입력한 filter_a/filter_b를 전달합니다.

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
    JSON 파일을 읽어 Python 객체로 변환한다.
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
    패턴 키에서 행렬 크기 N을 추출한다.

    예:

        size_5_0  -> 5
        size_13_3 -> 13
        size_25_10 -> 25
    """

    match = re.fullmatch(
        r"size_(\d+)_(\d+)",
        key
    )

    if not match:
        return None

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

        label = normalize_label(
            raw_label
        )

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

        normalized[label] = to_float_matrix(
            matrix
        )

    return True, "", normalized


# ============================================================
# 13. JSON 개별 패턴 분석
# ============================================================

def analyze_pattern_case(
    case_id: str,
    case_data: Any,
    filters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    하나의 JSON 패턴 케이스를 분석한다.

    MAC 계산뿐만 아니라
    실제 해당 패턴을 사용한 성능 측정 결과도
    함께 반환한다.
    """

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
    # 1. case_id에서 크기 추출
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
    # 2. 지원 크기 확인
    # --------------------------------------------------------

    if size not in SUPPORTED_JSON_SIZES:

        result["reason"] = (
            f"지원하지 않는 크기입니다: {size}x{size}"
        )

        return result

    # --------------------------------------------------------
    # 3. case_data 확인
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 4. expected 정규화
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
    # 5. 패턴 검사
    # --------------------------------------------------------

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

    pattern = to_float_matrix(
        pattern
    )

    # --------------------------------------------------------
    # 6. Cross/X 필터 검사
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

    cross_filter = normalized_filters["Cross"]

    x_filter = normalized_filters["X"]

    # --------------------------------------------------------
    # 7. Cross MAC
    # --------------------------------------------------------

    cross_score = mac_score(
        pattern,
        cross_filter
    )

    # --------------------------------------------------------
    # 8. X MAC
    # --------------------------------------------------------

    x_score = mac_score(
        pattern,
        x_filter
    )

    # --------------------------------------------------------
    # 9. 판정
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
    # 10. PASS / FAIL
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 11. ★ 실제 입력 패턴 성능 측정
    # --------------------------------------------------------
    #
    # 기존에는 run_performance_analysis()에서
    # 별도의 1.0 패턴을 생성했습니다.
    #
    # 이제는 현재 case의 실제 input을
    # 그대로 전달합니다.

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

    average_time_ms = (
        cross_time_ms
        + x_time_ms
    ) / 2.0

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
    data.json을 읽어 모든 패턴 케이스를 분석한다.

    각 패턴의 실제 input을 사용하여
    MAC 성능도 측정한다.
    """

    # --------------------------------------------------------
    # [1] JSON 데이터 로드
    # --------------------------------------------------------

    print("\n#---------------------------------------")
    print("# [1] JSON 데이터 로드")
    print("#---------------------------------------")

    if not os.path.exists(json_path):

        print(
            f"파일을 찾을 수 없습니다: {json_path}"
        )

        return

    try:

        data = load_json_file(
            json_path
        )

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

    filters = data.get(
        "filters"
    )

    patterns = data.get(
        "patterns"
    )

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

    # --------------------------------------------------------
    # [2] 필터 로드
    # --------------------------------------------------------

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
    # [3] 패턴 분석
    # --------------------------------------------------------

    print("\n#---------------------------------------")
    print("# [3] 패턴 분석 + 실제 패턴 성능 측정")
    print("#---------------------------------------")

    total = 0

    passed = 0

    failed = 0

    failures = []

    performance_results = []

    # 모든 JSON 패턴을 하나씩 분석합니다.
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

            failures.append(
                result
            )

        # ----------------------------------------------------
        # 패턴 결과 출력
        # ----------------------------------------------------

        print(
            f"\n--- {case_id} ---"
        )

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

            # ------------------------------------------------
            # 실제 패턴 성능
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

                performance_results.append(
                    result
                )

        else:

            print(
                f"판정: FAIL | "
                f"원인: {result['reason']}"
            )

    # --------------------------------------------------------
    # [4] 성능 분석 결과
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

    if performance_results:

        print(
            f"{'패턴':<18}"
            f"{'크기':<10}"
            f"{'Cross(ms)':>15}"
            f"{'X(ms)':>15}"
            f"{'평균(ms)':>15}"
        )

        print("-" * 75)

        for result in performance_results:

            case_id = result["case_id"]

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
    # [5] 결과 요약
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
    프로그램 제목을 출력한다.
    """

    print(
        "\n======================================="
    )

    print(
        "        NPU Simulator"
    )

    print(
        "======================================="
    )


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

        choice = input(
            "선택: "
        ).strip()

        # ----------------------------------------------------
        # 모드 1
        # ----------------------------------------------------

        if choice == "1":

            run_user_mode()

            break

        # ----------------------------------------------------
        # 모드 2
        # ----------------------------------------------------

        elif choice == "2":

            json_path = input(
                "data.json 경로 "
                "(기본값: data.json): "
            ).strip()

            if not json_path:

                json_path = "data.json"

            run_json_mode(
                json_path
            )

            break

        # ----------------------------------------------------
        # 종료
        # ----------------------------------------------------

        elif choice == "0":

            print(
                "프로그램을 종료합니다."
            )

            break

        # ----------------------------------------------------
        # 잘못된 입력
        # ----------------------------------------------------

        else:

            print(
                "입력 오류: "
                "1, 2 또는 0을 선택하세요."
            )


# ============================================================
# 프로그램 시작
# ============================================================

if __name__ == "__main__":
    main()
