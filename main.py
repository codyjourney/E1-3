import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# NPU Simulator
# Python 3.12+
# External libraries: None
# ============================================================

EPSILON = 1e-9
MIN_SIZE = 3
SUPPORTED_JSON_SIZES = (5, 13, 25)
PERFORMANCE_REPEATS = 10


# ------------------------------------------------------------
# Label normalization
# ------------------------------------------------------------

def normalize_label(label: Any) -> Optional[str]:
    """
    입력 라벨을 프로그램 내부의 표준 라벨로 변환한다.

    표준 라벨:
        Cross
        X

    지원 입력:
        expected: '+' -> Cross, 'x' -> X
        filter key: 'cross' -> Cross, 'x' -> X
    """
    if not isinstance(label, str):
        return None

    value = label.strip().lower()

    if value in ("+", "cross"):
        return "Cross"

    if value in ("x",):
        return "X"

    return None


# ------------------------------------------------------------
# Matrix validation
# ------------------------------------------------------------

def is_matrix(value: Any) -> bool:
    """2차원 리스트 형태인지 확인한다."""
    if not isinstance(value, list) or len(value) == 0:
        return False

    if not all(isinstance(row, list) for row in value):
        return False

    return True


def matrix_size(matrix: Any) -> Optional[Tuple[int, int]]:
    """행 x 열 크기를 반환한다."""
    if not is_matrix(matrix):
        return None

    rows = len(matrix)

    if rows == 0:
        return None

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
    정사각형 2차원 배열인지 검사한다.
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

    # 모든 값이 숫자로 변환 가능한지 확인
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
    """행렬의 값을 float으로 변환한다."""
    return [
        [float(value) for value in row]
        for row in matrix
    ]


# ------------------------------------------------------------
# MAC operation
# ------------------------------------------------------------

def mac_score(
    pattern: List[List[float]],
    filter_matrix: List[List[float]]
) -> float:
    """
    패턴과 필터의 MAC 점수를 계산한다.

    MAC:
        score = sum(pattern[r][c] * filter[r][c])

    외부 라이브러리 없이 반복문으로 직접 구현한다.
    """
    rows = len(pattern)

    score = 0.0

    for r in range(rows):
        for c in range(rows):
            score += pattern[r][c] * filter_matrix[r][c]

    return score


# ------------------------------------------------------------
# Classification
# ------------------------------------------------------------

def classify_scores(
    cross_score: float,
    x_score: float,
    epsilon: float = EPSILON
) -> str:
    """
    두 점수를 비교하여 Cross / X / UNDECIDED를 반환한다.

    |Cross - X| < epsilon 이면 동점으로 처리한다.
    """
    difference = abs(cross_score - x_score)

    if difference < epsilon:
        return "UNDECIDED"

    if cross_score > x_score:
        return "Cross"

    return "X"


# ------------------------------------------------------------
# Performance measurement
# ------------------------------------------------------------

def measure_mac(
    pattern: List[List[float]],
    filter_matrix: List[List[float]],
    repeats: int = PERFORMANCE_REPEATS
) -> Tuple[float, float]:
    """
    MAC 연산만 반복 측정한다.

    반환:
        (평균 시간 ms, 마지막 결과)
    """
    elapsed_times = []
    last_score = 0.0

    # 함수 호출 자체의 준비시간 영향을 줄이기 위해
    # 실제 측정 전 1회 실행
    last_score = mac_score(pattern, filter_matrix)

    for _ in range(repeats):
        start = time.perf_counter()

        last_score = mac_score(pattern, filter_matrix)

        end = time.perf_counter()

        elapsed_times.append((end - start) * 1000.0)

    average_ms = sum(elapsed_times) / len(elapsed_times)

    return average_ms, last_score


def create_performance_matrix(size: int) -> List[List[float]]:
    """
    성능 측정용 N x N 행렬을 생성한다.
    모든 원소가 1.0이다.
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
    지정된 크기에 대해 MAC 성능을 측정한다.
    """
    print("\n#---------------------------------------")
    print(f"# [성능 분석] 평균/{repeats}회")
    print("#---------------------------------------")

    print(
        f"{'크기':<12}"
        f"{'평균 시간(ms)':>18}"
        f"{'연산 횟수(N²)':>18}"
    )
    print("-" * 48)

    for size in sizes:
        pattern = create_performance_matrix(size)
        filter_matrix = create_performance_matrix(size)

        average_ms, _ = measure_mac(
            pattern,
            filter_matrix,
            repeats
        )

        operation_count = size * size

        print(
            f"{size}x{size:<8}"
            f"{average_ms:>18.6f}"
            f"{operation_count:>18}"
        )


# ------------------------------------------------------------
# User input mode
# ------------------------------------------------------------

def read_matrix_from_console(
    size: int,
    matrix_name: str
) -> List[List[float]]:
    """
    콘솔에서 N x N 행렬을 한 줄씩 입력받는다.

    잘못된 입력이면 재입력을 요구한다.
    """
    while True:
        print(f"\n{matrix_name} ({size}줄 입력, 공백 구분)")

        matrix = []
        valid = True

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
                    row = [float(value) for value in parts]
                except ValueError:
                    print(
                        "입력 형식 오류: 숫자만 입력하세요."
                    )
                    continue

                matrix.append(row)
                break

        # 최종 검증
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
    모드 1:
    3x3 필터 A/B와 패턴을 입력받아 MAC 및 판정을 수행한다.
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

    # 실제 점수 계산
    score_a = mac_score(pattern, filter_a)
    score_b = mac_score(pattern, filter_b)

    # 시간 측정
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

    average_ms = (average_a_ms + average_b_ms) / 2.0

    result = classify_scores(
        score_a,
        score_b,
        EPSILON
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

    run_performance_analysis(
        (3,),
        PERFORMANCE_REPEATS
    )


# ------------------------------------------------------------
# JSON loading
# ------------------------------------------------------------

def load_json_file(path: str) -> Dict[str, Any]:
    """JSON 파일을 읽는다."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_size_from_pattern_key(key: str) -> Optional[int]:
    """
    size_{N}_{idx} 형태의 패턴 키에서 N을 추출한다.

    예:
        size_5_1 -> 5
        size_13_2 -> 13
        size_25_1 -> 25
    """
    match = re.fullmatch(r"size_(\d+)_(\d+)", key)

    if not match:
        return None

    return int(match.group(1))


def validate_filter_group(
    filters: Any,
    size: int
) -> Tuple[bool, str, Optional[Dict[str, List[List[float]]]]]:
    """
    size_N 필터 그룹의 스키마를 검사한다.
    """
    if not isinstance(filters, dict):
        return False, "filters가 객체가 아닙니다.", None

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


# ------------------------------------------------------------
# JSON pattern analysis
# ------------------------------------------------------------

def analyze_pattern_case(
    case_id: str,
    case_data: Any,
    filters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    하나의 JSON 패턴 케이스를 분석한다.

    항상 예외를 외부로 던지지 않고
    케이스 단위 결과를 반환한다.
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

    # 패턴 키 검사
    size = extract_size_from_pattern_key(case_id)

    if size is None:
        result["reason"] = (
            "패턴 키 형식 오류: "
            "size_{N}_{idx} 형식이 필요합니다."
        )
        return result

    # 지원 크기 검사
    if size not in SUPPORTED_JSON_SIZES:
        result["reason"] = (
            f"지원하지 않는 크기입니다: {size}x{size}"
        )
        return result

    # case_data 객체 검사
    if not isinstance(case_data, dict):
        result["reason"] = "패턴 데이터가 객체가 아닙니다."
        return result

    if "input" not in case_data:
        result["reason"] = "'input' 필드가 없습니다."
        return result

    if "expected" not in case_data:
        result["reason"] = "'expected' 필드가 없습니다."
        return result

    # expected 정규화
    expected = normalize_label(case_data["expected"])

    if expected is None:
        result["reason"] = (
            f"알 수 없는 expected 라벨: "
            f"{case_data['expected']!r}"
        )
        return result

    result["expected"] = expected

    # 패턴 검증
    pattern = case_data["input"]

    ok, reason = validate_square_matrix(
        pattern,
        expected_size=size
    )

    if not ok:
        result["reason"] = f"패턴 크기/데이터 오류: {reason}"
        return result

    pattern = to_float_matrix(pattern)

    # 해당 크기의 필터 선택
    ok, reason, normalized_filters = validate_filter_group(
        filters,
        size
    )

    if not ok:
        result["reason"] = f"필터 오류: {reason}"
        return result

    cross_filter = normalized_filters["Cross"]
    x_filter = normalized_filters["X"]

    # MAC
    cross_score = mac_score(
        pattern,
        cross_filter
    )

    x_score = mac_score(
        pattern,
        x_filter
    )

    prediction = classify_scores(
        cross_score,
        x_score,
        EPSILON
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
                f"expected={expected}, prediction={prediction}"
            )

    return result


# ------------------------------------------------------------
# JSON mode
# ------------------------------------------------------------

def run_json_mode(json_path: str = "data.json") -> None:
    """
    모드 2:
    data.json을 읽어 모든 패턴 케이스를 분석한다.
    """
    print("\n#---------------------------------------")
    print("# [1] JSON 데이터 로드")
    print("#---------------------------------------")

    if not os.path.exists(json_path):
        print(f"파일을 찾을 수 없습니다: {json_path}")
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
        print("스키마 오류: 최상위 JSON은 객체여야 합니다.")
        return

    filters = data.get("filters")
    patterns = data.get("patterns")

    if not isinstance(filters, dict):
        print("스키마 오류: 'filters'가 없습니다.")
        return

    if not isinstance(patterns, dict):
        print("스키마 오류: 'patterns'가 없습니다.")
        return

    # --------------------------------------------------------
    # Filter load report
    # --------------------------------------------------------

    print("\n#---------------------------------------")
    print("# [2] 필터 로드")
    print("#---------------------------------------")

    valid_filter_sizes = []

    for size in SUPPORTED_JSON_SIZES:
        ok, reason, normalized_filters = validate_filter_group(
            filters,
            size
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
    # Pattern analysis
    # --------------------------------------------------------

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
    # Performance
    # --------------------------------------------------------

    print("\n#---------------------------------------")
    print("# [4] 성능 분석")
    print("#---------------------------------------")

    run_performance_analysis(
        (3, 5, 13, 25),
        PERFORMANCE_REPEATS
    )

    # --------------------------------------------------------
    # Summary
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


# ------------------------------------------------------------
# Main menu
# ------------------------------------------------------------

def print_title() -> None:
    print("\n=======================================")
    print("        NPU Simulator")
    print("=======================================")


def main() -> None:
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


if __name__ == "__main__":
    main()
