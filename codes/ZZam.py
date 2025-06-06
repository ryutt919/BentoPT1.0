import os
import re

def numeric_key(filename):
    """
    파일명에서 마지막 숫자 덩어리를 추출해 정렬 키로 반환합니다.
    숫자가 없으면 무한대로 처리해 맨 뒤로 배치됩니다.
    """
    name = os.path.splitext(filename)[0]
    all_matches = re.findall(r'\d+', name)
    return int(all_matches[-1]) if all_matches else float('inf')

def validate_npz_folder(folder_path):
    # 1) 폴더가 존재하는지 확인
    if not os.path.isdir(folder_path):
        print(f"Error: 지정된 경로가 없거나 폴더가 아닙니다 → {folder_path}")
        return

    # 2) 디렉토리 내 전체 파일 목록 출력
    all_entries = sorted(os.listdir(folder_path))
    print(f"\n[전체 항목] ({len(all_entries)}개)")
    for fname in all_entries:
        print(f"  {fname}")
    
    # 3) .npz 파일만 필터링 (소문자/대문자 구분 없이)
    npz_files = [f for f in all_entries if f.lower().endswith('.npz')]
    print(f"\n[.npz 확장자 파일] → 총 {len(npz_files)}개")
    
    # 4) 대소문자 구분 없이 .npz 구분했을 때, 소문자 .npz인 것과 아닌 것 구분
    lower_npz = [f for f in npz_files if f.endswith('.npz')]
    upper_npz = [f for f in npz_files if f.endswith('.NPZ')]
    mixed_npz = [f for f in npz_files if not (f.endswith('.npz') or f.endswith('.NPZ'))]
    
    print(f"  - 소문자 .npz : {len(lower_npz)}개")
    print(f"  - 대문자 .NPZ : {len(upper_npz)}개")
    if mixed_npz:
        print(f"  - 기타(.NpZ 등) : {len(mixed_npz)}개 → {mixed_npz}")
    
    # 5) 숫자 기준 정렬해서 출력
    sorted_npz = sorted(npz_files, key=numeric_key)
    print("\n[숫자 기준 정렬된 .npz 목록]")
    for i, fname in enumerate(sorted_npz, 1):
        key = numeric_key(fname)
        print(f"{i:3d}. ({key}) {fname}")
    
    # 6) 누락된 번호 확인 (예: 1~N 사이에 빠진 숫자 찾기)
    #    파일명에서 뽑은 숫자 리스트
    nums = sorted(set(numeric_key(f) for f in sorted_npz if numeric_key(f) != float('inf')))
    if nums:
        full_range = set(range(nums[0], nums[-1] + 1))
        missing = sorted(full_range - set(nums))
        if missing:
            print(f"\n[누락된 번호] {nums[0]} ~ {nums[-1]} 사이에 {len(missing)}개 → {missing}")
        else:
            print(f"\n모든 숫자 ({nums[0]} ~ {nums[-1]})가 존재합니다.")
    else:
        print("\n숫자를 포함한 .npz 파일이 없습니다.")
    
    # 7) 요약
    print(f"\n=== 요약 ===")
    print(f"전체 항목 수           : {len(all_entries)}")
    print(f"확장자 .npz(대소문자 무관) : {len(npz_files)}")
    print(f"  └ 소문자 .npz         : {len(lower_npz)}")
    print(f"  └ 대문자 .NPZ         : {len(upper_npz)}")
    print(f"  └ 기타 확장자(.NpZ 등): {len(mixed_npz)}\n")

if __name__ == "__main__":
    folder = r"C:\Users\kimt9\Desktop\RyuTTA\2025_3_1\ComputerVision\TermP\mmaction2\data\kinetics400\keypoints\bench_pressing"
    validate_npz_folder(folder)
