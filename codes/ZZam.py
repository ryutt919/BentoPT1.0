import numpy as np

# NPZ 파일 로드
data = np.load(r'mmaction2\data\kinetics400\normalized\bench_pressing\bench_press (6).npz')

# NPZ 안에 저장된 배열(키) 목록 출력
print("파일에 저장된 항목들:", data.files)

# 각 항목별 데이터 정보 출력 (형태, 데이터 일부 등)
for key in data.files:
    print(f"--- {key} ---")
    print("Shape:", data[key].shape)
    print("데이터 예시:", data[key].flatten()[:10])  # 1차원으로 펴서 앞 10개만 출력
    print()
