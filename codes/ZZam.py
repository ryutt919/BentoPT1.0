import pandas as pd
import os
print(os.getcwd())


# CSV 파일 불러오기
df = pd.read_csv(r'C:\Users\kimt9\Desktop\RyuTTA\2025_3_1\ComputerVision\TermP\mmaction2\data\kinetics400\annotations\kinetics_train.csv')

# 'activity' 컬럼에서 'bench pressing'인 행 필터링
bench_df = df[df['label'] == 'bench pressing']

# 뒤에서 200개 추출 (tail 함수 사용)
last_200 = bench_df.tail(200)
last_200.to_csv('bench_pressing_last_200.csv', index=False)

print("bench_pressing_last_200.csv 파일이 생성되었습니다.")