import numpy as np

npz_path = r"C:\Users\kimt9\Desktop\RyuTTA\2025_3_1\ComputerVision\TermP\mmaction2\data\kinetics400\keypoints\bench pressing\bench_press (12).npz"
data = np.load(npz_path)

print("Archive contains:", data.files)
print("Archive shape:", data['keypoints'].shape)
data.close()
