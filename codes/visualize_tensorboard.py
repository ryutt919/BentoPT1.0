import os
from pathlib import Path
from datetime import datetime
import tensorflow as tf
import webbrowser
import subprocess
import time
import signal
import sys

def find_latest_tensorboard_dir():
    """가장 최근의 텐서보드 로그 디렉토리 찾기"""
    base_dir = Path(r"C:\Users\user\Desktop\TermP\mmaction2\work_dirs\custom_stgcnpp")
    
    # 날짜_시간 형식의 디렉토리 찾기
    exp_dirs = []
    for d in base_dir.iterdir():
        if d.is_dir() and len(d.name) == 15 and d.name[0].isdigit():  # YYYYMMDD_HHMMSS 형식 체크
            try:
                datetime.strptime(d.name, "%Y%m%d_%H%M%S")
                exp_dirs.append(d)
            except ValueError:
                continue
    
    if not exp_dirs:
        raise FileNotFoundError("텐서보드 로그 디렉토리를 찾을 수 없습니다.")
    
    # 가장 최근 디렉토리 찾기
    latest_dir = max(exp_dirs, key=lambda x: datetime.strptime(x.name, "%Y%m%d_%H%M%S"))
    return latest_dir

def start_tensorboard(logdir, port=6006):
    """텐서보드 서버 시작"""
    print(f"텐서보드 시작 중... (로그 디렉토리: {logdir})")
    
    # 이미 실행 중인 텐서보드 프로세스 종료
    if sys.platform == 'win32':
        os.system('taskkill /f /im "tensorboard.exe" 2>nul')
    else:
        os.system('pkill -f "tensorboard"')
    
    # 새로운 텐서보드 프로세스 시작
    cmd = f"tensorboard --logdir={logdir} --port={port}"
    process = subprocess.Popen(cmd, shell=True)
    
    # 서버가 시작될 때까지 잠시 대기
    time.sleep(3)
    
    # 브라우저에서 텐서보드 열기
    url = f"http://localhost:{port}"
    print(f"텐서보드 URL: {url}")
    webbrowser.open(url)
    
    return process

def main():
    try:
        # 최신 텐서보드 로그 디렉토리 찾기
        log_dir = find_latest_tensorboard_dir()
        print(f"최신 실험 디렉토리: {log_dir}")
        
        # 텐서보드 시작
        tensorboard_process = start_tensorboard(log_dir)
        
        print("\n텐서보드가 실행 중입니다...")
        print("종료하려면 Ctrl+C를 누르세요.")
        
        # 프로세스가 실행 중일 때까지 대기
        try:
            tensorboard_process.wait()
        except KeyboardInterrupt:
            print("\n텐서보드를 종료합니다...")
            tensorboard_process.terminate()
            
    except Exception as e:
        print(f"에러 발생: {str(e)}")

if __name__ == '__main__':
    main() 