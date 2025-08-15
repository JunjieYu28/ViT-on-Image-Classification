import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

script_path = "/home/sichongjie/sichongjie-sub/ViT_torch/train_final.py"  
commands = [
    # [sys.executable, script_path, "--name", "cifar10_No_20", "--model_type", "ViT-Ours_adp0"],
    # [sys.executable, script_path, "--name", "cifar10_No_21", "--model_type", "ViT-Ours_adp1"],
    # [sys.executable, script_path, "--name", "cifar10_No_22", "--model_type", "ViT-Ours_adp2"],
    [sys.executable, script_path, "--name", "cifar10_No_23", "--model_type", "ViT-Ours_adp3"]
]

def run_command(cmd):
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},  # 合并当前环境变量
        text=True
    )
    return {
        "command": cmd,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode
    }

max_workers = 1
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_command = {executor.submit(run_command, cmd): cmd for cmd in commands}
    
    for future in as_completed(future_to_command):
        result = future.result()
        command = result["command"]
        print(f"Command: {' '.join(command)}")
        if result["returncode"] == 0:
            print("STDOUT:", result["stdout"])
        else:
            print("STDERR:", result["stderr"])
            print(f"Command failed with return code {result['returncode']}")
