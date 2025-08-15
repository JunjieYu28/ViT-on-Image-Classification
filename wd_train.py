import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

script_path = "/home/sichongjie/sichongjie-sub/ViT_torch/train_final.py"  
commands = [
    [sys.executable, script_path, "--name", "cifar10_No_10", "--weight_decay", "0.0"],
    [sys.executable, script_path, "--name", "cifar10_No_11", "--weight_decay", "5e-1"],
    [sys.executable, script_path, "--name", "cifar10_No_12", "--weight_decay", "5e-2"],
    [sys.executable, script_path, "--name", "cifar10_No_13", "--weight_decay", "5e-3"],
    [sys.executable, script_path, "--name", "cifar10_No_14", "--weight_decay", "5e-4"],
    [sys.executable, script_path, "--name", "cifar10_No_15", "--weight_decay", "5e-5"]
]

def run_command(cmd):
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "2"},  
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
