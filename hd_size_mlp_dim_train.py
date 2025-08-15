import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import signal

script_path = "/home/sichongjie/sichongjie-sub/ViT_torch/train_final.py"  
commands = [
    [sys.executable, script_path, "--name", "cifar10_No_34", "--model_type", "ViT-Ours_set_288_288"],
    [sys.executable, script_path, "--name", "cifar10_No_35", "--model_type", "ViT-Ours_set_288_384"],
    [sys.executable, script_path, "--name", "cifar10_No_36", "--model_type", "ViT-Ours_set_288_768"],
    [sys.executable, script_path, "--name", "cifar10_No_37", "--model_type", "ViT-Ours_set_384_768"]
]

processes = []

def run_command(cmd):
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "2"},
        text=True
    )
    processes.append(process)  
    stdout, stderr = process.communicate()
    return {
        "command": cmd,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": process.returncode
    }

try:
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(run_command, cmd): cmd for cmd in commands}
        for future in as_completed(futures):
            result = future.result()
            command = result["command"]
            print(f"Command: {' '.join(command)}")
            if result["returncode"] == 0:
                print("STDOUT:", result["stdout"])
            else:
                print("STDERR:", result["stderr"])
                print(f"Command failed with return code {result['returncode']}")
except KeyboardInterrupt:
    print("KeyboardInterrupt detected. Terminating all child processes...")
    for process in processes:
        os.kill(process.pid, signal.SIGKILL)  
