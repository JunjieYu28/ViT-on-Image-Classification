import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import signal

script_path = "/home/sichongjie/sichongjie-sub/ViT_torch/train_final.py"  
commands = [
    [sys.executable, script_path, "--name", "cifar10_No_30", "--model_type", "ViT-Ours_nb4"],
    [sys.executable, script_path, "--name", "cifar10_No_31", "--model_type", "ViT-Ours_nb12"]
]

processes = []

def run_command(cmd):
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
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
