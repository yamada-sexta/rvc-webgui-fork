import os
import subprocess

for root, _, files in os.walk("."):
    for file in files:
        if file.endswith(".py"):
            subprocess.run(
                ["black", os.path.join(root, file)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
