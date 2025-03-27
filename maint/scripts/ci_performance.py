import subprocess
import re
from tabulate import tabulate

import os

env = os.environ.copy()
env["TILELANG_CLEAR_CACHE"] = "1"


def parse_output(output):
    data = {}
    for line in output.split('\n'):
        line = line.strip()
        if line.startswith('Latency:'):
            match = re.search(r'Latency: ([\d.]+)', line)
            data['latency'] = match.group(1) if match else 'N/A'
        elif line.startswith('TFlops:'):
            match = re.search(r'TFlops: ([\d.]+)', line)
            data['best_tflops'] = match.group(1) if match else 'N/A'
        elif line.startswith('Config:'):
            data['config'] = line.split('Config: ')[-1]
        elif line.startswith('Reference TFlops:'):
            match = re.search(r'Reference TFlops: ([\d.]+)', line)
            data['ref_tflops'] = match.group(1) if match else 'N/A'
    return data


output_v1 = subprocess.run(['./tl/bin/python', './maint/scripts/performance.py'],
                           capture_output=True,
                           text=True,
                           env=env).stdout
data_v1 = parse_output(output_v1)

output_v2 = subprocess.run(['./tll/bin/python', './maint/scripts/performance.py'],
                           capture_output=True,
                           text=True,
                           env=env).stdout
data_v2 = parse_output(output_v2)

table = [[
    "original", data_v1['latency'], data_v1['best_tflops'], data_v1['ref_tflops'], data_v1['config']
], [
    "current", data_v2['latency'], data_v2['best_tflops'], data_v2['ref_tflops'], data_v2['config']
]]

headers = ["version", "Best Latency (s)", "Best TFlops", "Reference TFlops", "Best Config"]

print(tabulate(table, headers=headers, tablefmt="github", stralign="left", numalign="decimal"))
