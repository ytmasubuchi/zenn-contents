"""実験B2のドライバ"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run(lib):
    cmd = [sys.executable, os.path.join(HERE, "exp_b2_single.py"), "--lib", lib]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main():
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    for lib in ["pandas", "polars"]:
        data = run(lib)
        out_path = os.path.join(ROOT, "results", f"exp_b2_{lib}.json")
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"done: {lib} -> loop_total={data['loop_add_time_total_sec']*1000:.1f}ms "
              f"batch={data['batch_add_time_sec']*1000:.1f}ms "
              f"warning_seen={data.get('fragmentation_warning_seen')}")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
