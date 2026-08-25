#!/usr/bin/env python3
"""Power logger for Raspberry Pi 5 using the on-board power management IC
(PMIC). No extra hardware needed -- the Pi 5's PMIC reports voltage and
current per board rail via `vcgencmd pmic_read_adc`.

Usage:
    python3 power_logger.py --out out/power.csv --hz 20
Stop with Ctrl-C. Output CSV: t (unix timestamp), watts (total board power).
"""
import argparse, subprocess, time, sys, re

def read_power():
    out = subprocess.run(["vcgencmd", "pmic_read_adc"],
                         capture_output=True, text=True, timeout=1).stdout
    volts, amps = {}, {}
    for line in out.splitlines():
        line = line.strip()
        m = re.match(r"\s*(\w+)_V volt\(\d+\)=([\d.]+)V", line)
        if m:
            volts[m.group(1)] = float(m.group(2)); continue
        m = re.match(r"\s*(\w+)_A current\(\d+\)=([\d.]+)A", line)
        if m:
            amps[m.group(1)] = float(m.group(2))
    return sum(volts[r] * amps[r] for r in volts if r in amps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--hz", type=float, default=20.0)
    args = ap.parse_args()
    period = 1.0 / args.hz
    with open(args.out, "w") as f:
        f.write("t,watts\n")
        print(f"logging to {args.out} at {args.hz} Hz — Ctrl-C to stop")
        try:
            while True:
                t0 = time.time()
                try:
                    w = read_power()
                    f.write(f"{t0:.4f},{w:.4f}\n")
                except Exception as e:
                    print(f"read error: {e}", file=sys.stderr)
                dt = time.time() - t0
                if dt < period:
                    time.sleep(period - dt)
        except KeyboardInterrupt:
            print("\nstopped")

if __name__ == "__main__":
    main()
