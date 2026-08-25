#!/usr/bin/env python3
"""Regenerate data/commands.jsonl (333 smart-home commands, seeded)."""
import json, random, itertools, pathlib

random.seed(20260803)
ROOMS = ["living room", "bedroom", "kitchen", "office", "hallway", "bathroom"]
ROOM_KEY = {r: r.replace(" ", "_") for r in ROOMS}
FILLERS_PRE = ["", "hey, ", "um, ", "please ", "could you ", "can you ", "hey can you "]
FILLERS_POST = ["", " please", " thanks", " for me", " right now", " when you get a chance"]

def mk(utt, call, family, corr=False):
    return {"utterance": utt.strip().capitalize(), "ground_truth": call,
            "family": family, "has_correction": corr}

examples = []
for room, pct in itertools.product(ROOMS, [10, 20, 25, 30, 40, 50, 60, 75, 80]):
    pre, post = random.choice(FILLERS_PRE), random.choice(FILLERS_POST)
    examples.append(mk(f"{pre}dim the {room} lights to {pct}%{post}",
                       f"dim({ROOM_KEY[room]}, {pct})", "lights_dim"))
for room, state in itertools.product(ROOMS, ["on", "off"]):
    pre, post = random.choice(FILLERS_PRE), random.choice(FILLERS_POST)
    examples.append(mk(f"{pre}turn {state} the {room} lights{post}",
                       f"lights_{state}({ROOM_KEY[room]})", "lights_switch"))
for temp in range(17, 27):
    pre = random.choice(FILLERS_PRE)
    examples.append(mk(f"{pre}set the thermostat to {temp} degrees", f"set_temp({temp})", "thermostat"))
    examples.append(mk(f"{pre}make it {temp} degrees in here", f"set_temp({temp})", "thermostat"))
for door in ["front door", "back door", "garage"]:
    for act in ["lock", "unlock"]:
        pre, post = random.choice(FILLERS_PRE), random.choice(FILLERS_POST)
        examples.append(mk(f"{pre}{act} the {door}{post}", f"{act}({door.replace(' ', '_')})", "locks"))
for room, act in itertools.product(ROOMS, ["open", "close"]):
    pre = random.choice(FILLERS_PRE)
    examples.append(mk(f"{pre}{act} the blinds in the {room}", f"blinds_{act}({ROOM_KEY[room]})", "blinds"))
for vol in [3, 5, 7, 10, 12, 15, 18, 20]:
    examples.append(mk(f"set the speaker volume to {vol}", f"set_volume({vol})", "media"))
for act in ["play", "pause", "stop"]:
    pre = random.choice(FILLERS_PRE)
    examples.append(mk(f"{pre}{act} the music", f"music_{act}()", "media"))
for mins in [5, 10, 15, 20, 25, 30, 45, 60, 90]:
    pre = random.choice(FILLERS_PRE)
    examples.append(mk(f"{pre}set a timer for {mins} minutes", f"set_timer({mins})", "timers"))
for room, spd in itertools.product(ROOMS[:4], ["low", "medium", "high"]):
    examples.append(mk(f"set the {room} fan to {spd}", f"fan({ROOM_KEY[room]}, {spd})", "fan"))
for room, pct in itertools.product(ROOMS, [15, 35, 45, 55, 65, 70, 85, 90]):
    v = random.choice([f"set the {room} lights to {pct} percent", f"lights in the {room} down to {pct}%",
                       f"i want the {room} at {pct}% brightness", f"bring the {room} lighting to {pct}%"])
    examples.append(mk(v, f"dim({ROOM_KEY[room]}, {pct})", "lights_dim"))
for room, state in itertools.product(ROOMS, ["on", "off"]):
    v = random.choice([f"switch {state} the lights in the {room}", f"{room} lights {state}",
                       f"kill the {room} lights" if state == "off" else f"lights {state} in the {room}"])
    examples.append(mk(v, f"lights_{state}({ROOM_KEY[room]})", "lights_switch"))
for door in ["front door", "back door", "garage"]:
    examples.append(mk(f"is the {door} locked? if not lock it", f"lock({door.replace(' ', '_')})", "locks"))
    examples.append(mk(f"secure the {door}", f"lock({door.replace(' ', '_')})", "locks"))
for mins in [2, 3, 7, 12, 35, 40, 50, 75, 120]:
    v = random.choice([f"timer, {mins} minutes", f"remind me in {mins} minutes", f"start a {mins} minute timer"])
    examples.append(mk(v, f"set_timer({mins})", "timers"))
for vol in [1, 2, 4, 6, 8, 9, 11, 13, 14, 16]:
    examples.append(mk(f"volume {vol}", f"set_volume({vol})", "media"))
for room, spd in itertools.product(ROOMS[2:], ["low", "medium", "high"]):
    examples.append(mk(f"put the {room} fan on {spd}", f"fan({ROOM_KEY[room]}, {spd})", "fan"))

corrections = []
for _ in range(60):
    room = random.choice(ROOMS)
    p1, p2 = random.sample([10, 20, 30, 40, 50, 60, 75, 80], 2)
    corrections.append(mk(f"turn off the {room} lights... actually wait, just dim them to {p2}%, "
                          f"the kids are still reading", f"dim({ROOM_KEY[room]}, {p2})", "lights_dim", corr=True))
    t1, t2 = random.sample(range(18, 25), 2)
    corrections.append(mk(f"set the heat to {t1}... no hold on, make it {t2} degrees",
                          f"set_temp({t2})", "thermostat", corr=True))
examples.extend(corrections[:100])

random.shuffle(examples)
out = pathlib.Path(__file__).parent.parent.parent / "data" / "commands.jsonl"
out.parent.mkdir(exist_ok=True)
with open(out, "w") as f:
    for i, ex in enumerate(examples):
        ex["id"] = f"cmd_{i:04d}"
        f.write(json.dumps(ex) + "\n")
print(f"wrote {len(examples)} examples to {out}")
