# =============================================================
# noise_injection.py — Noise Injection Engine
# Evaluasi Robustness Model Sentimen Bahasa Indonesia
# =============================================================

import random
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Callable

VOWELS = set("aiueo")
REPETITION_CHARS = list("aiueoknrsh")
REPETITION_MIN   = 2
REPETITION_MAX   = 4

CHAR_SUBSTITUTION_MAP = {
    "a": ["4", "@", "q"], "e": ["3", "eh"], "i": ["1", "y", "!"],
    "o": ["0", "u"],      "u": ["v", "uw"], "s": ["z", "5"],
    "g": ["9", "q"],      "k": ["q", "c"],  "n": ["m", "ng"],
    "t": ["7", "th"],
}

SLANG_MAP = {
    "tidak": ["gak","ga","gk","nggak","enggak","ndak"],
    "kamu" : ["km","lo","elu","loe"],
    "saya" : ["gue","gw","aku","w"],
    "sangat": ["bgt","banget","poll","parah"],
    "dengan": ["dgn","ama","sama"],
    "yang"  : ["yg","yng"],
    "sudah" : ["udah","udh","dah"],
    "belum" : ["blm","blom","belom"],
    "karena": ["krn","karna","soalnya"],
    "mereka": ["mrk","mereka"],
    "juga"  : ["jg","jugak","pun"],
    "tapi"  : ["tp","tpi","tapii"],
    "untuk" : ["utk","buat","bwt"],
    "kalau" : ["klo","kalo","klw"],
    "bagus" : ["oke","ok","mantap","mantul","mantab"],
    "jelek" : ["parah","ancur","rusak","boo"],
    "bisa"  : ["bs","bsa","iso"],
    "mau"   : ["mo","mw","pengen"],
    "ada"   : ["ad","ade"],
    "apa"   : ["ap","apaan","apasih"],
}

KEYBOARD_PROXIMITY_MAP = {
    "q":["w","a"],       "w":["q","e","a","s"],   "e":["w","r","s","d"],
    "r":["e","t","d","f"],"t":["r","y","f","g"],  "y":["t","u","g","h"],
    "u":["y","i","h","j"],"i":["u","o","j","k"],  "o":["i","p","k","l"],
    "p":["o","l"],        "a":["q","w","s","z"],   "s":["a","w","e","d","z","x"],
    "d":["s","e","r","f","x","c"],"f":["d","r","t","g","c","v"],
    "g":["f","t","y","h","v","b"],"h":["g","y","u","j","b","n"],
    "j":["h","u","i","k","n","m"],"k":["j","i","o","l","m"],
    "l":["k","o","p"],    "z":["a","s","x"],       "x":["z","s","d","c"],
    "c":["x","d","f","v"],"v":["c","f","g","b"],   "b":["v","g","h","n"],
    "n":["b","h","j","m"],"m":["n","j","k"],
}

def apply_char_substitution(word):
    chars = list(word)
    candidates = [(i,c) for i,c in enumerate(chars) if c in CHAR_SUBSTITUTION_MAP]
    if not candidates: return word
    idx, char = random.choice(candidates)
    chars[idx] = random.choice(CHAR_SUBSTITUTION_MAP[char])
    return "".join(chars)

def apply_vowel_removal(word):
    if len(word) < 4: return word
    result = word[0]
    for char in word[1:-1]:
        if char not in VOWELS: result += char
    result += word[-1]
    return result if len(result) >= 2 else word

def apply_slang_replacement(word):
    return random.choice(SLANG_MAP[word]) if word in SLANG_MAP else word

def apply_char_repetition(word):
    if len(word) < 3: return word
    chars = list(word)
    candidates = [i for i,c in enumerate(chars[1:],start=1) if c in REPETITION_CHARS]
    if not candidates: return word
    idx = random.choice(candidates)
    chars[idx] = chars[idx] * random.randint(REPETITION_MIN, REPETITION_MAX)
    return "".join(chars)

def apply_keyboard_proximity(word):
    chars = list(word)
    candidates = [(i,c) for i,c in enumerate(chars) if c in KEYBOARD_PROXIMITY_MAP]
    if not candidates: return word
    idx, char = random.choice(candidates)
    chars[idx] = random.choice(KEYBOARD_PROXIMITY_MAP[char])
    return "".join(chars)

def apply_char_deletion(word):
    if len(word) < 4: return word
    idx = random.randint(1, len(word)-2)
    return word[:idx] + word[idx+1:]

NOISE_FUNCTIONS = [
    ("char_substitution",  apply_char_substitution,  0.20),
    ("vowel_removal",      apply_vowel_removal,       0.20),
    ("slang_replacement",  apply_slang_replacement,   0.25),
    ("char_repetition",    apply_char_repetition,     0.15),
    ("keyboard_proximity", apply_keyboard_proximity,  0.10),
    ("char_deletion",      apply_char_deletion,       0.10),
]
NOISE_NAMES   = [n for n,_,_ in NOISE_FUNCTIONS]
NOISE_WEIGHTS = [w for _,_,w in NOISE_FUNCTIONS]
NOISE_FUNCS   = [f for _,f,_ in NOISE_FUNCTIONS]

def inject_noise(text, noise_level, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    tokens   = text.split()
    n_tokens = len(tokens)
    if n_tokens == 0:
        return text, {"n_tokens": 0, "n_noised": 0, "noise_types": {}}
    n_to_noise  = max(1, int(np.ceil(n_tokens * noise_level)))
    noise_idxs  = sorted(random.sample(range(n_tokens), min(n_to_noise, n_tokens)))
    noisy_tokens = tokens.copy()
    noise_log    = defaultdict(int)
    for idx in noise_idxs:
        original    = tokens[idx]
        noise_func  = random.choices(NOISE_FUNCS, weights=NOISE_WEIGHTS, k=1)[0]
        func_name   = NOISE_NAMES[NOISE_FUNCS.index(noise_func)]
        noisy       = noise_func(original)
        if noisy != original:
            noisy_tokens[idx] = noisy
            noise_log[func_name] += 1
    stats = {
        "n_tokens"   : n_tokens,
        "n_noised"   : sum(noise_log.values()),
        "actual_rate": sum(noise_log.values()) / n_tokens if n_tokens > 0 else 0,
        "noise_types": dict(noise_log),
    }
    return " ".join(noisy_tokens), stats
