#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import numpy as np


def acc2int(acc):
    acc_map = {"#": 1, "b": -1, "": 0}
    return sum([acc_map[a] for a in acc])


def nat2int(nat):
    return dict(C=0, D=2, E=4, F=5, G=7, A=9, B=11)[nat]


def sitch2pitch(sitch, rest=-1):
    m = re.match(r"^(?P<natural>[A-Ga-g])"
                 r"(?P<accidental>[#b]*)"
                 r"(?P<octave>[+-]?\d+)",
                 sitch)

    if not m: return rest

    natural    = m.group("natural").upper()
    octave     = m.group("octave")
    accidental = m.group("accidental")

    return 12 * (int(octave) + 1) + nat2int(natural) + acc2int(accidental)


def sitchclass2pitchclass(sitchclass):
    m = re.match(r"^(?P<natural>[A-Ga-g])"
                 r"(?P<accidental>[#b]*)"
                 r"(?P<octave>[+-]?\d+)?",
                 sitchclass)

    if not m: return -1

    natural    = m.group("natural").upper()
    accidental = m.group("accidental")

    return (nat2int(natural) + acc2int(accidental)) % 12


def pitch2sitch(pitch, rest="R"):
    if pitch < 0 or pitch > 127: return rest

    pclass = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"][(pitch + 120) % 12]
    return f"{pclass}{pitch // 12 - 1}"


def pitch2cent(pitch, base_pitch=21):
    return 100.0 * (pitch - base_pitch)


def pitch2freq(pitch):
    return 440.0 * (2 ** ((pitch - 69) / 12.0))


def freq2pitch(freq):
    return (12 * np.log2(freq / 440.0) + 69).round()


def cent2pitch(cent, base_pitch=21):
    return (cent / 100.0 + base_pitch).round()


def freq2cent(freq, base_pitch=21):
    return 1200.0 * np.log2(freq / pitch2freq(base_pitch))


def cent2freq(cent, base_pitch=21):
    return (2 ** (cent / 1200.0)) * pitch2freq(base_pitch)
