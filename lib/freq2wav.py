#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import math
import argparse
import numpy as np
import soundfile as sf
import time

PI = math.pi


class Sine:
    @staticmethod
    def amplitude(_):
        return 1

    @staticmethod
    def harmonic(k):
        return k


class Triangle:
    @staticmethod
    def amplitude(k):
        return 8.0 / (PI * PI) * math.sin((2*k-1) * PI / 2.0) / ((2*k-1) * (2*k-1))

    @staticmethod
    def harmonic(k):
        return 2*k-1


class Square:
    @staticmethod
    def amplitude(k):
        return 4.0 / (PI * (2*k-1))

    @staticmethod
    def harmonic(k):
        return 2*k-1


class Sawtooth:
    @staticmethod
    def amplitude(k):
        return 2.0 / (PI * k)

    @staticmethod
    def harmonic(k):
        return k


def reset_cumsum(xs):
    cs = np.cumsum(xs)
    return cs - np.maximum.accumulate(cs * (xs == 0))

def synthesize(f0s, samplerate, frametime, wavetype, numharmonics):
    times = []
    times.append(time.time())

    F = len(f0s)
    S = int(samplerate * frametime / 1000.0)

    delta_t = np.tile(np.arange(S), F)
    times.append(time.time())

    waveform = dict(
        sine     = Sine,
        triangle = Triangle,
        square   = Square,
        sawtooth = Sawtooth
    )[wavetype]
    times.append(time.time())

    wav = np.zeros(F * S, dtype=np.float32)
    times.append(time.time())

    for k in range(1, numharmonics+1):
        A  = waveform.amplitude(k)
        times.append(time.time())

        w  = 2.0 * PI * f0s * waveform.harmonic(k)
        times.append(time.time())

        theta  = np.repeat(w, S) * delta_t
        times.append(time.time())

        theta0 = np.repeat((reset_cumsum(w) - w) * S, S)
        times.append(time.time())

        wav += A * np.sin((theta + theta0) / samplerate)
        times.append(time.time())

    print(*[f"{t:.10f} [sec]\n" for t in np.diff(times)])

    max_amp = np.max(np.abs(wav))
    if max_amp > 0:
        wav /= max_amp

    return wav


def main(args):
    parser = argparse.ArgumentParser(description="WAV Synthesizer of an F0 Sequence.")
    parser.add_argument("f_f0s", type=str, help="Filename of an input F0 sequence.")
    parser.add_argument("f_wav", type=str, help="Filename of an output WAV.")
    parser.add_argument("-r", type=int, default=44100,
                        help="Sample rate. (default: 44100)")
    parser.add_argument("-c", type=int, default=1,
                        help="The number of channels. (default: 1)")
    parser.add_argument("-f", type=int, default=10,
                        help="Milliseconds corresponding to one frame. (default: 10)")
    parser.add_argument("-w", type=str, default="sine",
                        choices=["sine", "triangle", "square", "sawtooth"],
                        help="Type of a synthesized wave.")
    parser.add_argument("-k", type=int, default=1,
                        help="The number of harmonics. (default: 1)")
    args = parser.parse_args(args)

    print(f"Convert:   \"{args.f_f0s}\" -> \"{args.f_wav}\"")
    print(f"Channels:   {args.c}")
    print(f"Samplerate: {args.r} [Hz]")
    print(f"Frametime:  {args.f} [ms]")
    print(f"Wavetype:   {args.w}")
    print(f"Harmonics:  {args.k}")

    f0s = np.loadtxt(args.f_f0s)
    wav = synthesize(f0s, args.r, args.f, args.w, args.k)

    print("Writing...")
    sf.write(args.f_wav, wav, args.r, subtype="FLOAT")


if __name__ == "__main__":
    main(sys.argv[1:])
