#!/usr/bin/env python
# -*- coding: utf-8 -*-
#version 1 by Ryo Nishikimi

import os
import sys
import math
import enum
import copy
import operator
import functools
import librosa
import time
import subprocess as sp
import soundfile as sf
import uuid
from toolz.itertoolz import sliding_window
from collections.abc import Iterable
from addict import Dict

import numpy as np
import matplotlib.pyplot as plt
import pretty_midi as pm

#from .pitch_converter import sitch2pitch, sitchclass2pitchclass, pitch2sitch
#from .freq2wav import synthesize as freq2wav
from pitch_converter import sitch2pitch, sitchclass2pitchclass, pitch2sitch
from freq2wav import synthesize as freq2wav


def run_bash(cmd, comment=None):
    cmd = cmd.replace("(", "\\(").replace(")", "\\)")
    print("+", cmd)
    try:
        sp.run(cmd, shell=True, check=True)
    except Exception:
        if comment is not None:
            print(f"Error in '{comment}'")
        sys.exit()


class Event:
    class Type(enum.Enum):
        INSTR = enum.auto()
        KEY   = enum.auto()
        METER = enum.auto()
        SPQN  = enum.auto()
        CLEF  = enum.auto()
        NOTE  = enum.auto()
        EMPTY = enum.auto()
        CHORD = enum.auto()

    def __init__(self, evt_type):
        self.evt_type = evt_type

        # discriminators: Event.is_note(), Event.is_instr(), ...
        for et in Event.Type:
            self.__set_evt_type_discriminator(et)

    def __set_evt_type_discriminator(self, evt_type):
        setattr(self, f"is_{evt_type.name.lower()}", lambda: self.evt_type == evt_type)


class InstrEvt(Event):
    def __init__(self, ch, pc, name):
        super().__init__(Event.Type.INSTR)

        self.ch = int(ch)
        self.pc = int(pc)
        self.name = name.strip()

    def __str__(self):
        return f"# Instr\t{self.ch}\t{self.pc}\t{self.name}"

    def __lt__(self, rhs):
        return self.ch < rhs.ch

    def set_pc(self, pc):
        self.pc = pc

    @staticmethod
    def from_cpp(evt):
        return InstrEvt(evt.channel, evt.program_change, evt.name)


class KeyEvt(Event):
    def __init__(self, stime, tonic, mode, fifth):
        super().__init__(Event.Type.KEY)

        self.stime = int(stime)
        self.tonic = tonic
        self.mode  = mode
        self.fifth = int(fifth)

    def __str__(self):
        return f"# Key\t{self.stime}\t{self.tonic}\t{self.mode}\t{self.fifth}"

    def __lt__(self, rhs):
        return self.stime < rhs.stime

    def is_convertible_to_tpqn_of(self, old_tpqn, new_tpqn):
        return self.stime * new_tpqn % old_tpqn == 0

    def convert_tpqn(self, old_tpqn, new_tpqn):
        self.stime = self.stime * new_tpqn // old_tpqn

    def to_number(self):
        return sitchclass2pitchclass(self.tonic) + 12 * int(self.mode == "minor")

    def to_pretty_midi(self, stime_to_sec):
        return pm.KeySignature(self.to_number(), stime_to_sec[self.stime])

    @staticmethod
    def from_cpp(evt):
        return KeyEvt(evt.stime, evt.tonic, evt.mode, evt.keyfifth)


class MeterEvt(Event):
    def __init__(self, stime, numer, denom, barlen):
        super().__init__(Event.Type.METER)

        self.stime  = int(stime)
        self.numer  = int(numer)
        self.denom  = int(denom)
        self.barlen = int(barlen)

    def __str__(self):
        return f"# Meter\t{self.stime}\t{self.numer}\t{self.denom}\t{self.barlen}"

    def __lt__(self, rhs):
        return self.stime < rhs.stime

    def is_same(self, rhs, include_stime=False):
        res  =  self.numer  == rhs.numer
        res &=  self.denom  == rhs.denom
        res &=  self.barlen == rhs.barlen
        res &= (self.stime  == rhs.stime) if include_stime else True
        return res

    def is_convertible_to_tpqn_of(self, old_tpqn, new_tpqn):
        return (self.stime  * new_tpqn % old_tpqn == 0) and  \
               (self.barlen * new_tpqn % old_tpqn == 0)

    def convert_tpqn(self, old_tpqn, new_tpqn):
        self.stime  = self.stime  * new_tpqn // old_tpqn
        self.barlen = self.barlen * new_tpqn // old_tpqn

    def get_complete_barlen(self, tpqn):
        assert tpqn * self.numer * 4 % self.denom == 0,     \
               f"Cannot calculate complete barlen for TPQN = {tpqn}"
        return tpqn * self.numer * 4 // self.denom

    def is_auftakt(self, tpqn):
        return (self.stime == 0) and (self.get_complete_barlen(tpqn) != self.barlen)

    def to_pretty_midi(self, tpqn, stime_to_sec):
        if self.numer == 0 or self.denom == 0:
            return pm.TimeSignature(4, 4, stime_to_sec[self.stime])

        if self.is_auftakt(tpqn):
            gcd    = math.gcd(tpqn, self.barlen)
            tpqn   = tpqn // gcd
            barlen = self.barlen // gcd
            assert (tpqn & (tpqn - 1)) == 0, f"tpqn(= {tpqn}) should be 2^n"
            return pm.TimeSignature(barlen, 4 * tpqn, 0)

        return pm.TimeSignature(self.numer, self.denom, stime_to_sec[self.stime])

    @staticmethod
    def from_cpp(evt):
        return MeterEvt(evt.stime, evt.num, evt.den, evt.barlen)


class SPQNEvt(Event):
    def __init__(self, stime, value, bpm):
        super().__init__(Event.Type.SPQN)

        self.stime = int(stime)
        self.value = float(value)
        self.bpm  = float(bpm)

    def __str__(self):
        return f"# SPQN\t{self.stime}\t{self.value}\t{self.bpm}"

    def __lt__(self, rhs):
        return self.stime < rhs.stime if self.stime != rhs.stime else self.value < rhs.value

    def is_convertible_to_tpqn_of(self, old_tpqn, new_tpqn):
        return self.stime * new_tpqn % old_tpqn == 0

    def convert_tpqn(self, old_tpqn, new_tpqn):
        self.stime = self.stime * new_tpqn // old_tpqn

    def stretch_time(self, rate):
        if rate == 1.0: return
        self.value /= rate

    @staticmethod
    def from_cpp(evt):
        return SPQNEvt(evt.stime, evt.value)


class ClefEvt(Event):
    def __init__(self, stime, channel, clef):
        super().__init__(Event.Type.CLEF)

        self.stime   = int(stime)
        self.channel = int(channel)
        self.clef    = clef

    def __str__(self):
        return f"# Clef\t{self.stime}\t{self.channel}\t{self.clef}"

    def __lt__(self, rhs):
        return self.stime < rhs.stime if self.stime != rhs.stime else self.channel < rhs.channel

    def is_convertible_to_tpqn_of(self, old_tpqn, new_tpqn):
        return self.stime * new_tpqn % old_tpqn == 0

    def convert_tpqn(self, old_tpqn, new_tpqn):
        self.stime = self.stime * new_tpqn // old_tpqn

    @staticmethod
    def from_cpp(evt):
        return ClefEvt(evt.stime, evt.channel, evt.clef)


class BarEvt(Event):
    def __init__(self):
        self.meter   = None
        self.key = None
        self.notes    = []


class Mode(enum.Enum):
    PITCH = enum.auto()
    SITCH = enum.auto()


class NoteEvt(Event):
    def __init__(self, ID, onstime, offstime, pitch_or_sitch, channel, subvoice, label,
                 onsec=None, offsec=None, onmetpos=None, onfractime=None):
        super().__init__(Event.Type.NOTE)

        self.ID = ID
        self.onstime = int(onstime)
        self.offstime = int(offstime)

        if (not isinstance(pitch_or_sitch, str)) or pitch_or_sitch.isdecimal():
            self.pitch = NoteEvt.__fix_rest_pitch(int(pitch_or_sitch))
            self.sitch = pitch2sitch(self.pitch)
        else:
            self.sitch = pitch_or_sitch
            self.pitch = sitch2pitch(self.sitch, rest=-1)

        self.channel  = int(channel)
        self.subvoice = int(subvoice)
        self.label    = label

        self.onsec  = float(onsec)  if onsec  is not None else None
        self.offsec = float(offsec) if offsec is not None else None

        self.onmetpos  = int(onmetpos)  if onmetpos  is not None else None
        self.onfractime = int(onfractime) if onfractime is not None else None

    @staticmethod
    def __fix_rest_pitch(pitch):
        if 0 <= pitch < 128: return pitch
        return -1

    def to_str(self, mode=None, qprx=None):
        if mode is None: mode = Mode.SITCH
        if qprx is None: qprx = self.has_onoff_sec()

        assert (not qprx) or self.has_onoff_sec(),   \
               "if qprx is True: onsec and offsec should not be None"

        return f"{self.ID}\t{self.onstime}\t{self.offstime}\t"           \
             + f"{self.pitch if mode == Mode.PITCH else self.sitch}\t"      \
             + f"{self.channel}\t{self.subvoice}\t{self.label}"             \
             + ("" if not qprx else f"\t{self.onsec:.6f}\t{self.offsec:.6f}")

    def __str__(self):
        return self.to_str()

    def __len__(self):
        return self.offstime - self.onstime

    @property
    def note_value(self):
        return len(self)

    def __lt__(self, rhs):
        if self.onstime != rhs.onstime:
            return self.onstime < rhs.onstime
        if self.offstime != rhs.offstime:
            return self.offstime < rhs.offstime
        if self.channel != rhs.channel:
            return self.channel < rhs.channel
        if self.subvoice != rhs.subvoice:
            return self.subvoice < rhs.subvoice
        if self.pitch != rhs.pitch:
            return self.pitch < rhs.pitch
        return False

    def is_rest(self):
        return not self.has_pitch()

    def has_pitch(self):
        return 0 <= self.pitch < 128

    def has_onoff_sec(self):
        return (self.onsec is not None) and (self.offsec is not None)

    def is_convertible_to_tpqn_of(self, old_tpqn, new_tpqn):
        return (self.onstime  * new_tpqn % old_tpqn == 0) and  \
               (self.offstime * new_tpqn % old_tpqn == 0)

    def convert_tpqn(self, old_tpqn, new_tpqn):
        self.onstime  = self.onstime  * new_tpqn // old_tpqn
        self.offstime = self.offstime * new_tpqn // old_tpqn

    def shift_pitch(self, semitone):
        if semitone == 0:  return
        if self.is_rest(): return

        self.pitch += semitone
        self.sitch = pitch2sitch(self.pitch)
        assert (0 <= self.pitch < 128), "Invalid pitch shift."

    def stretch_time(self, rate):
        if rate == 1.0: return
        if not self.has_onoff_sec():
            print("Warning: Note.onsec or Note.offsec is None")
            return

        self.set_onoff_secs(self.onsec / rate, self.offsec / rate)

    def get_onsec_map(self, stime_to_sec=None, performance=True):
        if performance and self.has_onoff_sec:
            return np.linspace(self.onsec, self.offsec, self.note_value, endpoint=False)
        elif stime_to_sec is not None:
            return stime_to_sec[self.onstime:self.offstime]
        else:
            raise ValueError("Cannot return onsecs")

    def get_onoff_secs(self, stime_to_sec=None, performance=True):
        if performance and self.has_onoff_sec():
            return self.onsec, self.offsec
        elif stime_to_sec is not None:
            return stime_to_sec[self.onstime], stime_to_sec[self.offstime]
        else:
            raise ValueError("Cannot return onoff_secs")

    def get_onoff_frames(self, fs, stime_to_sec=None, performance=False):
        onsec, offsec = self.get_onoff_secs(stime_to_sec, performance)
        return int(onsec * fs), int(offsec * fs)

    def get_frequency(self):
        if self.has_pitch():
            return librosa.midi_to_hz(self.pitch)
        else:
            return 0

    def set_ID(self, ID):
        self.ID = ID

    def set_onoff_secs(self, onsec, offsec):
        self.onsec = onsec
        self.offsec = offsec


    def to_pretty_midi(self, stime_to_sec=None, performance=False):
        onsec, offsec = self.get_onoff_secs(stime_to_sec, performance)
        return pm.Note(velocity=100, pitch=self.pitch, start=onsec, end=offsec)

    @staticmethod
    def from_cpp(evt):
        return NoteEvt(evt.ID, evt.onstime, evt.offstime, evt.sitch,
                       evt.channel, evt.subvoice, evt.label)


class ChordEvt(Event):
    def __init__(self, onstime, channel, chord,  onmetpos=None, onfractime=None):
        super().__init__(Event.Type.CHORD)

        self.onstime = int(onstime)
        self.channel  = int(channel)
        self.chord    = chord

        self.onmetpos  = int(onmetpos)  if onmetpos  is not None else None
        self.onfractime = int(onfractime) if onfractime is not None else None

    def __str__(self):
        return f"# Chord\t{self.onstime}\t{self.channel}\t{self.chord}"

    def convert_tpqn(self, old_tpqn, new_tpqn):
        self.onstime  = self.onstime  * new_tpqn // old_tpqn


class Qpr:
    def __init__(self, fname=None):
        self.name     = None
        self.memo     = None
        self.n_bar    = None
        self.TPQN     = 4
        self.comments = []
        self.instrs   = []
        self.keys     = []
        self.meters   = []  # include the dummy meter representing the end of music
        self.SPQNs    = []
        self.clefs    = []
        self.notes    = []
        self.chords   = []
        self.bars     = []

        if fname is None: return

        # _, ext = os.path.splitext(fname)
        # if ext in [".xml", ".musicxml"]:
        #     self.read_musicxml(fname)
        # else:
        self.read_file(fname)

        assert len(self.meters) >= 2, f"len(self.meters) = {len(self.meters)}"
        assert len(self.SPQNs)  >= 1, f"len(self.SPQNs) = {len(self.SPQNs)}"

    # Read & Write -------------------------------------------------------------

    def __parse_command(self, cmd, *args):
        if cmd == "TPQN":
            self.TPQN = int(args[0])
        elif cmd == "Instr":
            self.instrs.append(InstrEvt(*args))
        elif cmd == "Key":
            self.keys.append(KeyEvt(*args))
        elif cmd == "Meter":
            self.meters.append(MeterEvt(*args))
        elif cmd == "SPQN":
            self.SPQNs.append(SPQNEvt(*args))
        elif cmd == "Clef":
            self.clefs.append(ClefEvt(*args))
        elif cmd == "Name":
            self.name = args[0]
        elif cmd == "nBar":
            self.n_bar = int(args[0])
        elif cmd == "Memo":
            self.memo = args[0]
        elif cmd == "Boundary":
            pass
        elif cmd == "Chord":
            self.chords.append(ChordEvt(*args))
        else:
            raise ValueError(f"Invalid command: {cmd}")

    def __parse_note(self, *args):
        self.notes.append(NoteEvt(*args))

    def read_file(self, fname, qprx=False):
        if not os.path.exists(fname):
            raise FileNotFoundError(fname)

        lines = filter(None, [l.strip() for l in open(fname, "r", errors="ignore").read().split("\n")])

        for line in lines:
            if line[0] == "/":
                line = line[2:].strip()
                self.comments.append(line)
            elif line[0] == "#":
                line = line[1:].strip()
                self.__parse_command(*line.split("\t"))
            else:
                self.__parse_note(*line.split("\t"))

    # def read_musicxml(self, fname, mode=Mode.SITCH):
    #     if not os.path.exists(fname):
    #         raise FileNotFoundError(fname)
    #
    #     nqpr = NQpr()
    #     nqpr.read_musicxml_file(fname)
    #     self.TPQN = nqpr.TPQN
    #     self.comments = nqpr.comments
    #     self.instrs   = [InstrEvt.from_cpp(e) for e in nqpr.instrs]
    #     self.keys     = [KeyEvt.from_cpp(e)   for e in nqpr.keys]
    #     self.meters   = [MeterEvt.from_cpp(e) for e in nqpr.meters]
    #     self.SPQNs    = [SPQNEvt.from_cpp(e)  for e in nqpr.spqns]
    #     self.clefs    = [ClefEvt.from_cpp(e)  for e in nqpr.clefs]
    #     self.notes    = [NoteEvt.from_cpp(e)  for e in nqpr.noterests]

    def to_str(self, mode=None, qprx=None):
        s  = [f"// {c}" for c in self.comments]
        s += [f"# Name\t{self.name}"] if self.name is not None else []
        s += [f"# Memo\t{self.memo}"] if self.memo is not None else []
        s += [f'# nBar\t{self.n_bar}'] if self.n_bar is not None else []
        s += [f"# TPQN\t{self.TPQN}"]
        s += [str(e) for e in self.instrs]
        s += [str(e) for e in self.keys]
        s += [str(e) for e in self.meters]
        s += [str(e) for e in self.SPQNs]
        s += [str(e) for e in self.clefs]
        s += [e.to_str(mode, qprx) for e in self.notes]
        s  = "\n".join(s)
        return s

    def __repr__(self):
        return self.to_str(None)

    def write_file(self, fname, mode=None, qprx=None):
        open(fname, "w").write(f"{self.to_str(mode, qprx)}\n")

    def write_midi(self, fname, with_score=False):
        self.to_pretty_midi().write(str(fname))
        # if with_score:
        #     run_bash(f'{MID_TO_SCORE} "{os.path.splitext(fname)[0]}"')

    def reset_ID(self):
        for i, n in enumerate(self.notes):
            n.set_ID(str(i))

    # --------------------------------------------------------------------------

    def take_drop_channels_impl(self, op, channels, inplace, reset_ID):
        if not isinstance(channels, Iterable):
            channels = [channels]

        res = self if inplace else copy.deepcopy(self)
        res.notes = [n for n in self.notes if op(n.channel, channels)]

        if reset_ID: res.reset_ID()

        return res

    def take_channels(self, channels, inplace=True, reset_ID=True):
        return self.take_drop_channels_impl(
            lambda c, cs: c in cs, channels, inplace, reset_ID)

    def drop_channels(self, channels, inplace=True, reset_ID=True):
        return self.take_drop_channels_impl(
            lambda c, cs: c not in cs, channels, inplace, reset_ID)

    def shift_pitch(self, semitone, inplace=True):
        res = self if inplace else copy.deepcopy(self)
        if semitone == 0: return res
        for n in res.notes: n.shift_pitch(semitone)
        return res

    def stretch_time(self, speed_rate, inplace=True):
        res = self if inplace else copy.deepcopy(self)
        if speed_rate == 1.0: return res
        for s in res.SPQNs: s.stretch_time(speed_rate)
        for n in res.notes: n.stretch_time(speed_rate)
        return res

    def convert_tpqn(self, target_tpqn, inplace=True):
        assert self.is_convertible_to_tpqn_of(target_tpqn), "Invalid TPQN"
        res = self if inplace else copy.deepcopy(self)
        for k in res.keys:   k.convert_tpqn(self.TPQN, target_tpqn)
        for m in res.meters: m.convert_tpqn(self.TPQN, target_tpqn)
        for s in res.SPQNs:  s.convert_tpqn(self.TPQN, target_tpqn)
        for c in res.clefs:  c.convert_tpqn(self.TPQN, target_tpqn)
        for n in res.notes:  n.convert_tpqn(self.TPQN, target_tpqn)
        for c in res.chords: c.convert_tpqn(self.TPQN, target_tpqn)
        res.TPQN = target_tpqn
        return res

    """
    Setters
    """
    def sort(self):
        self.instrs.sort()
        self.keys.sort()
        self.meters.sort()
        self.SPQNs.sort()
        self.clefs.sort()
        self.notes.sort()
        self.reset_ID()

    def set_instr(self, ch, pc, inplace=True):
        assert 0 <= pc < 128, "Invalid program change: `{pc}`"

        res = self if inplace else copy.deepcopy(self)

        exist = False
        for evt in res.instrs:
            if evt.ch == ch:
                exist = True
                evt.set_pc(pc)

        if not exist:
            res.instrs.append(InstrEvt(ch, pc, "Default"))

        res.instrs.sort()

        return res

    def set_bpm(self, stime, bpm, inplace=True):
        return self.set_bpms([(stime, bpm)], inplace)

    def set_bpms(self, bpms, inplace=True):
        """
        Args:
            bpms (list of (int, int)): the list of stime & bpm pairs
        """
        res = self if inplace else copy.deepcopy(self)
        res.SPQNs = []
        res.add_bpms(bpms, inplace=True)
        return res

    def add_bpm(self, stime, bpm, inplace=True):
        return self.add_bpms([(stime, bpm)], inplace)

    def add_bpms(self, bpms, inplace=True):
        """
        Args:
            bpms (list of (int, int)): the list of stime & bpm pairs
        """
        if not isinstance(bpms, Iterable): bpms = [bpms]
        res = self if inplace else copy.deepcopy(self)

        last_stime = res.get_last_stime()
        spqns = {s.stime: s.value for s in res.SPQNs}
        spqns.update({stime: 60.0/bpm for stime, bpm in bpms if stime < last_stime})

        res.SPQNs = [SPQNEvt(stime, value) for stime, value in spqns.items()]
        res.SPQNs.sort()

        return res

    def set_onoff_secs(self, onoff_secs, inplace=True):
        assert len(onoff_secs) == len(self.notes), "len(onoff_secs) != len(self.notes)"

        res = self if inplace else copy.deepcopy(self)
        for (on, off), n in zip(onoff_secs, res.notes):
            n.set_onoff_secs(on, off)

        return res

    def __make_zeros(self, dtype, last_stime=False):
        return np.zeros(self.get_last_stime() + int(last_stime), dtype=dtype)

    def __make_full(self, value, last_stime=False):
        return np.full(self.get_last_stime() + int(last_stime), value)

    def set_met_pos(self, resol, inplace=True):
        #set metrical position and fractional time
        #resolution is used for setting onFractime
        res = self if inplace else copy.deepcopy(self)
        last_stime = res.get_last_stime()
        bars = []
        curKeyEvtPos = 0
        for i in range(len(res.meters)-1):
            cur_bar = []
            for tau in range(res.meters[i].stime, res.meters[i+1].stime, res.meters[i].barlen):
                cur_bar.append(copy.deepcopy(res.meters[i]))
                cur_bar.append(tau)
                if curKeyEvtPos < len(res.keys)-1:
                    if cur_bar[0].stime >= res.keys[curKeyEvtPos+1].stime:
                        curKeyEvtPos += 1
                cur_bar.append(copy.deepcopy(res.keys[curKeyEvtPos]))
                cur_bar.append([])
                bars.append(cur_bar)
        for n in range(len(res.notes)-1,-1,-1):
            barpos = 0
            for m in range(len(bars)):
                if res.notes[n].onstime < bars[m][0].stime + bars[m][0].barlen:
                    barpos = m
                    break
            res.notes[n].barpos = barpos
            res.notes[n].onmetpos = res.notes[n].onstime - bars[barpos][0].stime
            res.notes[n].onfractime = (((res.notes[n].onstime - bars[barpos][0].stime) * resol) // bars[barpos][0].barlen) % resol
            if ((res.notes[n].onstime - bars[barpos][0].stime) * resol) % bars[barpos][0].barlen != 0:
                del res.notes[n]
        for n in range(len(res.chords)-1,-1,-1):
            barpos = 0
            for m in range(len(bars)):
                if res.chords[n].onstime < bars[m][0].stime + bars[m][0].barlen:
                    barpos = m
                    break
            res.chords[n].barpos = barpos
            res.chords[n].onmetpos = res.chords[n].onstime - bars[barpos][0].stime
            res.chords[n].onfractime = (((res.chords[n].onstime - bars[barpos][0].stime) * resol) // bars[barpos][0].barlen) % resol
            if ((res.chords[n].onstime - bars[barpos][0].stime) * resol) % bars[barpos][0].barlen != 0:
                del res.chords[n]
        return res


    """
    Getters
    """
    def get_num_channels(self):
        # return np.max([i.ch for i in self.instrs]) + 1
        return np.max([n.channel for n in self.notes]) + 1

    def get_channel_notes(self):
        self.sort()

        res = dict()
        for n in self.notes:
            res.setdefault(n.channel, [])
            res[n.channel].append(n)
        return res

    def get_beats(self):
        beats = [range(mi.stime-(self.TPQN * mi.numer * 4 // mi.denom - mi.barlen),
                       mj.stime, self.TPQN * 4 // mi.denom)
                 for mi, mj in sliding_window(2, self.meters)]
        return np.array([b for b in functools.reduce(operator.iconcat, beats, []) if b >= 0])

    def get_stime_to_fifth(self, last_stime=False):
        stime_to_fifth = self.__make_zeros(int, last_stime)
        for k in self.keys:
            stime_to_fifth[k.stime:] = k.fifth
        return stime_to_fifth

    def get_stime_to_beats(self, last_stime=False):
        stime_to_beats = self.__make_zeros(int, last_stime)
        stime_to_beats[self.get_beats()] = 1
        if last_stime: stime_to_beats[-1] = 1
        return stime_to_beats

    def get_downbeats(self):
        downbeats = [range(mi.stime, mj.stime, mi.barlen)
                     for mi, mj in sliding_window(2, self.meters)
                     if not mi.is_auftakt(self.TPQN)]
        return np.array(functools.reduce(operator.iconcat, downbeats, []))

    def get_stime_to_downbeats(self, last_stime=False):
        stime_to_downbeats = self.__make_zeros(int, last_stime)
        stime_to_downbeats[self.get_downbeats()] = 1
        if last_stime: stime_to_downbeats[-1] = 1
        return stime_to_downbeats

    def remove_rests_and_set_tied_notes(self):
        for n in reversed(range(len(self.notes))):
            if self.notes[n].label == 'ー':
                self.notes[n-1].offstime = self.notes[n].offstime
                del self.notes[n]
                continue
            if self.notes[n].sitch == 'R':
                del self.notes[n]

    def get_notes_splitted_per_bar(self, copied=False):
        self.remove_rests_and_set_tied_notes()
        barnums = self.get_stime_to_barnum()
        if len(barnums) == 0: return []
        notes = [[] for _ in range(barnums[-1]+1)]
        for n in self.notes:
            notes[barnums[n.onstime]].append(copy.deepcopy(n) if copied else n)
        return notes

    def get_chords_splitted_per_bar(self, copied=False):
        barnums = self.get_stime_to_barnum()
        if len(barnums) == 0: return []
        chords = [[] for _ in range(barnums[-1]+1)]
        for c in self.chords:
            chords[barnums[c.onstime]].append(copy.deepcopy(c) if copied else c)
        return chords

    def split_bars(self, resol):
        self.bars = []
        last_stime = self.meters[-1].stime
        bar = BarEvt()
        cur_key_evt_pos = 0
        # make bars
        for i in range(len(self.meters)-1):
            for tau in range(self.meters[i].stime, self.meters[i+1].stime, self.meters[i].barlen):
                bar.meter = copy.copy(self.meters[i])
                bar.meter.stime = tau
                if cur_key_evt_pos < len(self.keys)-1:
                    if bar.meter.stime >= self.keys[cur_key_evt_pos+1].stime:
                        cur_key_evt_pos += 1
                bar.key = copy.copy(self.keys[cur_key_evt_pos])
                bar.notes = []
                self.bars.append(copy.copy(bar))

        # put notes
        for n in range(len(self.notes)):
            barpos = 0
            for m in range(len(self.bars)):
                if self.notes[n].onstime < self.bars[m].meter.stime + self.bars[m].meter.barlen:
                    barpos = m
                    break
#           self.notes[n].barpos = barpos
#           self.notes[n].barNotePos = len(bars[barpos].notes)
            self.notes[n].onmetpos = self.notes[n].onstime - self.bars[barpos].meter.stime
            self.notes[n].onfractime = (((self.notes[n].onstime-self.bars[barpos].meter.stime)*resol)//self.bars[barpos].meter.barlen)%resol
            self.bars[barpos].notes.append(copy.copy(self.notes[n]))

    def __calc_residual_stimes(self, mi, mj):
        # | (4/4) quarter 8th | -> [6, 5, 4, 3, 2, 1] under TPQN = 4
        if mi.is_auftakt(self.TPQN):
            return range(mi.barlen, 0, -1)

        # | (4/4) whole | (4/4) half | -> [0, 3, 2, 1, 0, 3] under TPQN = 1
        stime_diff = mj.stime - mi.stime
        num_bars = (stime_diff + mi.barlen - 1) // mi.barlen    # may contain the last incomplete bar
        on  = num_bars * mi.barlen
        off = on - stime_diff
        res = (np.arange(on, off, -1) % mi.barlen).tolist()
        return res

    def get_init_residual(self):
        return self.__calc_residual_stimes(self.meters[0], self.meters[1])[0]

    def get_stime_to_residual(self, last_stime=False):
        # self.meters contain the dummy meter representing the end of music
        residuals = [self.__calc_residual_stimes(mi, mj)
                     for mi, mj in sliding_window(2, self.meters)]
        residuals = functools.reduce(operator.iconcat, residuals, [])

        if last_stime:
            residuals.append(-1)

        return np.array(residuals)

    def get_stime_to_onpos(self, last_stime=False):
        onposs = [(np.arange(0, mj.stime - mi.stime) % mi.barlen).tolist()
                  for mi, mj in sliding_window(2, self.meters)]
        onposs = functools.reduce(operator.iconcat, onposs, [])

        if last_stime:
            onposs.append(-1)

        return np.array(onposs)

    def get_barstarts(self, last_stime=False):
        downbeats = [range(mi.stime, mj.stime, mi.barlen)
                     for mi, mj in sliding_window(2, self.meters)]
        if last_stime: downbeats.append([self.get_last_stime()])
        return np.array(functools.reduce(operator.iconcat, downbeats, []))

    def get_stime_to_barnum(self, last_stime=False):
        barnum = self.__make_zeros(dtype=int, last_stime=last_stime)
        barnum[self.get_barstarts(last_stime)] = 1
        barnum = np.cumsum(barnum) - 1
        assert np.all(barnum >= 0), "all elements in barnum should be greater than or equal to zero."
        return barnum

    @staticmethod
    def numden_to_str(numer, denom):
        return f"{numer}/{denom}"

    @staticmethod
    def str_to_numden(meter_str):
        numer, denom = meter_str.split("/")
        return int(numer), int(denom)

    def get_stime_to_meter(self, dtype="tuple", last_stime=False):
        assert dtype in ["tuple", "str"], f"`dtype` must be 'tuple' or 'str'."

        meters = [[(mi.numer, mi.denom)] * (mj.stime - mi.stime)
                  for mi, mj in sliding_window(2, self.meters)]
        meters = functools.reduce(operator.iconcat, meters, [])

        if last_stime:
            meters.append((0, 0))

        if dtype == "str":
            meters = [Qpr.numden_to_str(num, den) for num, den in meters]

        return np.array(meters)

    def get_stime_to_spqn(self):
        spqns = self.SPQNs + [SPQNEvt(self.get_last_stime(), 0)]
        spqns = [[si.value] * (sj.stime - si.stime)
                 for si, sj in sliding_window(2, spqns)]

        return np.array(functools.reduce(operator.iconcat, spqns, []))

    def __stime2spqn_to_stime2sec(self, spqn_map, last_stime=True):
        """ convert SPQN-map to SEC-map
        Args:
            spqn_map   (list(int)): gotten from `self.get_stime_to_spqn()`
            last_stime (bool): if true, SEC-map includes the onset time of the last stime.
        """
        sec_per_tick = np.asarray(spqn_map) / self.TPQN
        sec_map = np.cumsum([0] + sec_per_tick.tolist())

        if not last_stime:
            sec_map = sec_map[:-1]

        return sec_map

    def get_stime_to_sec(self, last_stime=True):
        """
        Args:
            last_stime (bool):
                if true, `SEC-map` includes the onset time of the last stime.
        """
        return self.__stime2spqn_to_stime2sec(self.get_stime_to_spqn(), last_stime)

    def get_stime_to_performance_sec(self, last_stime=True):
        """
        Args:
            last_stime (bool):
                if true, `stime_to_sec` includes the onset time of the last stime.
        """
        assert self.is_monophonic(), "Qpr should be monophonic"

        stime_to_sec = self.get_stime_to_sec(last_stime)
        args = dict(stime_to_sec=stime_to_sec, performance=True)

        onsecs = np.zeros(self.get_last_stime() + int(last_stime))

        for n in self.notes:
            onsecs[n.onstime:n.offstime] = n.get_onsec_map(**args)

        dummy_note = NoteEvt(ID='-1', onstime=0, offstime=0, pitch_or_sitch=-1,
                             channel=-1, subvoice=-1, label=-1, onsec=0, offsec=0)
        for pn, nn in sliding_window(2, [dummy_note, *self.notes]):
            onsecs[pn.offstime:nn.onstime] =    \
                np.linspace(pn.offsec, nn.onsec, nn.onstime - pn.offstime, endpoint=False)

        def get_last_onsec():
            if len(self.notes) > 0:
                return self.notes[-1].get_onoff_secs(**args)[1]
            else:
                return stime_to_sec[-1]

        if last_stime:
            onsecs[-1] = get_last_onsec()

        assert np.all(np.array(onsecs[1:]) >  0)
        assert np.all(np.diff(onsecs)      >= 0)

        return np.asarray(onsecs)

    def get_stime_to_pitch(self, rest=-1, last_stime=True):
        """
        Args:
            last_stime (bool):
                if true, `stime_to_sec` includes the onset time of the last stime.
        """
        assert self.is_monophonic(), "Qpr should be monophonic"

        stime_to_pitch = self.__make_full(rest, last_stime)

        for n in self.notes:
            stime_to_pitch[n.onstime:n.offstime] = n.pitch if not n.is_rest() else rest

        return stime_to_pitch

    def get_stime_to_isonset(self, last_stime=True):
        """
        Args:
            last_stime (bool):
                if true, `stime_to_sec` includes the onset time of the last stime.
        """
        assert self.is_monophonic(), "Qpr should be monophonic"

        stime_to_isonset = self.__make_zeros(int, last_stime)

        for n in self.notes:
            stime_to_isonset[n.onstime] = 1

        if last_stime and (len(stime_to_isonset) > 0):
            stime_to_isonset[-1] = 1

        return stime_to_isonset


    def get_onoff_secs(self, performance=False):
        """
        Args:
            performance (bool):
                use the performed onset & offset times
        """
        stime_to_sec = self.get_stime_to_sec(last_stime=True)
        return [n.get_onoff_secs(stime_to_sec, performance) for n in self.notes]

        # return [(n.onsec, n.offsec) if performance and n.has_onoff_sec() else
        #         (stime_to_sec[n.onstime], stime_to_sec[n.offstime]) for n in self.notes]

    def get_onoff_frames(self, fs=100, performance=False):
        """
        Args:
            performance (bool):
                use the performed onset & offset times
        """
        stime_to_sec = self.get_stime_to_sec(last_stime=True)
        return [n.get_onoff_frames(fs, stime_to_sec, performance) for n in self.notes]

        # onoff_secs = self.get_onoff_secs(performance)
        # return [(int(on * fs), int(off * fs)) for on, off in onoff_secs]

    def get_onsetroll(self, fs=100, performance=False, offset=0, condense=False):
        """
        Args:
            fs  (int) : the number of frames in one second.
            performance (bool):
                use the performed onset & offset times
        """
        onoff_frames = self.get_onoff_frames(fs, performance)

        nchannels = self.get_num_channels()
        lastframe = np.max([off for _, off in onoff_frames])
        pianoroll = np.zeros((nchannels, 128, lastframe), dtype=np.int32)
        for (on, _), n in zip(onoff_frames, self.notes):
            if n.is_rest(): continue
            pianoroll[n.channel, n.pitch, on] = 1

        if condense:
            # pianoroll = (np.sum(pianoroll, axis=0) > 0).astype(np.int32)
            pianoroll = np.sum(pianoroll, axis=0)

        return pianoroll

    def get_pianoroll(self, fs=100, performance=False, offset=0, condense=False):
        """
        Args:
            fs  (int) : the number of frames in one second.
        """
        onoff_frames = self.get_onoff_frames(fs, performance)

        nchannels = self.get_num_channels()
        lastframe = np.max([off for _, off in onoff_frames])
        pianoroll = np.zeros((nchannels, 128, lastframe), dtype=np.int32)
        for (on, off), n in zip(onoff_frames, self.notes):
            if n.is_rest(): continue
            pianoroll[n.channel, n.pitch, on:max(on, off-offset)] = 1

        if condense:
            # pianoroll = (np.sum(pianoroll, axis=0) > 0).astype(np.int32)
            pianoroll = np.sum(pianoroll, axis=0)

        return pianoroll

    def get_chroma(self, fs=100, performance=False, condense=False):
        """
        Args:
            fs  (int) : the number of frames in one second.
        """
        pianoroll = self.get_pianoroll(fs, performance, condense=False)

        chroma = np.stack([np.sum(pianoroll[:, i::12, :], axis=1) for i in range(12)], axis=1)

        if condense:
            chroma = np.sum(chroma, axis=0)

        return chroma

    def get_last_stime(self):
        assert len(self.meters) > 0, "In get_last_stime()"
        return self.meters[-1].stime


    """
    Validators
    """
    def is_convertible_to_tpqn_of(self, target_tpqn):
        res  = True
        res &= all([k.is_convertible_to_tpqn_of(self.TPQN, target_tpqn) for k in self.keys])
        res &= all([m.is_convertible_to_tpqn_of(self.TPQN, target_tpqn) for m in self.meters])
        res &= all([s.is_convertible_to_tpqn_of(self.TPQN, target_tpqn) for s in self.SPQNs])
        res &= all([c.is_convertible_to_tpqn_of(self.TPQN, target_tpqn) for c in self.clefs])
        res &= all([n.is_convertible_to_tpqn_of(self.TPQN, target_tpqn) for n in self.notes])
        return res

    def is_monophonic(self):
        self.sort()
        return all([ni.offstime <= nj.onstime for ni, nj in sliding_window(2, self.notes)])

    def is_full(self):
        pass

    def check_meters(self, *meters):
        return all([(m.numer, m.denom) in meters for m in self.meters[:-1]])

    """
    Synthesizer
    """
    def __write_wav(self, f_wav, wav, fs):
        if f_wav is not None:
            sf.write(f_wav, wav, fs)

    def chiptune(self, f_wav=None, fs=44100, performance=False):
        onoff_frames = self.get_onoff_frames(fs, performance)
        lengs = [off - on for on, off in onoff_frames]
        freqs = [n.get_frequency() for n in self.notes]
        freqs = np.repeat(freqs, lengs)
        wav   = freq2wav(freqs, fs, 1/fs*1000, "sine", 1)

        self.__write_wav(f_wav, wav, fs)
        return wav

    def synthesize(self, f_wav=None, fs=44100, performance=False):
        wav = self.to_pretty_midi(performance).synthesize(fs)
        self.__write_wav(f_wav, wav, fs)
        return wav

    def fluidsynth(self, f_wav=None, fs=44100, performance=False):
        # self.__pretty_midi_fluidsynth(f_wav, fs, performance)
        return self.__bash_fluidsynth(f_wav, fs, performance)

    def __pretty_midi_fluidsynth(self, f_wav=None, fs=44100, performance=False):
        wav = self.to_pretty_midi(performance).fluidsynth(fs)
        self.__write_wav(f_wav, wav, fs)
        return wav

    def __bash_fluidsynth(self, f_wav=None, fs=44100, performance=False):
        not_save_wav = f_wav is None
        tmp_id       = uuid.uuid4()
        f_wav        = f"{tmp_id}.wav" if not_save_wav else f_wav
        f_wav_tmp    = f"{tmp_id}.wav"
        f_mid_tmp    = f"{tmp_id}.mid"
        soundfont    = "/usr/share/sounds/sf2/FluidR3_GM.sf2"

        self.to_pretty_midi(performance).write(f_mid_tmp)

        run_bash(f"fluidsynth -F {f_wav_tmp} -L 0 -r {fs} {soundfont} {f_mid_tmp} && "
                 f"sox {f_wav_tmp} {f_wav} channels 1 && "
                 f"command rm {f_wav_tmp} && "
                 f"command rm {f_mid_tmp}")

        wav = librosa.load(f_wav, sr=None, mono=True)

        if not_save_wav:
            run_bash(f"command rm {f_wav}")

        return wav

    def to_pretty_midi(self, performance=False):
        stime_to_spqns = self.get_stime_to_spqn()
        stime_to_sec   = self.__stime2spqn_to_stime2sec(stime_to_spqns)

        resolution = 2520
        smf = pm.PrettyMIDI(resolution=resolution, initial_tempo=120)

        # meters
        smf.time_signature_changes = [
            m.to_pretty_midi(self.TPQN, stime_to_sec) for m in self.meters]

        # keys
        smf.key_signature_changes = [
            k.to_pretty_midi(stime_to_sec) for k in self.keys]

        # tempo changes
        for s in self.SPQNs:
            smf._tick_scales.append(
                (smf.time_to_tick(stime_to_sec[s.stime]), stime_to_spqns[s.stime] / smf.resolution))
            smf._update_tick_to_time(smf.time_to_tick(smf.get_end_time()))

        # tracks
        max_ch = 15
        num_ch = min(max_ch, max([i.ch for i in self.instrs]) + 1)
        smf.instruments = [pm.Instrument(program=0) for _ in range(num_ch)]
        for i in self.instrs:
            if i.ch >= max_ch: continue
            smf.instruments[i.ch].program = i.pc if i.pc >= 0 else 0

        # notes
        for n in self.notes:
            if n.is_rest(): continue
            if n.channel >= max_ch: continue
            smf.instruments[n.channel].notes.append(
                n.to_pretty_midi(stime_to_sec, performance))

        return smf

    """
    Plot
    """
    def plot(self, ax=None, ch=0, note_color="B"):
        if ax is None: ax = plt.gca()

        notes = [e for e in self.notes if e.channel == ch]
        pitches = [e.pitch for e in notes if not e.is_rest()]
        max_x = max([e.offstime for e in notes])
        max_p = max(pitches) + 1 if len(pitches) > 0 else 128
        min_p = min(pitches)     if len(pitches) > 0 else 128
        # if max_p is None: max_p = 128
        # if min_p is None: min_p = 128
        height = max_p - min_p + 1

        ax.set_xlim(        0, max_x)
        ax.set_ylim(min_p-0.5, max_p+0.5)

        def add_rect(*args, **kwargs):
            ax.add_patch(plt.Rectangle(*args, **kwargs))

        # Black keys
        for p in range(128):
            if p % 12 not in [1, 3, 6, 8, 10]: continue
            add_rect(xy=(0, p-0.5), width=max_x, height=1, fc="#e5efff", lw=0, zorder=0)

        # Grids
        ax.vlines(np.arange(max_x), 0, 1, transform=ax.get_xaxis_transform(),
                  color="#aaaaaa", lw=0.3, zorder=1)

        # Beats & Downbeats
        for b in self.get_beats():
            add_rect(xy=(b, min_p-0.5), width=1, height=height, fc="#b6ebb6", lw=0, zorder=3)
        for b in self.get_downbeats():
            add_rect(xy=(b, min_p-0.5), width=1, height=height, fc="#fdc9d1", lw=0, zorder=3)

        # Notes
        note_color = dict(
            R = "#ff5050",
            G = "#2ca02c",
            B = "#0070c0",
        )[note_color]

        def plot_note(note):
            p = note.pitch if not note.is_rest() else max_p
            c = note_color if not note.is_rest() else "#aaaaaa"

            w = note.offstime - note.onstime
            add_rect(xy=(note.onstime, p-0.5), width=w, height=1, zorder=4,
                     fc=c, ec="#000000", lw=0.8)

        for note in notes:
            plot_note(note)

    def imshow(self, ax=None, fs=100, performance=False, cmap="viridis"):
        if ax is None: ax = plt.gca()

        pianoroll = self.get_pianoroll(fs=fs, performance=performance, offset=1, condense=True)
        ax.imshow(pianoroll, aspect="auto", origin="lower", cmap=cmap)
