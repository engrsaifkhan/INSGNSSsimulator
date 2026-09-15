#!/usr/bin/env python3
"""Headless runner for PocketSDR: open IF file and print TCA CSV and debug traces."""
import sys, os, time
from ctypes import c_char_p, c_int32, c_double, POINTER

import pocket_sdr
import sdr_rtk


class Var:
    def __init__(self, v): self._ = v
    def get(self): return self._
    def set(self, v): self._ = v


class OptObj:
    pass


def build_min_opts(if_path, fmt='INT8', fs_msps='24.000'):
    inp = OptObj()
    inp.lpf_bw = [Var('0.0') for _ in range(8)]
    inp.fmt = Var(fmt)
    inp.fs = Var(fs_msps)
    inp.fo = [Var('1568.000') for _ in range(8)]
    inp.IQ = [Var('IQ') for _ in range(8)]
    inp.bits = [Var('3') for _ in range(8)]
    inp.str_path = Var(if_path)
    inp.toff = Var('0.0')
    inp.tscale = Var('1.0')
    inp.fmt_ids = {'INT8': 1, 'INT8X2': 2, 'RAW8': 3, 'RAW16': 4,
                   'RAW16I': 5, 'RAW32': 6, 'CS8': 7, 'CS16': 8,
                   'INT16X2': 8}
    inp.dev_opt = Var('')

    out = OptObj()
    out.path_ena = [Var(0) for _ in range(4)]
    out.path = [Var('') for _ in range(4)]
    out.array_sep = Var(0)

    sig = OptObj()
    # Minimal signal selection: enable GPS L1CA and GLONASS G1CA
    sig.sys = ('GPS', 'GLONASS', 'Galileo', 'QZSS', 'BeiDou', 'NavIC', 'SBAS')
    sig.sig = (
        ('L1CA',),
        ('G1CA',),
        ('E1B',),
        ('L1CA',),
        ('B1I',),
        ('I5S',),
        ('L1CA',),
    )
    sig.sys_sel = [Var(1), Var(1), Var(0), Var(0), Var(0), Var(0), Var(0)]
    sig.satno = [Var('1-32'), Var('-7-6/1-27'), Var('1-36'), Var('1-9'),
                 Var('1-63'), Var('1-14'), Var('120-158')]
    sig.sig_sel = [[Var(1) for _ in s] for s in sig.sig]
    sig.sig_rfch = Var('')

    sysopt = OptObj()
    sysopt.rcv_options = Var('')
    sysopt.acq_mode = Var('')
    sysopt.bump_jump = Var('0')

    array_opt = OptObj()
    array_opt.no_array = Var('0')

    # If settings INI exists, try to load matching input options
    try:
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read(os.path.join(os.path.dirname(__file__), 'settings_ALL_L1.ini'))
        if 'inp_opt' in cfg:
            s = cfg['inp_opt']
            if 'fmt' in s:
                inp.fmt = Var(s.get('fmt'))
            if 'fs' in s:
                inp.fs = Var(s.get('fs'))
            for i in range(8):
                k = f'fo@{i}'
                if k in s:
                    try:
                        inp.fo[i] = Var(s.get(k))
                    except Exception:
                        pass
                kIQ = f'IQ@{i}'
                if kIQ in s:
                    inp.IQ[i] = Var(s.get(kIQ))
                kbits = f'bits@{i}'
                if kbits in s:
                    inp.bits[i] = Var(s.get(kbits))
                klpf = f'lpf_bw@{i}'
                if klpf in s:
                    inp.lpf_bw[i] = Var(s.get(klpf))
            if 'toff' in s:
                inp.toff = Var(s.get('toff'))
            if 'tscale' in s:
                inp.tscale = Var(s.get('tscale'))
    except Exception:
        pass

    return sysopt, inp, out, sig, array_opt


def main():
    if len(sys.argv) < 2:
        print('usage: pocket_sdr_headless.py <if_file> [timeout_s]')
        return
    if_path = os.path.abspath(sys.argv[1])
    timeout = float(sys.argv[2]) if len(sys.argv) >= 3 else 180.0
    if not os.path.isfile(if_path):
        print('IF file not found:', if_path)
        return

    sysopt, inp, out, sig, array_opt = build_min_opts(if_path)

    # Optional: force capture to GLONASS-only (set env POCKETSDR_FORC_GLO=1)
    if os.environ.get('POCKETSDR_FORC_GLO', '').strip() == '1':
        sig.sys_sel = [Var(0) for _ in sig.sys]
        sig.sys_sel[1] = Var(1)
        # enable only G1CA
        sig.sig = tuple((('G1CA',),) if i == 1 else (tuple(),) for i in range(len(sig.sys)))
        sig.sig_sel = [[Var(1) if j == 0 and i == 1 else Var(0) for j in range(len(sig.sig[i]) if i < len(sig.sig) else 1)] for i in range(len(sig.sys))]

    print('Opening IF file:', if_path)
    # some helper functions in pocket_sdr.py reference module-level names
    pocket_sdr.out_opt = out
    rcv, info = pocket_sdr.rcv_open_file(sysopt, inp, out, sig, array_opt)
    if not rcv:
        print('rcv_open_file failed:', info)
        return

    print('Receiver opened', info)
    # Enable RTKLIB trace file to capture TCA debug traces
    try:
        sdr_rtk.traceopen('/tmp/pocket_tca.trace')
        sdr_rtk.tracelevel(4)
    except Exception:
        pass
    libsdr = pocket_sdr.libsdr
    try:
        from ctypes import c_void_p
        libsdr.sdr_rcv_tca_stat.argtypes = (c_void_p,)
    except Exception:
        pass
    libsdr.sdr_rcv_tca_stat.restype = c_char_p

    # Poll for TCA CSV for up to `timeout` seconds and print PVT periodically
    start = time.time()
    seen = False
    last_pvt = 0.0
    while time.time() - start < timeout:
        try:
            s = libsdr.sdr_rcv_tca_stat(rcv)
        except Exception:
            s = None
        if s:
            txt = s.decode('utf-8', errors='ignore') if isinstance(s, (bytes, bytearray)) else str(s)
            print('--- TCA CSV BEGIN ---')
            print(txt)
            print('--- TCA CSV END ---')
            seen = True
            break
        # print PVT every 2 seconds to show processing progress
        if time.time() - last_pvt > 2.0:
            try:
                pvt = pocket_sdr.get_rcv_pvt_sol(rcv)
                print('PVT:', pvt)
            except Exception:
                pass
            last_pvt = time.time()
        # print GLONASS channel status every 5 seconds
        try:
            if time.time() - start > 0 and int(time.time() - start) % 5 == 0:
                chs = pocket_sdr.get_ch_stat(rcv, 'R')
                if chs:
                    print('GLONASS CH STAT:')
                    for line in chs[:10]:
                        print(line)
        except Exception:
            pass
        time.sleep(0.1)

    if not seen:
        print('No TCA CSV produced within timeout.')

    # keep running a short while to allow C trace output to flush
    time.sleep(0.5)

    # Close receiver
    try:
        libsdr.sdr_rcv_close(rcv)
    except Exception:
        pass


if __name__ == '__main__':
    main()
