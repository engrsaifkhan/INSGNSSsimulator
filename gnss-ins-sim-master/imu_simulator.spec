# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

# Collect all gnss_ins_sim data files
datas = []
datas += collect_data_files('gnss_ins_sim')

# Add specific data directories
if os.path.exists('demo_motion_def_files'):
    datas.append(('demo_motion_def_files', 'demo_motion_def_files'))

a = Analysis(
    ['imu_simulator.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'numpy',
        'matplotlib',
        'scipy',
        'numpy.core._methods',
        'numpy.lib.format',
        'matplotlib.backends.backend_tkagg',
        'gnss_ins_sim',
        'gnss_ins_sim.sim',
        'gnss_ins_sim.sim.imu_model',
        'gnss_ins_sim.sim.ins_sim',
        'gnss_ins_sim.geoparams',
        'gnss_ins_sim.geoparams.geomag',
    ] + collect_submodules('gnss_ins_sim'),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter.test', 'matplotlib.tests'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='imu_simulator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add 'icon.ico' if you have one
)