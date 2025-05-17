from setuptools import setup
import os

APP = ['main.py']

DATA_FILES = [
    ('audio', [os.path.join('audio', f) for f in os.listdir('audio')]),
    ('img', [os.path.join('img', f) for f in os.listdir('img')]),
    ('level', [os.path.join('level', f) for f in os.listdir('level')]),
]

OPTIONS = {
    'argv_emulation': False,
    'includes': ['button'],  # add any custom .py files you wrote
    'resources': ['audio', 'img', 'level'],
    'plist': {
        'CFBundleName': 'ScrollingShooter',
        'CFBundleShortVersionString': '1.0.0',
    },
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
