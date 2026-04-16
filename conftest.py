"""Root conftest.py — ensures the repo root is on sys.path so the qualibact
package can be imported without requiring a pip install."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
