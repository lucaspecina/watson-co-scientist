#!/usr/bin/env python3
"""
Test if the necessary packages are installed and can be imported.
"""

import sys
import site
from importlib import import_module, util

# Add user site-packages to path
user_site = site.USER_SITE
if user_site not in sys.path:
    sys.path.append(user_site)

def check_module(module_name):
    """Check if a module can be imported."""
    try:
        import_module(module_name)
        return True
    except ImportError:
        return False

# List of modules to check
modules = [
    "fitz",  # PyMuPDF
    "pytesseract",
    "PIL",
    "numpy",
    "cv2"  # OpenCV
]

# Check each module
print("Testing package imports:")
for module in modules:
    status = "INSTALLED" if check_module(module) else "NOT FOUND"
    print(f"  {module}: {status}")

# Print Python path
print("\nPython path:")
for path in sys.path:
    print(f"  {path}")

# Print system info
print("\nSystem information:")
print(f"  Python version: {sys.version}")
print(f"  User site-packages: {user_site}")
print(f"  Executable: {sys.executable}")

if __name__ == "__main__":
    pass