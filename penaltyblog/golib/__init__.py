import ctypes
import os
import platform

# Determine the correct shared library name based on OS
if platform.system() == "Windows":
    lib_name = "penaltyblog.dll"
else:
    lib_name = "penaltyblog.so"

# Load the shared library
lib_path = os.path.join(os.path.dirname(__file__), lib_name)
go_lib = ctypes.CDLL(lib_path)
