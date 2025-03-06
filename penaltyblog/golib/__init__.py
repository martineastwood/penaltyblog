import ctypes
import os
import platform

# Determine the correct shared library name based on OS
system = platform.system()
if system == "Windows":
    lib_name = "penaltyblog.dll"
elif system == "Darwin":
    lib_name = "penaltyblog.dylib"
else:
    lib_name = "penaltyblog.so"

# Construct the full path to the shared library
lib_path = os.path.join(os.path.dirname(__file__), lib_name)
go_lib = ctypes.CDLL(lib_path)
