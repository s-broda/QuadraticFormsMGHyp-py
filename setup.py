"""Build the QuadraticFormsMGHyp C extension. Metadata lives in pyproject.toml.

Windows compiles with clang-cl and links with link.exe. clang-cl provides
C11 complex and targets the MSVC ABI, so a later wheel build uses this
same path. link.exe cannot read clang bitcode, so whole-program
optimization is left off, and compiler-rt supplies the complex
multiply and divide helpers.
"""

import os
import platform
import shutil
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext


def _win_arch():
    if sys.maxsize <= 2 ** 32:
        return "x86"
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "arm64"
    return "x64"


def _clang_cl():
    found = shutil.which("clang-cl")
    if found:
        return found
    for env in ("ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(env)
        if not root:
            continue
        candidate = os.path.join(root, "LLVM", "bin", "clang-cl.exe")
        if os.path.isfile(candidate):
            return candidate
    vswhere = os.path.join(
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        "Microsoft Visual Studio",
        "Installer",
        "vswhere.exe",
    )
    if not os.path.isfile(vswhere):
        return None
    try:
        install = subprocess.check_output(
            [vswhere, "-latest", "-products", "*", "-property", "installationPath"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    relatives = {
        "x64": ["x64"],
        "arm64": ["ARM64"],
        "x86": ["x86", ""],
    }
    for folder in relatives[_win_arch()]:
        candidate = os.path.join(
            install, "VC", "Tools", "Llvm", folder, "bin", "clang-cl.exe"
        )
        if os.path.isfile(candidate):
            return candidate
    return None


def _compiler_rt(clang):
    probe = subprocess.run(
        [clang, "/clang:-print-resource-dir"],
        check=False,
        capture_output=True,
        text=True,
    )
    text = (probe.stdout or "") + "\n" + (probe.stderr or "")
    resource = ""
    for line in text.splitlines():
        line = line.strip().strip('"')
        if line and os.path.isdir(line):
            resource = line
            break
    if not resource:
        raise RuntimeError("clang-cl did not report its resource directory:\n" + text)
    names = {
        "x64": "clang_rt.builtins-x86_64.lib",
        "arm64": "clang_rt.builtins-aarch64.lib",
        "x86": "clang_rt.builtins-i386.lib",
    }
    builtin = os.path.join(resource, "lib", "windows", names[_win_arch()])
    if not os.path.isfile(builtin):
        raise RuntimeError("clang-cl builtins library was not found at " + builtin)
    return builtin


def _drop(flags, rejected):
    return [flag for flag in flags if flag.upper() not in rejected]


class build_ext(_build_ext):
    def build_extensions(self):
        if self.compiler.compiler_type == "msvc":
            clang = _clang_cl()
            if not clang:
                raise RuntimeError(
                    "Windows builds use clang-cl from LLVM, which has C11 complex "
                    "and the same ABI as python.org Python. Install LLVM "
                    "(https://github.com/llvm/llvm-project/releases) or the Visual "
                    "Studio component C++ Clang Compiler for Windows, plus the "
                    "Microsoft C++ build tools."
                )
            if not self.compiler.initialized:
                self.compiler.initialize()
            self.compiler.cc = clang
            self.compiler.compile_options = _drop(self.compiler.compile_options, {"/GL"})
            seen = set()
            for flags in self.compiler._ldflags.values():
                if id(flags) in seen:
                    continue
                seen.add(id(flags))
                flags[:] = _drop(flags, {"/LTCG"})
            builtin = _compiler_rt(clang)
            for ext in self.extensions:
                ext.extra_compile_args = ["/O2", "/std:c11"]
                ext.libraries = []
                ext.extra_link_args = [builtin]
            print("QuadraticFormsMGHyp: compiling with " + clang)
        else:
            compile_args = ["-O3", "-std=c11", "-Wall"]
            link_args = []
            if sys.platform == "darwin":
                compile_args.append("-fblocks")
                link_args.append("-framework")
                link_args.append("Accelerate")
            for ext in self.extensions:
                ext.extra_compile_args = compile_args
                ext.extra_link_args = link_args
                ext.libraries = ["m"]
        super().build_extensions()


setup(
    cmdclass={"build_ext": build_ext},
    ext_modules=[
        Extension(
            "QuadraticFormsMGHyp._native",
            sources=["src/QuadraticFormsMGHyp/_native.c", "es4mgh.c"],
            include_dirs=["."],
        )
    ],
)
