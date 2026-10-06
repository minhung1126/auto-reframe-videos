"""Build pinned static FFmpeg dependencies on each native desktop runner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from auto_reframe_core.platform_profile import desktop_target


def prepare_source(name, entry, cache, destination):
    archive = cache / (name + '.tar.gz')
    if not archive.is_file() or hashlib.sha256(archive.read_bytes()).hexdigest() != entry['sha256']:
        temporary = archive.with_suffix('.download')
        with urlopen(entry['url'], timeout=60) as response, temporary.open('wb') as sink:
            shutil.copyfileobj(response, sink)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != entry['sha256']:
            temporary.unlink()
            raise RuntimeError('Source checksum mismatch: ' + name)
        temporary.replace(archive)
    if destination.exists():
        marker = destination / '.source-sha256'
        roots = [p for p in destination.iterdir() if p.is_dir()]
        if marker.is_file() and marker.read_text() == entry['sha256'] and len(roots) == 1:
            return roots[0]
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        if sum(m.size for m in members) > 512 * 1024 * 1024:
            raise RuntimeError('Source archive exceeds limit')
        bundle.extractall(destination, filter='data')
    roots = list(destination.iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise RuntimeError('Invalid source archive layout')
    (destination / '.source-sha256').write_text(entry['sha256'])
    return roots[0]


def build_vendor(target, destination):
    windows = target == 'windows-x64'
    macos = target.startswith('macos')
    destination = Path(destination).resolve()
    workspace = ROOT / 'build' / 'ffmpeg-build'
    prefix = workspace / 'prefix'
    cache = ROOT / 'build' / 'ffmpeg-source-cache'
    cache.mkdir(parents=True, exist_ok=True)
    prefix.mkdir(parents=True, exist_ok=True)
    entries = json.loads((ROOT / 'packaging' / 'ffmpeg-sources.json').read_text())
    names = ['ffmpeg', 'x264', 'x265', 'freetype', 'harfbuzz', 'zlib']
    if windows:
        names += ['nvcodec', 'amf', 'vpl']
    sources = {name: prepare_source(name, entries[name], cache, workspace / ('src-' + name)) for name in names}
    # Archive snapshots must not inherit this application's parent Git version.
    (sources['ffmpeg'] / 'VERSION').write_text(entries['ffmpeg']['version'] + '\n')
    jobs = str(min(8, os.cpu_count() or 2))
    env = dict(os.environ)
    env['PKG_CONFIG_PATH'] = str(prefix / 'lib' / 'pkgconfig')
    env['CMAKE_PREFIX_PATH'] = str(prefix)
    if macos:
        env['MACOSX_DEPLOYMENT_TARGET'] = '13.0'
    def run(*args, cwd=None):
        print('Building:', args[0], flush=True)
        subprocess.run([str(a) for a in args], cwd=cwd or workspace, env=env, check=True)
    def cmake(name, *options, source=None, install=True):
        directory = workspace / ('cmake-' + name)
        command = [sys.executable, '-m', 'cmake', '-S', str(source or sources[name]), '-B', str(directory), '-G', 'Ninja',
                   '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_INSTALL_PREFIX=' + prefix.as_posix(),
                   '-DCMAKE_INSTALL_LIBDIR=lib', '-DBUILD_SHARED_LIBS=OFF',
                   '-DCMAKE_POSITION_INDEPENDENT_CODE=ON', '-DCMAKE_POLICY_VERSION_MINIMUM=3.5']
        if windows:
            command += ['-DCMAKE_C_COMPILER=gcc', '-DCMAKE_CXX_COMPILER=g++']
        if macos:
            command += ['-DCMAKE_OSX_DEPLOYMENT_TARGET=13.0', '-DCMAKE_OSX_ARCHITECTURES=' + ('arm64' if target.endswith('arm64') else 'x86_64')]
        run(*command, *options)
        run(sys.executable, '-m', 'cmake', '--build', directory, '--parallel', jobs)
        if install:
            run(sys.executable, '-m', 'cmake', '--install', directory)
        return directory
    zlib_build = cmake('zlib', '-DZLIB_BUILD_EXAMPLES=OFF', install=False)
    static_zlib = zlib_build / ('libzlibstatic.a' if windows else 'libz.a')
    (prefix / 'lib').mkdir(exist_ok=True)
    (prefix / 'include').mkdir(exist_ok=True)
    shutil.copyfile(static_zlib, prefix / 'lib' / 'libz.a')
    shutil.copyfile(sources['zlib'] / 'zlib.h', prefix / 'include' / 'zlib.h')
    shutil.copyfile(zlib_build / 'zconf.h', prefix / 'include' / 'zconf.h')
    cmake('freetype', '-DFT_DISABLE_ZLIB=ON', '-DFT_DISABLE_BZIP2=ON', '-DFT_DISABLE_PNG=ON',
          '-DFT_DISABLE_HARFBUZZ=ON', '-DFT_DISABLE_BROTLI=ON')
    cmake('harfbuzz', '-DHB_BUILD_UTILS=OFF', '-DHB_BUILD_SUBSET=OFF', '-DHB_BUILD_TESTS=OFF',
          '-DHB_HAVE_FREETYPE=ON', '-DHB_HAVE_GLIB=OFF', '-DHB_HAVE_ICU=OFF', '-DHB_HAVE_CORETEXT=OFF')
    # Upstream's optional CMake build may omit the pkg-config file.
    pkgconfig = prefix / 'lib' / 'pkgconfig'
    pkgconfig.mkdir(parents=True, exist_ok=True)
    if not (pkgconfig / 'harfbuzz.pc').exists():
        (pkgconfig / 'harfbuzz.pc').write_text(
            f'prefix={prefix.as_posix()}\nlibdir=${{prefix}}/lib\nincludedir=${{prefix}}/include\n'
            f'Name: harfbuzz\nDescription: HarfBuzz text shaping\nVersion: {entries["harfbuzz"]["version"]}\n'
            'Libs: -L${libdir} -lharfbuzz\nLibs.private: -lstdc++\nCflags: -I${includedir}/harfbuzz\n')
    x264_args = ['bash', './configure', '--prefix=' + prefix.as_posix(), '--enable-static', '--enable-pic', '--disable-cli']
    if windows:
        x264_args += ['--host=x86_64-w64-mingw32']
    run(*x264_args, cwd=sources['x264'])
    run('make', '-j' + jobs, cwd=sources['x264'])
    run('make', 'install', cwd=sources['x264'])
    cmake('x265', '-DENABLE_SHARED=OFF', '-DENABLE_CLI=OFF', '-DENABLE_PIC=ON', source=sources['x265'] / 'source')
    hardware = []
    if windows:
        run('make', 'PREFIX=' + prefix.as_posix(), 'install', cwd=sources['nvcodec'])
        shutil.copytree(sources['amf'] / 'amf' / 'public' / 'include', prefix / 'include' / 'AMF', dirs_exist_ok=True)
        cmake('vpl', '-DBUILD_SHARED_LIBS=OFF', '-DBUILD_DEV=OFF', '-DBUILD_TOOLS=OFF', '-DBUILD_TESTS=OFF', '-DINSTALL_EXAMPLE_CODE=OFF')
        hardware = ['--enable-ffnvcodec', '--enable-cuda', '--enable-cuvid', '--enable-nvdec',
                    '--enable-nvenc', '--enable-amf', '--enable-libvpl', '--enable-d3d11va', '--enable-dxva2']
    configure = ['sh', './configure', '--prefix=' + prefix.as_posix(), '--enable-static', '--disable-shared',
                 '--disable-debug', '--disable-doc', '--disable-ffplay', '--disable-autodetect',
                 '--enable-gpl', '--enable-zlib', '--enable-libx264', '--enable-libx265', '--enable-libfreetype',
                 '--enable-libharfbuzz', '--pkg-config-flags=--static', '--extra-cflags=-I' + (prefix / 'include').as_posix(),
                 '--extra-ldflags=-L' + (prefix / 'lib').as_posix()] + hardware
    if macos:
        configure += ['--enable-videotoolbox', '--enable-audiotoolbox', '--extra-libs=-lc++']
    elif windows:
        configure += ['--target-os=mingw32', '--arch=x86_64', '--extra-ldflags=-static -static-libgcc -static-libstdc++ -L' + (prefix / 'lib').as_posix(), '--extra-libs=-lstdc++']
    else:
        configure += ['--extra-libs=-lstdc++']
    run(*configure, cwd=sources['ffmpeg'])
    run('make', '-j' + jobs, cwd=sources['ffmpeg'])
    (destination / 'bin').mkdir(parents=True, exist_ok=True)
    licenses = destination / 'licenses'
    licenses.mkdir(exist_ok=True)
    suffix = '.exe' if windows else ''
    hashes = {}
    for name in ('ffmpeg', 'ffprobe'):
        output = destination / 'bin' / (name + suffix)
        shutil.copy2(sources['ffmpeg'] / (name + suffix), output)
        hashes[output.name] = hashlib.sha256(output.read_bytes()).hexdigest()
    license_patterns = {
        'ffmpeg': ['COPYING.GPLv2', 'COPYING.GPLv3'], 'x264': ['COPYING'], 'x265': ['COPYING'],
        'freetype': ['docs/FTL.TXT'], 'harfbuzz': ['COPYING'],
        'nvcodec': ['include/ffnvcodec/' + h for h in ('dynlink_cuda.h', 'dynlink_cuviddec.h',
                    'dynlink_loader.h', 'dynlink_nvcuvid.h', 'nvEncodeAPI.h')],
        'amf': ['LICENSE.txt', 'LICENSE'], 'vpl': ['LICENSE'], 'zlib': ['LICENSE'],
    }
    for name in names:
        files = [sources[name] / p for p in license_patterns[name] if (sources[name] / p).is_file()]
        if not files:
            raise RuntimeError('Missing upstream license: ' + name)
        label = 'FFmpeg' if name == 'ffmpeg' else name
        (licenses / (label + '.txt')).write_text('\n\n'.join(p.read_text(errors='replace') for p in files), encoding='utf-8')
    runtime_sources = json.loads((ROOT / 'packaging' / 'runtime-licenses.json').read_text())
    for name, entry in runtime_sources.items():
        cached = cache / (name + '-license.txt')
        if not cached.is_file() or hashlib.sha256(cached.read_bytes()).hexdigest() != entry['sha256']:
            with urlopen(entry['url'], timeout=30) as response:
                content = response.read(1024 * 1024)
            if hashlib.sha256(content).hexdigest() != entry['sha256']:
                raise RuntimeError('Runtime license checksum mismatch')
            cached.write_bytes(content)
        shutil.copyfile(cached, licenses / (name + '.txt'))
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    provenance = {'target': target, 'version': entries['ffmpeg']['version'], 'binaries': hashes,
        'source_url': entries['ffmpeg']['url'],
        'build_recipe_url': f'https://github.com/minhung1126/auto-reframe-videos/blob/{revision}/scripts/build_ffmpeg_vendor.py',
        'corresponding_sources': {name: entries[name]['url'] for name in names},
        'components': {name: entries[name] for name in names}, 'runtime_license_sources': runtime_sources}
    (destination / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print('Native FFmpeg vendor bundle:', destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    actual = 'local-linux' if sys.platform == 'linux' else desktop_target()
    if actual != args.target:
        parser.error('Build on the native OS/CPU')
    build_vendor(args.target, args.destination)


if __name__ == '__main__':
    main()
