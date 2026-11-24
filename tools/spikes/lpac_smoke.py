"""Disposable Windows LPAC experiment; no protected ACLs or host policy changes."""
import ctypes as C
from ctypes import wintypes as W
import datetime
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import uuid

ROOT = Path('D:/work/_wt/lpac-smoke')
PYTHON = r'C:\ProgramData\spyder-6\python.exe'
PUBLIC = ('1.1.1.1', 443)
K = C.WinDLL('kernel32', use_last_error=True)
A = C.WinDLL('advapi32', use_last_error=True)
U = C.WinDLL('userenv', use_last_error=True)
P = C.c_void_p
DWORD = W.DWORD


def api(dll, name, restype, *args):
    fn = getattr(dll, name)
    fn.restype, fn.argtypes = restype, args
    return fn


class STARTUPINFO(C.Structure):
    _fields_ = [('cb', DWORD), ('reserved', W.LPWSTR), ('desktop', W.LPWSTR),
                ('title', W.LPWSTR), ('x', DWORD), ('y', DWORD), ('xsize', DWORD),
                ('ysize', DWORD), ('xchars', DWORD), ('ychars', DWORD),
                ('fill', DWORD), ('flags', DWORD), ('show', W.WORD),
                ('reserved2', W.WORD), ('reservedptr', P), ('stdin', W.HANDLE),
                ('stdout', W.HANDLE), ('stderr', W.HANDLE)]


class STARTUPINFOEX(C.Structure):
    _fields_ = [('startup', STARTUPINFO), ('attributes', P)]


class PROCESS_INFORMATION(C.Structure):
    _fields_ = [('process', W.HANDLE), ('thread', W.HANDLE), ('pid', DWORD), ('tid', DWORD)]


class SECURITY_CAPABILITIES(C.Structure):
    _fields_ = [('sid', P), ('capabilities', P), ('count', DWORD), ('reserved', DWORD)]


create_profile = api(U, 'CreateAppContainerProfile', C.c_long, W.LPCWSTR, W.LPCWSTR, W.LPCWSTR, P, DWORD, C.POINTER(P))
delete_profile = api(U, 'DeleteAppContainerProfile', C.c_long, W.LPCWSTR)
sid_string = api(A, 'ConvertSidToStringSidW', W.BOOL, P, C.POINTER(W.LPWSTR))
free_sid = api(A, 'FreeSid', P, P)
local_free = api(K, 'LocalFree', P, P)
init_attrs = api(K, 'InitializeProcThreadAttributeList', W.BOOL, P, DWORD, DWORD, C.POINTER(C.c_size_t))
update_attr = api(K, 'UpdateProcThreadAttribute', W.BOOL, P, DWORD, C.c_size_t, P, C.c_size_t, P, P)
delete_attrs = api(K, 'DeleteProcThreadAttributeList', None, P)
create_process = api(K, 'CreateProcessW', W.BOOL, W.LPCWSTR, W.LPWSTR, P, P, W.BOOL, DWORD, P, W.LPCWSTR, C.POINTER(STARTUPINFOEX), C.POINTER(PROCESS_INFORMATION))
wait = api(K, 'WaitForSingleObject', DWORD, W.HANDLE, DWORD)
get_exit = api(K, 'GetExitCodeProcess', W.BOOL, W.HANDLE, C.POINTER(DWORD))
terminate = api(K, 'TerminateProcess', W.BOOL, W.HANDLE, W.UINT)
close = api(K, 'CloseHandle', W.BOOL, W.HANDLE)


def check(ok, operation):
    if not ok:
        raise C.WinError(C.get_last_error(), operation)


def error(exc):
    return {'type': type(exc).__name__, 'message': str(exc),
            'winerror': getattr(exc, 'winerror', None), 'errno': getattr(exc, 'errno', None)}


def child(folder, sibling, port):
    out = {}
    def probe(name, fn, expect_success):
        try:
            fn()
            out[name] = {'pass': expect_success, 'success': True, 'error': None}
        except OSError as exc:
            denied = getattr(exc, 'winerror', None) in (5, 10013)
            out[name] = {'pass': not expect_success and denied, 'success': False, 'error': error(exc)}
    def granted():
        path = folder / 'roundtrip.txt'
        path.write_text('lpac nonce', encoding='utf-8')
        if path.read_text(encoding='utf-8') != 'lpac nonce':
            raise ValueError('roundtrip mismatch')
    def connect(host, port):
        with socket.create_connection((host, port), timeout=5):
            pass
    probe('granted_write_read', granted, True)
    probe('sibling_read_denied', lambda: sibling.read_text(encoding='utf-8'), False)
    def sibling_write():
        with (sibling.parent / ('forbidden-' + folder.name + '.txt')).open('x') as f:
            f.write('x')
    probe('sibling_write_denied', sibling_write, False)
    probe('loopback_connect_denied', lambda: connect('127.0.0.1', port), False)
    probe('public_connect_denied', lambda: connect(*PUBLIC), False)
    (folder / 'child-result.json').write_text(json.dumps(out), encoding='utf-8')


def run():
    names = ['granted_write_read', 'sibling_read_denied', 'sibling_write_denied', 'loopback_connect_denied', 'public_connect_denied']
    result = {'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'checks': {name: {'pass': False, 'status': 'not_run', 'error': None} for name in names},
              'elevation': {'parent_elevated': bool(C.windll.shell32.IsUserAnAdmin()), 'needed': None},
              'steps': {}, 'cleanup': {}}
    folder = ROOT / ('run-' + uuid.uuid4().hex)
    sibling = ROOT / 'sibling.txt'
    profile = 'lpac-smoke-' + uuid.uuid4().hex
    sid = P()
    made_profile = made_root = made_sibling = made_folder = attrs_ready = False
    listener = None
    try:
        if result['elevation']['parent_elevated']:
            raise RuntimeError('Parent is elevated; refusing to run.')
        if not ROOT.exists():
            ROOT.mkdir()
            made_root = True
        folder.mkdir()
        made_folder = True
        with sibling.open('x', encoding='utf-8') as f:
            made_sibling = True
            f.write('same-user readable sibling')
        hr = create_profile(profile, profile, 'Disposable LPAC smoke test', None, 0, C.byref(sid))
        result['steps']['create_profile'] = {'hresult': hr & 0xffffffff, 'success': hr == 0}
        if hr != 0:
            raise OSError(hr & 0xffffffff, 'CreateAppContainerProfile failed')
        made_profile = True
        sid_text = W.LPWSTR()
        check(sid_string(sid, C.byref(sid_text)), 'ConvertSidToStringSidW')
        try:
            package_sid = sid_text.value
        finally:
            local_free(C.cast(sid_text, P))
        result['package_sid'] = package_sid
        for label, args in [('grant', ['/grant', '*' + package_sid + ':(OI)(CI)M']),
                            ('integrity', ['/setintegritylevel', '(OI)(CI)L'])]:
            proc = subprocess.run(['icacls.exe', str(folder), *args], capture_output=True, text=True)
            result['steps'][label] = {'exit_code': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr}
            if proc.returncode:
                raise RuntimeError('icacls ' + label + ' failed')
        child_script = folder / 'lpac_smoke.py'
        shutil.copyfile(__file__, child_script)
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        listener.listen(2)
        port = listener.getsockname()[1]
        with socket.create_connection(('127.0.0.1', port), timeout=5):
            conn, _ = listener.accept()
            conn.close()
        try:
            with socket.create_connection(PUBLIC, timeout=5):
                result['steps']['public_control'] = {'success': True, 'error': None}
        except OSError as exc:
            result['steps']['public_control'] = {'success': False, 'error': error(exc)}
        size = C.c_size_t()
        init_attrs(None, 2, 0, C.byref(size))
        buf = C.create_string_buffer(size.value)
        attrs = C.cast(buf, P)
        check(init_attrs(attrs, 2, 0, C.byref(size)), 'InitializeProcThreadAttributeList')
        attrs_ready = True
        caps = SECURITY_CAPABILITIES(sid, None, 0, 0)
        policy = DWORD(1)
        check(update_attr(attrs, 0, 0x20009, C.byref(caps), C.sizeof(caps), None, None), 'security capabilities')
        check(update_attr(attrs, 0, 0x2000f, C.byref(policy), C.sizeof(policy), None, None), 'LPAC policy')
        si = STARTUPINFOEX()
        si.startup.cb = C.sizeof(si)
        si.attributes = attrs
        pi = PROCESS_INFORMATION()
        command = subprocess.list2cmdline([PYTHON, '-I', str(child_script), '--child', str(folder), str(sibling), str(port)])
        ok = create_process(PYTHON, C.create_unicode_buffer(command), None, None, False, 0x00080000 | 0x08000000, None, str(folder), C.byref(si), C.byref(pi))
        if not ok:
            code = C.get_last_error()
            result['steps']['launch'] = {'success': False, 'winerror': code}
            raise C.WinError(code, 'CreateProcessW')
        result['steps']['launch'] = {'success': True, 'pid': pi.pid}
        try:
            if wait(pi.process, 30000) != 0:
                terminate(pi.process, 1)
                wait(pi.process, 5000)
                raise TimeoutError('child did not exit within 30 seconds')
            exit_code = DWORD()
            check(get_exit(pi.process, C.byref(exit_code)), 'GetExitCodeProcess')
            result['steps']['child_exit_code'] = exit_code.value
        finally:
            close(pi.thread)
            close(pi.process)
        child_result = folder / 'child-result.json'
        if child_result.exists():
            result['checks'].update(json.loads(child_result.read_text(encoding='utf-8')))
            if not result['steps']['public_control']['success']:
                result['checks']['public_connect_denied']['pass'] = False
                result['checks']['public_connect_denied']['status'] = 'invalid_public_control'
        else:
            raise RuntimeError('child did not produce a result')
        result['elevation']['needed'] = False
    except Exception as exc:
        result['failure'] = error(exc)
        if getattr(exc, 'winerror', None) == 740:
            result['elevation']['needed'] = True
    finally:
        if listener:
            listener.close()
        if attrs_ready:
            delete_attrs(attrs)
        if made_profile:
            hr = delete_profile(profile)
            result['cleanup']['profile'] = {'success': hr == 0, 'hresult': hr & 0xffffffff}
        if sid:
            free_sid(sid)
        try:
            if made_folder:
                shutil.rmtree(folder)
            if made_sibling:
                sibling.unlink()
                forbidden = ROOT / ('forbidden-' + folder.name + '.txt')
                if forbidden.exists():
                    forbidden.unlink()
            if made_root:
                ROOT.rmdir()
            result['cleanup']['files'] = {'success': True}
        except OSError as exc:
            result['cleanup']['files'] = {'success': False, 'error': error(exc)}
    return result


if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[1] == '--child':
        child(Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]))
    else:
        print(json.dumps(run(), indent=2))
