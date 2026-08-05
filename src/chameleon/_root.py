"""Repository root resolution shared by chameleon modules."""
import os as _os

ROOT = _os.environ.get("LEVER_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))


def rpath(*parts):
    """Join ``parts`` onto the artifact root, so paths do not depend on the cwd."""
    return _os.path.join(ROOT, *parts)


#: Records this process has already written, so a study that appends across
#: several dumps keeps overwriting its OWN output rather than being redirected
#: on every call.
_WRITTEN = set()


def atomic_json_dump(obj, path, allow_overwrite=None):
    """Crash-safe full-file JSON write: tmp + fsync + atomic replace.

    A shipped record is not overwritten. The records under results/ are the
    evidence the reported numbers were computed from, and a study re-run is a
    fresh measurement, not a correction of them -- so when the target already
    exists and this process did not write it, the result goes to
    ``<name>.rerun.json`` beside it and the path taken is printed. Set
    CHAM_OVERWRITE_RECORDS=1 (or pass allow_overwrite=True) to replace them in
    place.
    """
    import json, os
    if allow_overwrite is None:
        allow_overwrite = os.environ.get("CHAM_OVERWRITE_RECORDS") == "1"
    real = os.path.abspath(path)
    if os.path.exists(real) and real not in _WRITTEN and not allow_overwrite:
        stem, ext = os.path.splitext(real)
        real = stem + ".rerun" + (ext or ".json")
        print("shipped record %s left in place; writing %s "
              "(CHAM_OVERWRITE_RECORDS=1 to replace it)"
              % (os.path.basename(path), os.path.basename(real)), flush=True)
    tmp = real + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, real)
    _WRITTEN.add(real)
    return real


def _pid_alive(pid):
    import os
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:          # someone else's process: it exists
        return True
    return True


def acquire_lock(path):
    """Take an exclusive lock next to an output file.

    A lock whose owning process is gone is stale and is taken over. atexit does
    not run when a job is killed or the machine goes down, so without this a
    single interrupted run would block that output forever -- and the runners are
    documented as safe to interrupt. A lock held by a LIVE process still raises,
    with a message saying which pid holds it and how to clear it.
    """
    import os, atexit
    lock = path + ".lock"
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                with open(lock) as fh:
                    owner = int((fh.read() or "0").strip() or 0)
            except (ValueError, OSError):
                owner = 0
            if owner and _pid_alive(owner):
                raise SystemExit(
                    "another run holds %s (pid %d). Wait for it to finish, or "
                    "remove the lock if that process is gone." % (lock, owner))
            try:
                os.remove(lock)      # stale: the owner is gone
            except FileNotFoundError:
                pass                 # someone else cleared it first; retry
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    atexit.register(lambda: os.path.exists(lock) and os.remove(lock))
    return lock
