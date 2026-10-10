//! Replacing a file of a persist directory in one step.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{GprError, PersistErrorKind};

use super::persist_err;

/// Distinguishes temporary files of concurrent saves in one process.
static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

/// Writes `bytes` to `path` through a temporary file in the same directory,
/// then renames it over `path`.
///
/// A reader sees the old file or the new one, never a partial write. A model
/// that memory-maps the old file keeps its own copy: the rename gives `path`
/// a new file and leaves the mapped one untouched. On failure `path` is
/// unchanged and the temporary file is removed.
pub(super) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), GprError> {
    write_atomic_with(path, |file, temp| {
        file.write_all(bytes)
            .map_err(|err| persist_err(PersistErrorKind::Io, format!("write {temp:?}: {err}")))
    })
}

/// [`write_atomic`] of what `write` writes to the temporary file (handed
/// over open, with its path for messages), so a large file is written as it
/// is formed rather than built in memory first. The temporary file is made
/// new (an existing path or link there is refused) and synced through the
/// handle that wrote it before the rename.
pub(super) fn write_atomic_with(
    path: &Path,
    write: impl FnOnce(&mut File, &Path) -> Result<(), GprError>,
) -> Result<(), GprError> {
    let temp = temp_path(path)?;
    let mut file: File = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)
        .map_err(|err| persist_err(PersistErrorKind::Io, format!("create {temp:?}: {err}")))?;
    let synced = write(&mut file, &temp).and_then(|()| {
        file.sync_all()
            .map_err(|err| persist_err(PersistErrorKind::Io, format!("sync {temp:?}: {err}")))
    });
    // Closed before the rename or the removal, which Windows refuses on an
    // open file.
    drop(file);
    let written = synced.and_then(|()| {
        std::fs::rename(&temp, path)
            .map_err(|err| persist_err(PersistErrorKind::Io, format!("replace {path:?}: {err}")))
    });
    if written.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    written
}

fn temp_path(path: &Path) -> Result<PathBuf, GprError> {
    let name = path
        .file_name()
        .ok_or_else(|| persist_err(PersistErrorKind::Io, format!("{path:?} has no file name")))?;
    let mut temp_name = std::ffi::OsString::from(".");
    temp_name.push(name);
    temp_name.push(format!(
        ".tmp-{}-{}",
        std::process::id(),
        NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
    ));
    Ok(path.with_file_name(temp_name))
}

#[cfg(test)]
mod tests {
    use super::{write_atomic, write_atomic_with};
    use crate::error::{GprError, PersistErrorKind};

    fn scratch_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("gprx_atomic_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("dir");
        dir
    }

    fn entries(dir: &std::path::Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .expect("read dir")
            .map(|entry| {
                entry
                    .expect("entry")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        names.sort();
        names
    }

    #[test]
    fn replaces_the_file_and_leaves_no_temporary() {
        let dir = scratch_dir("replace");
        let path = dir.join("config.json");
        write_atomic(&path, b"old").expect("first");
        write_atomic(&path, b"new").expect("second");
        assert_eq!(std::fs::read(&path).expect("read"), b"new");
        assert_eq!(entries(&dir), ["config.json"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_failed_replace_keeps_the_target_and_removes_the_temporary() {
        let dir = scratch_dir("fail");
        // A non-empty directory cannot be replaced by a file.
        let path = dir.join("model.safetensors");
        std::fs::create_dir_all(path.join("inner")).expect("dir");
        assert!(write_atomic(&path, b"bytes").is_err());
        assert!(path.join("inner").is_dir());
        assert_eq!(entries(&dir), ["model.safetensors"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_failed_write_keeps_the_target_and_removes_the_temporary() {
        let dir = scratch_dir("write_fails");
        let path = dir.join("model.safetensors");
        write_atomic(&path, b"old").expect("first");
        let failed = write_atomic_with(&path, |file, _| {
            use std::io::Write;
            file.write_all(b"partial").expect("write");
            Err(GprError::PersistFailed {
                kind: PersistErrorKind::Tensor,
                reason: "stop".into(),
            })
        });
        assert!(failed.is_err());
        assert_eq!(std::fs::read(&path).expect("read"), b"old");
        assert_eq!(entries(&dir), ["model.safetensors"]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
