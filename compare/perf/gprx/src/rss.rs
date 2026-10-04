//! Peak resident set for the current process: peak working set (Windows),
//! `VmHWM` (Linux), or `ru_maxrss` (other Unix).
//!
//! Linux `ru_maxrss` is not used. `exec` folds the high-water mark of the
//! pre-exec address space, the parent's under fork or vfork, into it, so a
//! runner launched from a large harness reports the harness's size. `VmHWM`
//! belongs to the current address space, which `exec` replaces.

#[cfg(windows)]
pub fn peak_rss_bytes() -> Result<u64, String> {
    #[repr(C)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut core::ffi::c_void;
    }

    #[link(name = "psapi")]
    unsafe extern "system" {
        fn GetProcessMemoryInfo(
            process: *mut core::ffi::c_void,
            counters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
    }

    let mut counters = ProcessMemoryCounters {
        cb: 0,
        page_fault_count: 0,
        peak_working_set_size: 0,
        working_set_size: 0,
        quota_peak_paged_pool_usage: 0,
        quota_paged_pool_usage: 0,
        quota_peak_non_paged_pool_usage: 0,
        quota_non_paged_pool_usage: 0,
        pagefile_usage: 0,
        peak_pagefile_usage: 0,
    };
    counters.cb = u32::try_from(core::mem::size_of::<ProcessMemoryCounters>())
        .map_err(|_| "PROCESS_MEMORY_COUNTERS size does not fit u32".to_string())?;
    // SAFETY: `GetCurrentProcess` returns a pseudo-handle. `counters` is a
    // valid `PROCESS_MEMORY_COUNTERS` whose `cb` field is the struct size.
    let ok = unsafe {
        GetProcessMemoryInfo(
            GetCurrentProcess(),
            &mut counters as *mut ProcessMemoryCounters,
            counters.cb,
        )
    };
    if ok == 0 {
        return Err("GetProcessMemoryInfo failed".to_string());
    }
    u64::try_from(counters.peak_working_set_size)
        .map_err(|_| "peak working set does not fit u64".to_string())
}

#[cfg(target_os = "linux")]
pub fn peak_rss_bytes() -> Result<u64, String> {
    let status = std::fs::read_to_string("/proc/self/status")
        .map_err(|e| format!("read /proc/self/status: {e}"))?;
    let line = status
        .lines()
        .find_map(|l| l.strip_prefix("VmHWM:"))
        .ok_or_else(|| "VmHWM missing from /proc/self/status".to_string())?;
    let kib = line
        .trim()
        .strip_suffix("kB")
        .ok_or_else(|| format!("VmHWM has no kB suffix: {line:?}"))?
        .trim()
        .parse::<u64>()
        .map_err(|e| format!("parse VmHWM {line:?}: {e}"))?;
    Ok(kib.saturating_mul(1024))
}

#[cfg(all(unix, not(target_os = "linux")))]
pub fn peak_rss_bytes() -> Result<u64, String> {
    let mut usage = unsafe { core::mem::zeroed::<libc::rusage>() };
    // SAFETY: `usage` is a writable `rusage` and `RUSAGE_SELF` is a valid who.
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    if rc != 0 {
        return Err("getrusage failed".to_string());
    }
    let raw =
        u64::try_from(usage.ru_maxrss).map_err(|_| "ru_maxrss does not fit u64".to_string())?;
    // macOS reports bytes. The other BSDs report KiB.
    let bytes = if cfg!(target_os = "macos") {
        raw
    } else {
        raw.saturating_mul(1024)
    };
    Ok(bytes)
}

#[cfg(not(any(windows, unix)))]
pub fn peak_rss_bytes() -> Result<u64, String> {
    Err("peak RSS is not implemented on this OS".to_string())
}
