//! `model.safetensors` layout: original `x` / `y`, optional column-major `l`.

use std::fs::File;
use std::path::Path;

use faer::MatRef;
use memmap2::Mmap;
use std::io::Write;

use safetensors::tensor::{Dtype, TensorView};
use safetensors::{SafeTensors, View, serialize};

use crate::error::GprError;
use crate::error::PersistErrorKind;

use super::{TENSOR_FILE, persist_err};

pub(super) const TENSOR_X: &str = "x";
pub(super) const TENSOR_Y: &str = "y";
pub(super) const TENSOR_L: &str = "l";
pub(super) const TENSOR_ALPHA: &str = "alpha";

/// Memory map of `model.safetensors` kept so [`Self::l_view`] can borrow `L`.
pub(crate) struct MappedTensors {
    mmap: Mmap,
    l_offset: usize,
    n: usize,
}

/// `model.safetensors` opened once for a load: read into memory, or memory
/// mapped when its `f64` factor stays mapped ([`Self::into_mapped_l`]).
/// Every tensor of the load comes from this one read.
pub(super) struct TensorFile {
    source: TensorSource,
}

enum TensorSource {
    Read(Vec<u8>),
    Mapped(Mmap),
}

impl TensorFile {
    /// Reads the whole file into memory.
    pub(super) fn read(dir: &Path) -> Result<Self, GprError> {
        let path = dir.join(TENSOR_FILE);
        let bytes = std::fs::read(&path)
            .map_err(|err| persist_err(PersistErrorKind::Io, format!("read {path:?}: {err}")))?;
        Ok(Self {
            source: TensorSource::Read(bytes),
        })
    }

    /// Maps the file; only the pages a load touches are read.
    pub(super) fn map(dir: &Path) -> Result<Self, GprError> {
        let path = dir.join(TENSOR_FILE);
        let file = File::open(&path)
            .map_err(|err| persist_err(PersistErrorKind::Io, format!("open {path:?}: {err}")))?;
        // An empty file has no header to read (and cannot be mapped): the
        // same `Tensor` error a read of it gives.
        let len = file
            .metadata()
            .map_err(|err| persist_err(PersistErrorKind::Io, format!("stat {path:?}: {err}")))?
            .len();
        if len == 0 {
            return Err(persist_err(
                PersistErrorKind::Tensor,
                format!("{path:?} is empty: no safetensors header"),
            ));
        }
        // SAFETY: the map is not written through, and lives as long as this
        // `TensorFile` (a load that copies what it needs drops it before it
        // returns; `MappedTensors` keeps it for a model's `f64` factor).
        // gprx never writes into an existing `model.safetensors`: a save
        // renames a new file over the path (`atomic::write_atomic_with`), so
        // the mapping keeps the old file. Another program that truncates the
        // file in place while it is mapped makes a read of the lost pages
        // fault (SIGBUS on Unix); that is outside what gprx can guard.
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|err| persist_err(PersistErrorKind::Io, format!("mmap {path:?}: {err}")))?;
        Ok(Self {
            source: TensorSource::Mapped(mmap),
        })
    }

    fn bytes(&self) -> &[u8] {
        match &self.source {
            TensorSource::Read(bytes) => bytes,
            TensorSource::Mapped(mmap) => mmap,
        }
    }

    /// The parsed header over the file's bytes.
    pub(super) fn tensors(&self) -> Result<SafeTensors<'_>, GprError> {
        SafeTensors::deserialize(self.bytes()).map_err(|err| {
            persist_err(
                PersistErrorKind::Tensor,
                format!("safetensors header: {err}"),
            )
        })
    }

    /// Keeps the map so the model can borrow its `n × n` `f64` factor `l`.
    pub(super) fn into_mapped_l(self, n: usize) -> Result<MappedTensors, GprError> {
        match self.source {
            TensorSource::Mapped(mmap) => MappedTensors::from_mmap(mmap, n),
            TensorSource::Read(_) => Err(persist_err(
                PersistErrorKind::Tensor,
                "the factor was read, not mapped",
            )),
        }
    }
}

impl MappedTensors {
    fn from_mmap(mmap: Mmap, n: usize) -> Result<Self, GprError> {
        let tensors = SafeTensors::deserialize(&mmap).map_err(|err| {
            persist_err(
                PersistErrorKind::Tensor,
                format!("safetensors header: {err}"),
            )
        })?;
        let tensor = tensors.tensor(TENSOR_L).map_err(|err| {
            persist_err(
                PersistErrorKind::Tensor,
                format!("missing tensor {TENSOR_L}: {err}"),
            )
        })?;
        validate_f64_shape(&tensor, &[n, n], TENSOR_L)?;
        let data = tensor.data();
        if !(data.as_ptr() as usize).is_multiple_of(align_of::<f64>()) {
            return Err(persist_err(
                PersistErrorKind::Tensor,
                format!("tensor {TENSOR_L} is not aligned for f64"),
            ));
        }
        let l_offset = offset_in_mmap(&mmap, data)?;
        let nbytes = n
            .checked_mul(n)
            .and_then(|cells| cells.checked_mul(size_of::<f64>()))
            .ok_or_else(|| persist_err(PersistErrorKind::Tensor, "L byte length overflowed"))?;
        if l_offset.checked_add(nbytes).is_none() || l_offset + nbytes > mmap.len() {
            return Err(persist_err(
                PersistErrorKind::Tensor,
                "L tensor is outside the mapped file",
            ));
        }
        let mapped = Self { mmap, l_offset, n };
        // SAFETY: the bounds and alignment are checked above.
        let values = unsafe { f64_slice_unchecked(&mapped.mmap[l_offset..l_offset + nbytes]) };
        require_finite_tensor(values, TENSOR_L)?;
        Ok(mapped)
    }

    pub(crate) fn l_view(&self) -> MatRef<'_, f64> {
        let nbytes = self.n * self.n * size_of::<f64>();
        let bytes = &self.mmap[self.l_offset..self.l_offset + nbytes];
        // SAFETY: `from_mmap` checked this range for length and alignment, and the
        // mapping stays live for `'self`.
        let data = unsafe { f64_slice_unchecked(bytes) };
        MatRef::from_column_major_slice(data, self.n, self.n)
    }
}

pub(super) struct FactorBytes<'a> {
    pub l_dtype: Dtype,
    pub l: &'a [u8],
    pub alpha_dtype: Dtype,
    pub alpha: &'a [u8],
}

/// One more tensor a save writes beside `x` / `y` (and the factor): its
/// name, scalar, shape, and bytes.
pub(super) struct RawTensor<'a> {
    pub name: &'a str,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub bytes: &'a [u8],
}

impl<'a> RawTensor<'a> {
    fn view(&self) -> Result<(&'a str, TensorView<'a>), GprError> {
        let view = TensorView::new(self.dtype, self.shape.clone(), self.bytes).map_err(|err| {
            persist_err(
                PersistErrorKind::Tensor,
                format!("{} tensor: {err}", self.name),
            )
        })?;
        Ok((self.name, view))
    }
}

pub(super) fn scalar_bytes<T>(values: &[T]) -> &[u8] {
    // SAFETY: any `T` slice is a valid byte slice of `size_of_val` bytes.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

pub(super) fn pack_lower<T: crate::kernel::KernelScalar>(l: MatRef<'_, T>, out: &mut [T]) {
    let n = l.nrows();
    debug_assert_eq!(l.ncols(), n);
    debug_assert_eq!(out.len(), n * n);
    let zero = T::from_f64(0.0);
    out.fill(zero);
    for col in 0..n {
        for row in col..n {
            out[col * n + row] = l[(row, col)];
        }
    }
}

/// Writes `views` as `dir/model.safetensors`, tensor by tensor into a
/// temporary file that replaces the old one: the file is never built in
/// memory. The bytes are those of [`safetensors::serialize`].
fn write_views(dir: &Path, views: Vec<(&str, TensorView<'_>)>) -> Result<(), GprError> {
    let path = dir.join(TENSOR_FILE);
    super::atomic::write_atomic_with(&path, |file, temp| {
        let io = |err: std::io::Error| {
            persist_err(PersistErrorKind::Io, format!("write {temp:?}: {err}"))
        };
        let (header, views) = safetensors_header(views).map_err(|err| {
            persist_err(PersistErrorKind::Tensor, format!("write {temp:?}: {err}"))
        })?;
        let mut out = std::io::BufWriter::new(file);
        out.write_all(&header).map_err(io)?;
        for (_, view) in &views {
            out.write_all(view.data()).map_err(io)?;
        }
        out.flush().map_err(io)
    })
}

/// The header of a safetensors file of `views` (its length, then the JSON
/// padded to 8 bytes), and `views` in the order their data follows it:
/// descending dtype alignment, then name, as `safetensors` lays them out.
/// The header is `safetensors::serialize` of views that report their
/// lengths but carry no bytes, so it is the crate's own.
fn safetensors_header<'a, 'v>(
    mut views: Vec<(&'a str, TensorView<'v>)>,
) -> Result<(Vec<u8>, Vec<(&'a str, TensorView<'v>)>), safetensors::SafeTensorError> {
    views.sort_by(|(lname, left), (rname, right)| {
        right.dtype().cmp(&left.dtype()).then(lname.cmp(rname))
    });
    let header = serialize(
        views.iter().map(|(name, view)| (*name, HeaderOnly(view))),
        None,
    )?;
    Ok((header, views))
}

/// A tensor's dtype, shape, and length without its bytes.
struct HeaderOnly<'a, 'v>(&'a TensorView<'v>);

impl View for HeaderOnly<'_, '_> {
    fn dtype(&self) -> Dtype {
        self.0.dtype()
    }

    fn shape(&self) -> &[usize] {
        self.0.shape()
    }

    fn data(&self) -> std::borrow::Cow<'_, [u8]> {
        std::borrow::Cow::Borrowed(&[])
    }

    fn data_len(&self) -> usize {
        self.0.data_len()
    }
}

pub(super) fn write_tensors(
    dir: &Path,
    x: &[f64],
    y: &[f64],
    n: usize,
    d: usize,
    factor: Option<FactorBytes<'_>>,
    extra: &[RawTensor<'_>],
) -> Result<(), GprError> {
    if x.len() != n * d {
        return Err(persist_err(
            PersistErrorKind::Tensor,
            format!("x has {} values, expected n*d = {}", x.len(), n * d),
        ));
    }
    if y.len() != n {
        return Err(persist_err(
            PersistErrorKind::Tensor,
            format!("y has {} values, expected n = {n}", y.len()),
        ));
    }
    let x_bytes = f64_as_bytes(x);
    let y_bytes = f64_as_bytes(y);
    let x_view = TensorView::new(Dtype::F64, vec![n, d], x_bytes)
        .map_err(|err| persist_err(PersistErrorKind::Tensor, format!("x tensor: {err}")))?;
    let y_view = TensorView::new(Dtype::F64, vec![n], y_bytes)
        .map_err(|err| persist_err(PersistErrorKind::Tensor, format!("y tensor: {err}")))?;
    let views = if let Some(factor) = factor {
        let l_cells = match factor.l_dtype {
            Dtype::F32 => factor.l.len() / size_of::<f32>(),
            Dtype::F64 => factor.l.len() / size_of::<f64>(),
            other => {
                return Err(persist_err(
                    PersistErrorKind::Tensor,
                    format!("L dtype {other:?} is not f32 or f64"),
                ));
            }
        };
        let alpha_cells = match factor.alpha_dtype {
            Dtype::F32 => factor.alpha.len() / size_of::<f32>(),
            Dtype::F64 => factor.alpha.len() / size_of::<f64>(),
            other => {
                return Err(persist_err(
                    PersistErrorKind::Tensor,
                    format!("alpha dtype {other:?} is not f32 or f64"),
                ));
            }
        };
        if l_cells != n * n {
            return Err(persist_err(
                PersistErrorKind::Tensor,
                format!("L has {l_cells} values, expected n*n = {}", n * n),
            ));
        }
        if alpha_cells != n {
            return Err(persist_err(
                PersistErrorKind::Tensor,
                format!("alpha has {alpha_cells} values, expected n = {n}"),
            ));
        }
        let l_view = TensorView::new(factor.l_dtype, vec![n, n], factor.l)
            .map_err(|err| persist_err(PersistErrorKind::Tensor, format!("L tensor: {err}")))?;
        let alpha_view = TensorView::new(factor.alpha_dtype, vec![n], factor.alpha)
            .map_err(|err| persist_err(PersistErrorKind::Tensor, format!("alpha tensor: {err}")))?;
        vec![
            (TENSOR_X, x_view),
            (TENSOR_Y, y_view),
            (TENSOR_L, l_view),
            (TENSOR_ALPHA, alpha_view),
        ]
    } else {
        vec![(TENSOR_X, x_view), (TENSOR_Y, y_view)]
    };
    let mut views = views;
    for tensor in extra {
        views.push(tensor.view()?);
    }
    write_views(dir, views)
}

pub(super) fn read_xy(
    tensors: &SafeTensors<'_>,
    n: usize,
    d: usize,
) -> Result<(Vec<f64>, Vec<f64>), GprError> {
    let x = copy_f64_tensor(tensors, TENSOR_X, &[n, d])?;
    let y = copy_f64_tensor(tensors, TENSOR_Y, &[n])?;
    Ok((x, y))
}

pub(super) fn read_scalars<T: crate::kernel::KernelScalar>(
    tensors: &SafeTensors<'_>,
    name: &str,
    shape: &[usize],
    dtype: Dtype,
) -> Result<Vec<T>, GprError> {
    Ok(finite_tensor::<T>(tensors, name, shape, dtype)?.to_vec())
}

/// Tensor `name` of `shape` and `dtype`, read in place; a non-finite
/// value is [`PersistErrorKind::Tensor`].
pub(super) fn finite_tensor<'a, T: crate::kernel::KernelScalar>(
    tensors: &SafeTensors<'a>,
    name: &str,
    shape: &[usize],
    dtype: Dtype,
) -> Result<&'a [T], GprError> {
    let data = scalar_tensor::<T>(tensors, name, shape, dtype)?;
    require_finite_tensor(data, name)?;
    Ok(data)
}

/// Tensor `name` of `shape` and `dtype`, read in place. Its values are
/// not checked.
fn scalar_tensor<'a, T: crate::kernel::KernelScalar>(
    tensors: &SafeTensors<'a>,
    name: &str,
    shape: &[usize],
    dtype: Dtype,
) -> Result<&'a [T], GprError> {
    let tensor = tensors.tensor(name).map_err(|err| {
        persist_err(
            PersistErrorKind::Tensor,
            format!("missing tensor {name}: {err}"),
        )
    })?;
    validate_shape(&tensor, shape, name, dtype)?;
    scalar_slice::<T>(tensor.data())
}

pub(super) fn read_matrix<T: crate::kernel::KernelScalar>(
    tensors: &SafeTensors<'_>,
    n: usize,
    dtype: Dtype,
) -> Result<faer::Mat<T>, GprError> {
    let values = read_scalars::<T>(tensors, TENSOR_L, &[n, n], dtype)?;
    Ok(faer::Mat::from_fn(n, n, |row, col| values[col * n + row]))
}

fn copy_f64_tensor(
    tensors: &SafeTensors<'_>,
    name: &str,
    shape: &[usize],
) -> Result<Vec<f64>, GprError> {
    let tensor = tensors.tensor(name).map_err(|err| {
        persist_err(
            PersistErrorKind::Tensor,
            format!("missing tensor {name}: {err}"),
        )
    })?;
    validate_f64_shape(&tensor, shape, name)?;
    let data = f64_slice(tensor.data())?;
    require_finite_tensor(data, name)?;
    Ok(data.to_vec())
}

/// Rejects a stored tensor holding `NaN` or `±∞`. gprx never writes one, so
/// such a file was damaged or edited.
fn require_finite_tensor<T: crate::kernel::KernelScalar>(
    values: &[T],
    name: &str,
) -> Result<(), GprError> {
    match values.iter().position(|value| !value.to_f64().is_finite()) {
        None => Ok(()),
        Some(index) => Err(persist_err(
            PersistErrorKind::Tensor,
            format!("tensor {name} holds a non-finite value at index {index}"),
        )),
    }
}

fn validate_shape(
    tensor: &TensorView<'_>,
    shape: &[usize],
    name: &str,
    dtype: Dtype,
) -> Result<(), GprError> {
    if tensor.dtype() != dtype {
        return Err(persist_err(
            PersistErrorKind::Tensor,
            format!(
                "tensor {name} dtype is {:?}, expected {dtype:?}",
                tensor.dtype()
            ),
        ));
    }
    if tensor.shape() != shape {
        return Err(persist_err(
            PersistErrorKind::Tensor,
            format!(
                "tensor {name} shape is {:?}, expected {shape:?}",
                tensor.shape()
            ),
        ));
    }
    Ok(())
}

fn validate_f64_shape(
    tensor: &TensorView<'_>,
    shape: &[usize],
    name: &str,
) -> Result<(), GprError> {
    validate_shape(tensor, shape, name, Dtype::F64)?;
    Ok(())
}

fn offset_in_mmap(mmap: &Mmap, data: &[u8]) -> Result<usize, GprError> {
    let base = mmap.as_ptr() as usize;
    let ptr = data.as_ptr() as usize;
    if ptr < base {
        return Err(persist_err(
            PersistErrorKind::Tensor,
            "tensor pointer is before the mapped file",
        ));
    }
    Ok(ptr - base)
}

fn f64_as_bytes(values: &[f64]) -> &[u8] {
    // SAFETY: an `f64` slice is a valid byte slice of `size_of_val` bytes.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn scalar_slice<T: Copy>(bytes: &[u8]) -> Result<&[T], GprError> {
    // An empty tensor (`x` of a model without coordinates) may sit at any
    // offset.
    if bytes.is_empty() {
        return Ok(&[]);
    }
    if !(bytes.as_ptr() as usize).is_multiple_of(align_of::<T>()) {
        return Err(persist_err(
            PersistErrorKind::Tensor,
            "tensor is not aligned",
        ));
    }
    if !bytes.len().is_multiple_of(size_of::<T>()) {
        return Err(persist_err(
            PersistErrorKind::Tensor,
            "tensor length is not a multiple of the scalar size",
        ));
    }
    // SAFETY: alignment and the length multiple were checked above.
    Ok(unsafe {
        std::slice::from_raw_parts(bytes.as_ptr().cast::<T>(), bytes.len() / size_of::<T>())
    })
}

fn f64_slice(bytes: &[u8]) -> Result<&[f64], GprError> {
    scalar_slice::<f64>(bytes)
}

/// Reinterprets `bytes` as `f64` values.
///
/// # Safety
///
/// `bytes` must be aligned for `f64` and its length must be a multiple of
/// `size_of::<f64>()`. The caller keeps the underlying allocation alive for
/// the returned slice.
unsafe fn f64_slice_unchecked(bytes: &[u8]) -> &[f64] {
    // SAFETY: the caller upholds alignment, length, and lifetime.
    unsafe {
        std::slice::from_raw_parts(bytes.as_ptr().cast::<f64>(), bytes.len() / size_of::<f64>())
    }
}

/// Writes `model.safetensors` with the named `f64` tensors `(name, shape,
/// values)`, each column-major for a matrix.
pub(super) fn write_f64_tensors(
    dir: &Path,
    tensors: &[(&str, Vec<usize>, &[f64])],
    extra: &[RawTensor<'_>],
) -> Result<(), GprError> {
    let mut views = Vec::with_capacity(tensors.len());
    for (name, shape, values) in tensors {
        let cells: usize = shape.iter().product();
        if values.len() != cells {
            return Err(persist_err(
                PersistErrorKind::Tensor,
                format!("{name} has {} values, expected {cells}", values.len()),
            ));
        }
        let view =
            TensorView::new(Dtype::F64, shape.clone(), f64_as_bytes(values)).map_err(|err| {
                persist_err(PersistErrorKind::Tensor, format!("{name} tensor: {err}"))
            })?;
        views.push((*name, view));
    }
    for tensor in extra {
        views.push(tensor.view()?);
    }
    write_views(dir, views)
}

/// The `f64` tensor `name` of shape `shape`, read in place.
pub(super) fn f64_tensor<'a>(
    tensors: &SafeTensors<'a>,
    name: &str,
    shape: &[usize],
) -> Result<&'a [f64], GprError> {
    let tensor = tensors.tensor(name).map_err(|err| {
        persist_err(
            PersistErrorKind::Tensor,
            format!("missing tensor {name}: {err}"),
        )
    })?;
    validate_f64_shape(&tensor, shape, name)?;
    let data = f64_slice(tensor.data())?;
    require_finite_tensor(data, name)?;
    Ok(data)
}

/// Reads the `f64` tensor `name` of shape `shape` from `model.safetensors`.
pub(super) fn read_f64(
    tensors: &SafeTensors<'_>,
    name: &str,
    shape: &[usize],
) -> Result<Vec<f64>, GprError> {
    read_scalars::<f64>(tensors, name, shape, Dtype::F64)
}

#[cfg(test)]
mod tests {
    use safetensors::tensor::{Dtype, TensorView};

    use super::{TENSOR_FILE, TensorFile, write_views};
    use crate::error::{GprError, PersistErrorKind};

    fn scratch_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("gprx_tensors_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("dir");
        dir
    }

    fn bytes_of(values: &[f64]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn a_streamed_file_has_the_bytes_of_serialize() {
        let dir = scratch_dir("stream");
        let (a, b) = (bytes_of(&[1.0, 2.0, 3.0]), bytes_of(&[4.0]));
        let small: Vec<u8> = [1.5_f32, 2.5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let views = || {
            vec![
                ("y", TensorView::new(Dtype::F64, vec![3], &a).expect("y")),
                (
                    "s",
                    TensorView::new(Dtype::F32, vec![2], &small).expect("s"),
                ),
                ("x", TensorView::new(Dtype::F64, vec![1, 1], &b).expect("x")),
            ]
        };
        write_views(&dir, views()).expect("write");
        let want = safetensors::serialize(views(), None).expect("serialize");
        assert_eq!(std::fs::read(dir.join(TENSOR_FILE)).expect("read"), want);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_empty_file_is_a_tensor_error_mapped_or_read() {
        let dir = scratch_dir("empty");
        std::fs::write(dir.join(TENSOR_FILE), b"").expect("write");
        let kind = |result: Result<(), GprError>| match result {
            Err(GprError::PersistFailed { kind, .. }) => Some(kind),
            _ => None,
        };
        let mapped = TensorFile::map(&dir).and_then(|file| file.tensors().map(drop));
        let read = TensorFile::read(&dir).and_then(|file| file.tensors().map(drop));
        assert_eq!(kind(mapped), Some(PersistErrorKind::Tensor));
        assert_eq!(kind(read), Some(PersistErrorKind::Tensor));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
