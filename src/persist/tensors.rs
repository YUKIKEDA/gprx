//! `model.safetensors` layout: original `x` / `y`, optional column-major `l`.

use std::fs::File;
use std::path::Path;

use faer::MatRef;
use memmap2::Mmap;
use safetensors::tensor::{Dtype, TensorView};
use safetensors::{SafeTensors, serialize_to_file};

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
        // SAFETY: this map stays alive on `MappedTensors` and is not written
        // through. gprx never writes into an existing `model.safetensors`:
        // a save renames a new file over the path (`atomic::write_atomic`),
        // so this mapping keeps the old file. Another program that truncates
        // or rewrites the file in place is outside what gprx can guard.
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
/// memory.
fn write_views(dir: &Path, views: Vec<(&str, TensorView<'_>)>) -> Result<(), GprError> {
    let path = dir.join(TENSOR_FILE);
    super::atomic::write_atomic_with(&path, |temp| {
        serialize_to_file(views, None, temp).map_err(|err| {
            let kind = match err {
                safetensors::SafeTensorError::IoError(_) => PersistErrorKind::Io,
                _ => PersistErrorKind::Tensor,
            };
            persist_err(kind, format!("write {temp:?}: {err}"))
        })
    })
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
