//! `model.safetensors` layout: original `x` / `y`, optional column-major `l`.

use std::fs::File;
use std::path::Path;

use faer::MatRef;
use memmap2::Mmap;
use safetensors::tensor::{Dtype, TensorView};
use safetensors::{SafeTensors, serialize};

use crate::error::GprError;

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

impl MappedTensors {
    pub(super) fn open(dir: &Path, n: usize) -> Result<Self, GprError> {
        let path = dir.join(TENSOR_FILE);
        let file = File::open(&path).map_err(|err| persist_err(format!("open {path:?}: {err}")))?;
        // Safety: this map stays alive on `MappedTensors` and is not written
        // through. gprx never writes into an existing `model.safetensors`:
        // a save renames a new file over the path (`atomic::write_atomic`),
        // so this mapping keeps the old file. Another program that truncates
        // or rewrites the file in place is outside what gprx can guard.
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|err| persist_err(format!("mmap {path:?}: {err}")))?;
        let tensors = SafeTensors::deserialize(&mmap)
            .map_err(|err| persist_err(format!("safetensors header: {err}")))?;
        let tensor = tensors
            .tensor(TENSOR_L)
            .map_err(|err| persist_err(format!("missing tensor {TENSOR_L}: {err}")))?;
        validate_f64_shape(&tensor, &[n, n], TENSOR_L)?;
        let data = tensor.data();
        if data.as_ptr() as usize % align_of::<f64>() != 0 {
            return Err(persist_err(format!(
                "tensor {TENSOR_L} is not aligned for f64"
            )));
        }
        let l_offset = offset_in_mmap(&mmap, data)?;
        let nbytes = n
            .checked_mul(n)
            .and_then(|cells| cells.checked_mul(size_of::<f64>()))
            .ok_or_else(|| persist_err("L byte length overflowed"))?;
        if l_offset.checked_add(nbytes).is_none() || l_offset + nbytes > mmap.len() {
            return Err(persist_err("L tensor is outside the mapped file"));
        }
        Ok(Self { mmap, l_offset, n })
    }

    pub(crate) fn l_view(&self) -> MatRef<'_, f64> {
        let nbytes = self.n * self.n * size_of::<f64>();
        let bytes = &self.mmap[self.l_offset..self.l_offset + nbytes];
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

pub(super) fn scalar_bytes<T>(values: &[T]) -> &[u8] {
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

pub(super) fn write_tensors(
    dir: &Path,
    x: &[f64],
    y: &[f64],
    n: usize,
    d: usize,
    factor: Option<FactorBytes<'_>>,
) -> Result<(), GprError> {
    if x.len() != n * d {
        return Err(persist_err(format!(
            "x has {} values, expected n*d = {}",
            x.len(),
            n * d
        )));
    }
    if y.len() != n {
        return Err(persist_err(format!(
            "y has {} values, expected n = {n}",
            y.len()
        )));
    }
    let x_bytes = f64_as_bytes(x);
    let y_bytes = f64_as_bytes(y);
    let x_view = TensorView::new(Dtype::F64, vec![n, d], x_bytes)
        .map_err(|err| persist_err(format!("x tensor: {err}")))?;
    let y_view = TensorView::new(Dtype::F64, vec![n], y_bytes)
        .map_err(|err| persist_err(format!("y tensor: {err}")))?;
    let bytes = if let Some(factor) = factor {
        let l_cells = match factor.l_dtype {
            Dtype::F32 => factor.l.len() / size_of::<f32>(),
            Dtype::F64 => factor.l.len() / size_of::<f64>(),
            other => {
                return Err(persist_err(format!("L dtype {other:?} is not f32 or f64")));
            }
        };
        let alpha_cells = match factor.alpha_dtype {
            Dtype::F32 => factor.alpha.len() / size_of::<f32>(),
            Dtype::F64 => factor.alpha.len() / size_of::<f64>(),
            other => {
                return Err(persist_err(format!(
                    "alpha dtype {other:?} is not f32 or f64"
                )));
            }
        };
        if l_cells != n * n {
            return Err(persist_err(format!(
                "L has {l_cells} values, expected n*n = {}",
                n * n
            )));
        }
        if alpha_cells != n {
            return Err(persist_err(format!(
                "alpha has {alpha_cells} values, expected n = {n}"
            )));
        }
        let l_view = TensorView::new(factor.l_dtype, vec![n, n], factor.l)
            .map_err(|err| persist_err(format!("L tensor: {err}")))?;
        let alpha_view = TensorView::new(factor.alpha_dtype, vec![n], factor.alpha)
            .map_err(|err| persist_err(format!("alpha tensor: {err}")))?;
        serialize(
            [
                (TENSOR_X, x_view),
                (TENSOR_Y, y_view),
                (TENSOR_L, l_view),
                (TENSOR_ALPHA, alpha_view),
            ],
            None,
        )
    } else {
        serialize([(TENSOR_X, x_view), (TENSOR_Y, y_view)], None)
    }
    .map_err(|err| persist_err(format!("serialize safetensors: {err}")))?;
    let path = dir.join(TENSOR_FILE);
    super::atomic::write_atomic(&path, &bytes)
}

pub(super) fn read_xy(dir: &Path, n: usize, d: usize) -> Result<(Vec<f64>, Vec<f64>), GprError> {
    let path = dir.join(TENSOR_FILE);
    let bytes = std::fs::read(&path).map_err(|err| persist_err(format!("read {path:?}: {err}")))?;
    let tensors = SafeTensors::deserialize(&bytes)
        .map_err(|err| persist_err(format!("safetensors header: {err}")))?;
    let x = copy_f64_tensor(&tensors, TENSOR_X, &[n, d])?;
    let y = copy_f64_tensor(&tensors, TENSOR_Y, &[n])?;
    Ok((x, y))
}

pub(super) fn read_scalars<T: Copy>(
    dir: &Path,
    name: &str,
    shape: &[usize],
    dtype: Dtype,
) -> Result<Vec<T>, GprError> {
    let path = dir.join(TENSOR_FILE);
    let bytes = std::fs::read(&path).map_err(|err| persist_err(format!("read {path:?}: {err}")))?;
    let tensors = SafeTensors::deserialize(&bytes)
        .map_err(|err| persist_err(format!("safetensors header: {err}")))?;
    let tensor = tensors
        .tensor(name)
        .map_err(|err| persist_err(format!("missing tensor {name}: {err}")))?;
    validate_shape(&tensor, shape, name, dtype)?;
    let data = scalar_slice::<T>(tensor.data())?;
    Ok(data.to_vec())
}

pub(super) fn read_matrix<T: Copy>(
    dir: &Path,
    n: usize,
    dtype: Dtype,
) -> Result<faer::Mat<T>, GprError> {
    let values = read_scalars::<T>(dir, TENSOR_L, &[n, n], dtype)?;
    Ok(faer::Mat::from_fn(n, n, |row, col| values[col * n + row]))
}

fn copy_f64_tensor(
    tensors: &SafeTensors<'_>,
    name: &str,
    shape: &[usize],
) -> Result<Vec<f64>, GprError> {
    let tensor = tensors
        .tensor(name)
        .map_err(|err| persist_err(format!("missing tensor {name}: {err}")))?;
    validate_f64_shape(&tensor, shape, name)?;
    let data = f64_slice(tensor.data())?;
    Ok(data.to_vec())
}

fn validate_shape(
    tensor: &TensorView<'_>,
    shape: &[usize],
    name: &str,
    dtype: Dtype,
) -> Result<(), GprError> {
    if tensor.dtype() != dtype {
        return Err(persist_err(format!(
            "tensor {name} dtype is {:?}, expected {dtype:?}",
            tensor.dtype()
        )));
    }
    if tensor.shape() != shape {
        return Err(persist_err(format!(
            "tensor {name} shape is {:?}, expected {shape:?}",
            tensor.shape()
        )));
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
        return Err(persist_err("tensor pointer is before the mapped file"));
    }
    Ok(ptr - base)
}

fn f64_as_bytes(values: &[f64]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn scalar_slice<T: Copy>(bytes: &[u8]) -> Result<&[T], GprError> {
    if bytes.as_ptr() as usize % align_of::<T>() != 0 {
        return Err(persist_err("tensor is not aligned"));
    }
    if bytes.len() % size_of::<T>() != 0 {
        return Err(persist_err(
            "tensor length is not a multiple of the scalar size",
        ));
    }
    Ok(unsafe {
        std::slice::from_raw_parts(bytes.as_ptr().cast::<T>(), bytes.len() / size_of::<T>())
    })
}

fn f64_slice(bytes: &[u8]) -> Result<&[f64], GprError> {
    scalar_slice::<f64>(bytes)
}

unsafe fn f64_slice_unchecked(bytes: &[u8]) -> &[f64] {
    unsafe {
        std::slice::from_raw_parts(bytes.as_ptr().cast::<f64>(), bytes.len() / size_of::<f64>())
    }
}

/// Writes `model.safetensors` with the named `f64` tensors `(name, shape,
/// values)`, each column-major for a matrix.
pub(super) fn write_f64_tensors(
    dir: &Path,
    tensors: &[(&str, Vec<usize>, &[f64])],
) -> Result<(), GprError> {
    let mut views = Vec::with_capacity(tensors.len());
    for (name, shape, values) in tensors {
        let cells: usize = shape.iter().product();
        if values.len() != cells {
            return Err(persist_err(format!(
                "{name} has {} values, expected {cells}",
                values.len()
            )));
        }
        let view = TensorView::new(Dtype::F64, shape.clone(), f64_as_bytes(values))
            .map_err(|err| persist_err(format!("{name} tensor: {err}")))?;
        views.push((*name, view));
    }
    let bytes = serialize(views, None)
        .map_err(|err| persist_err(format!("serialize safetensors: {err}")))?;
    let path = dir.join(TENSOR_FILE);
    super::atomic::write_atomic(&path, &bytes)
}

/// Reads the `f64` tensor `name` of shape `shape` from `model.safetensors`.
pub(super) fn read_f64(dir: &Path, name: &str, shape: &[usize]) -> Result<Vec<f64>, GprError> {
    read_scalars::<f64>(dir, name, shape, Dtype::F64)
}
