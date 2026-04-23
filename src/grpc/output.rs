use super::client::Result;
use super::pb::ModelInferResponse;
use crate::types::{Bytes, TritonDataTypes};
use ndarray::ArrayD;
use std::collections::HashMap;

#[derive(Debug)]
pub enum ArrayOutputOneOf {
    BOOL(ArrayD<bool>),
    INT8(ArrayD<i8>),
    INT16(ArrayD<i16>),
    INT32(ArrayD<i32>),
    INT64(ArrayD<i64>),
    UINT8(ArrayD<u8>),
    UINT16(ArrayD<u16>),
    UINT32(ArrayD<u32>),
    UINT64(ArrayD<u64>),
    FP32(ArrayD<f32>),
    FP64(ArrayD<f64>),
    BYTES(ArrayD<Bytes>),
}

fn vec_u8_to_vec_t<T: Sized>(data: Vec<u8>) -> Vec<T> {
    let elem_size = std::mem::size_of::<T>();
    let len = data.len() / elem_size;
    let mut result = Vec::<T>::with_capacity(len);
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            result.as_mut_ptr() as *mut u8,
            len * elem_size,
        );
        result.set_len(len);
    }
    result
}

fn fp16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x3ff) as u32;

    let f32_bits = if exp == 0 {
        if mantissa == 0 {
            sign << 31
        } else {
            // Subnormal FP16: normalize by finding the leading 1
            let mut shift = 0u32;
            let mut m = mantissa;
            while (m & 0x400) == 0 {
                m <<= 1;
                shift += 1;
            }
            // New f32 exponent: (127 - 15) - shift = 112 - shift
            (sign << 31) | ((112u32.wrapping_sub(shift)) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 31 {
        // Inf or NaN
        (sign << 31) | (0xffu32 << 23) | (mantissa << 13)
    } else {
        // Normal: adjust exponent bias from 15 to 127 (add 112)
        (sign << 31) | ((exp + 112) << 23) | (mantissa << 13)
    };
    f32::from_bits(f32_bits)
}

fn vec_fp16_to_vec_f32(data: Vec<u8>) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|c| fp16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

// BF16 is the upper 16 bits of f32; in little-endian the two BF16 bytes map
// directly to bytes 2 and 3 of the corresponding f32 value.
fn vec_bf16_to_vec_f32(data: Vec<u8>) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|c| f32::from_le_bytes([0, 0, c[0], c[1]]))
        .collect()
}

fn vec_u8_to_bytes(data: Vec<u8>) -> Vec<Bytes> {
    let mut offset = 0;
    let mut vec = Vec::<Bytes>::new();
    while offset < data.len() {
        let length = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;
        let bytes = Vec::from(&data[offset..offset + length]);
        offset += length;
        vec.push(bytes);
    }
    vec
}

#[derive(Debug)]
pub struct ModelOutput {
    inner: HashMap<String, ArrayOutputOneOf>,
}

impl ModelOutput {
    pub fn new(response: ModelInferResponse) -> Result<Self> {
        let mut inner = HashMap::new();
        for (raw_content, output) in response
            .raw_output_contents
            .into_iter()
            .zip(response.outputs)
        {
            let shape = output
                .shape
                .into_iter()
                .map(|v| v as usize)
                .collect::<Vec<_>>();

            match TritonDataTypes::from(output.datatype) {
                TritonDataTypes::BOOL => {
                    let array =
                        ArrayD::from_shape_vec(shape, vec_u8_to_vec_t::<bool>(raw_content))?;
                    inner.insert(output.name, ArrayOutputOneOf::BOOL(array));
                }
                TritonDataTypes::INT8 => {
                    let array = ArrayD::from_shape_vec(shape, vec_u8_to_vec_t::<i8>(raw_content))?;
                    inner.insert(output.name, ArrayOutputOneOf::INT8(array));
                }
                TritonDataTypes::INT16 => {
                    let array = ArrayD::from_shape_vec(shape, vec_u8_to_vec_t::<i16>(raw_content))?;
                    inner.insert(output.name, ArrayOutputOneOf::INT16(array));
                }
                TritonDataTypes::INT32 => {
                    let array = ArrayD::from_shape_vec(shape, vec_u8_to_vec_t::<i32>(raw_content))?;
                    inner.insert(output.name, ArrayOutputOneOf::INT32(array));
                }
                TritonDataTypes::INT64 => {
                    let array = ArrayD::from_shape_vec(shape, vec_u8_to_vec_t::<i64>(raw_content))?;
                    inner.insert(output.name, ArrayOutputOneOf::INT64(array));
                }
                TritonDataTypes::UINT8 => {
                    let array = ArrayD::from_shape_vec(shape, vec_u8_to_vec_t::<u8>(raw_content))?;
                    inner.insert(output.name, ArrayOutputOneOf::UINT8(array));
                }
                TritonDataTypes::UINT16 => {
                    let array = ArrayD::from_shape_vec(shape, vec_u8_to_vec_t::<u16>(raw_content))?;
                    inner.insert(output.name, ArrayOutputOneOf::UINT16(array));
                }
                TritonDataTypes::UINT32 => {
                    let array = ArrayD::from_shape_vec(shape, vec_u8_to_vec_t::<u32>(raw_content))?;
                    inner.insert(output.name, ArrayOutputOneOf::UINT32(array));
                }
                TritonDataTypes::UINT64 => {
                    let array = ArrayD::from_shape_vec(shape, vec_u8_to_vec_t::<u64>(raw_content))?;
                    inner.insert(output.name, ArrayOutputOneOf::UINT64(array));
                }
                TritonDataTypes::FP16 => {
                    let array =
                        ArrayD::from_shape_vec(shape, vec_fp16_to_vec_f32(raw_content))?;
                    inner.insert(output.name, ArrayOutputOneOf::FP32(array));
                }
                TritonDataTypes::BF16 => {
                    let array =
                        ArrayD::from_shape_vec(shape, vec_bf16_to_vec_f32(raw_content))?;
                    inner.insert(output.name, ArrayOutputOneOf::FP32(array));
                }
                TritonDataTypes::FP32 => {
                    let array = ArrayD::from_shape_vec(shape, vec_u8_to_vec_t::<f32>(raw_content))?;
                    inner.insert(output.name, ArrayOutputOneOf::FP32(array));
                }
                TritonDataTypes::FP64 => {
                    let array = ArrayD::from_shape_vec(shape, vec_u8_to_vec_t::<f64>(raw_content))?;
                    inner.insert(output.name, ArrayOutputOneOf::FP64(array));
                }
                TritonDataTypes::BYTES => {
                    let array = ArrayD::from_shape_vec(shape, vec_u8_to_bytes(raw_content))?;
                    inner.insert(output.name, ArrayOutputOneOf::BYTES(array));
                }
            }
        }
        Ok(Self { inner })
    }

    pub fn as_ndarray(&self, name: &str) -> Option<&ArrayOutputOneOf> {
        self.inner.get(name)
    }

    pub fn pop(&mut self, name: &str) -> Option<ArrayOutputOneOf> {
        self.inner.remove(name)
    }

    pub fn into_inner(self) -> HashMap<String, ArrayOutputOneOf> {
        self.inner
    }
}
