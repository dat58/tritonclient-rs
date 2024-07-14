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
    let ratio = std::mem::size_of::<T>() / std::mem::size_of::<u8>();
    let capacity = data.len() / ratio;
    unsafe {
        let ptr = data.as_ptr() as *mut T;
        std::mem::forget(data);
        Vec::from_raw_parts(ptr, capacity, capacity)
    }
}

fn vec_u8_base_16bits_to_vec_f32(data: Vec<u8>) -> Vec<f32> {
    let chunks = data.chunks(2);
    let mut vec = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        vec.push(f32::from_be_bytes([0, 0, chunk[0], chunk[1]]));
    }
    vec
}

fn vec_u8_to_bytes(data: Vec<u8>) -> Vec<Bytes> {
    let mut offset = 0;
    let mut vec = Vec::<Bytes>::new();
    while offset < data.len() {
        let length = u32::from_be_bytes([
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
                TritonDataTypes::FP16 | TritonDataTypes::BF16 => {
                    let array =
                        ArrayD::from_shape_vec(shape, vec_u8_base_16bits_to_vec_f32(raw_content))?;
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
