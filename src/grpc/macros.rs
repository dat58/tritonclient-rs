macro_rules! array_to_tensor {
    ($array:expr; bool) => {
        if $array.type_id() != TypeId::of::<ArrayD<bool>>() {
            panic!("Expected ArrayD<bool> data type!");
        } else {
            InferTensorContents {
                bool_contents: $array.into_raw_vec(),
                ..Default::default()
            }
        }
    };

    ($array:expr; i8) => {
        if $array.type_id() != TypeId::of::<ArrayD<i8>>() {
            panic!("Expected ArrayD<i8> data type!");
        } else {
            InferTensorContents {
                int_contents: $array
                    .into_raw_vec()
                    .into_iter()
                    .map(|v| v as i32)
                    .collect::<Vec<_>>(),
                ..Default::default()
            }
        }
    };

    ($array:expr; i16) => {
        if $array.type_id() != TypeId::of::<ArrayD<i16>>() {
            panic!("Expected ArrayD<i16> data type!");
        } else {
            InferTensorContents {
                int_contents: $array
                    .into_raw_vec()
                    .into_iter()
                    .map(|v| v as i32)
                    .collect::<Vec<_>>(),
                ..Default::default()
            }
        }
    };

    ($array:expr; i32) => {
        if $array.type_id() != TypeId::of::<ArrayD<i32>>() {
            panic!("Expected ArrayD<i32> data type!");
        } else {
            InferTensorContents {
                int_contents: $array.into_raw_vec(),
                ..Default::default()
            }
        }
    };

    ($array:expr; i64) => {
        if $array.type_id() != TypeId::of::<ArrayD<i64>>() {
            panic!("Expected ArrayD<i64> data type!");
        } else {
            InferTensorContents {
                int64_contents: $array.into_raw_vec(),
                ..Default::default()
            }
        }
    };

    ($array:expr; u8) => {
        if $array.type_id() != TypeId::of::<ArrayD<u8>>() {
            panic!("Expected ArrayD<u8> data type!");
        } else {
            InferTensorContents {
                uint_contents: $array
                    .into_raw_vec()
                    .into_iter()
                    .map(|v| v as u32)
                    .collect::<Vec<_>>(),
                ..Default::default()
            }
        }
    };

    ($array:expr; u16) => {
        if $array.type_id() != TypeId::of::<ArrayD<u16>>() {
            panic!("Expected ArrayD<u16> data type!");
        } else {
            InferTensorContents {
                uint_contents: $array
                    .into_raw_vec()
                    .into_iter()
                    .map(|v| v as u32)
                    .collect::<Vec<_>>(),
                ..Default::default()
            }
        }
    };

    ($array:expr; u32) => {
        if $array.type_id() != TypeId::of::<ArrayD<u32>>() {
            panic!("Expected ArrayD<u32> data type!");
        } else {
            InferTensorContents {
                uint_contents: $array.into_raw_vec(),
                ..Default::default()
            }
        }
    };

    ($array:expr; u64) => {
        if $array.type_id() != TypeId::of::<ArrayD<u64>>() {
            panic!("Expected ArrayD<u64> data type!");
        } else {
            InferTensorContents {
                uint64_contents: $array.into_raw_vec(),
                ..Default::default()
            }
        }
    };

    ($array:expr; f32) => {
        if $array.type_id() != TypeId::of::<ArrayD<f32>>() {
            panic!("Expected ArrayD<f32> data type!");
        } else {
            InferTensorContents {
                fp32_contents: $array.into_raw_vec(),
                ..Default::default()
            }
        }
    };

    ($array:expr; f64) => {
        if $array.type_id() != TypeId::of::<ArrayD<f64>>() {
            panic!("Expected ArrayD<f64> data type!");
        } else {
            InferTensorContents {
                fp64_contents: $array.into_raw_vec(),
                ..Default::default()
            }
        }
    };

    ($array:expr; Bytes) => {
        if $array.type_id() != TypeId::of::<ArrayD<Bytes>>() {
            panic!("Expected ArrayD<Bytes> data type!");
        } else {
            InferTensorContents {
                bytes_contents: $array.into_raw_vec(),
                ..Default::default()
            }
        }
    };
}

macro_rules! generate_trait_transform_infer_tensor_contents {
    ($dtype:ident) => {
        impl TransformInferTensorContents for $dtype {
            fn transform(array: ArrayD<Self>) -> InferTensorContents {
                array_to_tensor!(array; $dtype)
            }
        }
    };
}

pub(crate) use array_to_tensor;
pub(crate) use generate_trait_transform_infer_tensor_contents;
