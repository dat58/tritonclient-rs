macro_rules! array_to_tensor {
    ($array:expr; bool) => {
        InferTensorContents {
            bool_contents: $array.into_raw_vec(),
            ..Default::default()
        }
    };

    ($array:expr; i8) => {
        InferTensorContents {
            int_contents: $array
                .into_raw_vec()
                .into_iter()
                .map(|v| v as i32)
                .collect::<Vec<_>>(),
            ..Default::default()
        }
    };

    ($array:expr; i16) => {
        InferTensorContents {
            int_contents: $array
                .into_raw_vec()
                .into_iter()
                .map(|v| v as i32)
                .collect::<Vec<_>>(),
            ..Default::default()
        }
    };

    ($array:expr; i32) => {
        InferTensorContents {
            int_contents: $array.into_raw_vec(),
            ..Default::default()
        }
    };

    ($array:expr; i64) => {
        InferTensorContents {
            int64_contents: $array.into_raw_vec(),
            ..Default::default()
        }
    };

    ($array:expr; u8) => {
        InferTensorContents {
            uint_contents: $array
                .into_raw_vec()
                .into_iter()
                .map(|v| v as u32)
                .collect::<Vec<_>>(),
            ..Default::default()
        }
    };

    ($array:expr; u16) => {
        InferTensorContents {
            uint_contents: $array
                .into_raw_vec()
                .into_iter()
                .map(|v| v as u32)
                .collect::<Vec<_>>(),
            ..Default::default()
        }
    };

    ($array:expr; u32) => {
        InferTensorContents {
            uint_contents: $array.into_raw_vec(),
            ..Default::default()
        }
    };

    ($array:expr; u64) => {
        InferTensorContents {
            uint64_contents: $array.into_raw_vec(),
            ..Default::default()
        }
    };

    ($array:expr; f32) => {
        InferTensorContents {
            fp32_contents: $array.into_raw_vec(),
            ..Default::default()
        }
    };

    ($array:expr; f64) => {
        InferTensorContents {
            fp64_contents: $array.into_raw_vec(),
            ..Default::default()
        }
    };

    ($array:expr; Bytes) => {
        InferTensorContents {
            bytes_contents: $array.into_raw_vec(),
            ..Default::default()
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
