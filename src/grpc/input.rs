use super::pb::{InferInputTensor, InferParameter, InferTensorContents, ModelInferRequest};
use crate::types::{Bytes, TritonDataTypes};
use crate::{array_to_tensor, generate_trait_transform_infer_tensor_contents};
use ndarray::ArrayD;
use std::any::{Any, TypeId};
use std::collections::HashMap;

pub trait TransformInferTensorContents: Sized + 'static {
    fn transform(array: ArrayD<Self>) -> InferTensorContents;
}

#[derive(Clone, Debug)]
pub struct InferInput {
    inner: InferInputTensor,
}

impl InferInput {
    pub fn new() -> Self {
        Self {
            inner: InferInputTensor::default(),
        }
    }

    pub fn set_name(&mut self, name: String) {
        self.inner.name = name;
    }

    pub fn name(mut self, name: String) -> Self {
        self.set_name(name);
        self
    }

    pub fn set_datatype(&mut self, datatype: TritonDataTypes) {
        self.inner.datatype = datatype.to_string();
    }

    pub fn datatype(mut self, datatype: TritonDataTypes) -> Self {
        self.set_datatype(datatype);
        self
    }

    pub fn set_parameters(&mut self, parameters: HashMap<String, InferParameter>) {
        self.inner.parameters = parameters;
    }

    pub fn parameters(mut self, parameters: HashMap<String, InferParameter>) -> Self {
        self.set_parameters(parameters);
        self
    }

    pub fn set_data_from_ndarray<T>(&mut self, array: ArrayD<T>)
    where
        T: TransformInferTensorContents,
    {
        self.inner.shape = array
            .shape()
            .to_vec()
            .into_iter()
            .map(|v| v as i64)
            .collect();
        self.inner.contents = Some(TransformInferTensorContents::transform(array));
    }

    pub fn data_from_ndarray<T>(mut self, array: ArrayD<T>) -> Self
    where
        T: TransformInferTensorContents,
    {
        self.set_data_from_ndarray(array);
        self
    }

    pub(crate) fn build(self) -> InferInputTensor {
        self.inner
    }
}

#[derive(Clone, Debug)]
pub struct ModelInput {
    inner: ModelInferRequest,
}

impl ModelInput {
    pub fn new() -> Self {
        Self {
            inner: ModelInferRequest::default(),
        }
    }

    pub fn set_model_name<S: ToString>(&mut self, model_name: S) {
        self.inner.model_name = model_name.to_string();
    }

    pub fn model_name<S: ToString>(mut self, model_name: S) -> Self {
        self.set_model_name(model_name);
        self
    }

    pub fn set_input(&mut self, input: InferInput) {
        if self.inner.inputs.is_empty() {
            self.inner.inputs = vec![input.build()];
        } else {
            self.inner.inputs.push(input.build());
        }
    }

    pub fn input(mut self, input: InferInput) -> Self {
        self.set_input(input);
        self
    }

    pub fn set_inputs(&mut self, inputs: Vec<InferInput>) {
        self.inner.inputs = inputs
            .into_iter()
            .map(|input| input.build())
            .collect::<Vec<_>>();
    }

    pub fn inputs(mut self, inputs: Vec<InferInput>) -> Self {
        self.set_inputs(inputs);
        self
    }

    pub(crate) fn build(self) -> ModelInferRequest {
        self.inner
    }
}

generate_trait_transform_infer_tensor_contents!(bool);
generate_trait_transform_infer_tensor_contents!(i8);
generate_trait_transform_infer_tensor_contents!(i16);
generate_trait_transform_infer_tensor_contents!(i32);
generate_trait_transform_infer_tensor_contents!(i64);
generate_trait_transform_infer_tensor_contents!(u8);
generate_trait_transform_infer_tensor_contents!(u16);
generate_trait_transform_infer_tensor_contents!(u32);
generate_trait_transform_infer_tensor_contents!(u64);
generate_trait_transform_infer_tensor_contents!(f32);
generate_trait_transform_infer_tensor_contents!(f64);
generate_trait_transform_infer_tensor_contents!(Bytes);

impl Into<ModelInferRequest> for ModelInput {
    fn into(self) -> ModelInferRequest {
        self.build()
    }
}
