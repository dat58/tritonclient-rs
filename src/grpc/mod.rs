pub mod client;
pub mod input;
pub mod macros;
pub mod output;

pub mod pb {
    include!("proto/inference.rs");
    include!("proto/grpc.health.v1.rs");
    pub use grpc_inference_service_client::*;
    pub use health_client::*;
    pub use infer_parameter::*;
    pub use model_infer_request::*;
    pub use model_infer_response::*;
}
