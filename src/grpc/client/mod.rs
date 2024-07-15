pub(crate) mod channel;
mod config;
pub use config::*;
mod error;
pub use error::*;

use crate::grpc::output::ModelOutput;
use crate::grpc::pb::{self, GrpcInferenceServiceClient, HealthClient};
use channel::ChannelPool;
use std::future::Future;
use tonic::transport::Channel;

pub struct InferenceServerClient {
    pub config: InferenceServerClientConfig,
    channel: ChannelPool,
}

impl InferenceServerClient {
    pub fn new(config: InferenceServerClientConfig) -> Self {
        Self {
            channel: ChannelPool::from(config.clone()),
            config,
        }
    }

    async fn with_root_client<T, O: Future<Output = Result<T>>>(
        &self,
        f: impl Fn(GrpcInferenceServiceClient<Channel>) -> O,
    ) -> Result<T> {
        let result = self
            .channel
            .with_channel(
                |channel| {
                    let mut client = GrpcInferenceServiceClient::new(channel)
                        .max_decoding_message_size(usize::MAX)
                        .max_encoding_message_size(usize::MAX);
                    if let Some(compression) = self.config.compression {
                        client = client
                            .send_compressed(compression.into())
                            .accept_compressed(compression.into());
                    }
                    f(client)
                },
                true,
            )
            .await?;
        Ok(result)
    }

    pub async fn health_check(&self) -> Result<pb::HealthCheckResponse> {
        let channel = self.channel.get_channel().await?;
        let mut health_check_client = HealthClient::new(channel);
        let result = health_check_client
            .check(pb::HealthCheckRequest {
                service: String::new(),
            })
            .await?;
        Ok(result.into_inner())
    }

    pub async fn infer(&self, request: impl Into<pb::ModelInferRequest>) -> Result<ModelOutput> {
        let request = &request.into();
        self.with_root_client(|mut client| async move {
            let result = client.model_infer(request.clone()).await?;
            ModelOutput::new(result.into_inner())
        })
        .await
    }

    pub async fn is_server_ready(&self) -> Result<bool> {
        self.with_root_client(|mut client| async move {
            let result = client.server_ready(pb::ServerReadyRequest {}).await?;
            Ok(result.into_inner().ready)
        })
        .await
    }
}
