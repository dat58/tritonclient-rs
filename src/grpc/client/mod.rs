pub(crate) mod channel;
mod config;
pub use config::*;
mod error;
pub use error::*;

use crate::grpc::output::ModelOutput;
use crate::grpc::pb::{self, GrpcInferenceServiceClient, HealthClient};
use crate::types::Bytes;
use channel::ChannelPool;
use std::collections::HashMap;
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

    pub async fn is_server_live(&self) -> Result<bool> {
        self.with_root_client(|mut client| async move {
            let result = client.server_live(pb::ServerLiveRequest {}).await?;
            Ok(result.into_inner().live)
        })
        .await
    }

    pub async fn is_model_ready(&self, model_name: &str, version: Option<&str>) -> Result<bool> {
        self.with_root_client(|mut client| async move {
            let result = client
                .model_ready(pb::ModelReadyRequest {
                    name: model_name.to_string(),
                    version: version.unwrap_or("").to_string(),
                })
                .await?;
            Ok(result.into_inner().ready)
        })
        .await
    }

    pub async fn server_metadata(&self) -> Result<pb::ServerMetadataResponse> {
        self.with_root_client(|mut client| async move {
            let result = client.server_metadata(pb::ServerMetadataRequest {}).await?;
            Ok(result.into_inner())
        })
        .await
    }

    pub async fn model_metadata(
        &self,
        model_name: &str,
        version: Option<&str>,
    ) -> Result<pb::ModelMetadataResponse> {
        self.with_root_client(|mut client| async move {
            let result = client
                .model_metadata(pb::ModelMetadataRequest {
                    name: model_name.to_string(),
                    version: version.unwrap_or("").to_string(),
                })
                .await?;
            Ok(result.into_inner())
        })
        .await
    }

    pub async fn model_config(
        &self,
        model_name: &str,
        version: Option<&str>,
    ) -> Result<pb::ModelConfigResponse> {
        self.with_root_client(|mut client| async move {
            let result = client
                .model_config(pb::ModelConfigRequest {
                    name: model_name.to_string(),
                    version: version.unwrap_or("").to_string(),
                })
                .await?;
            Ok(result.into_inner())
        })
        .await
    }

    pub async fn model_statistics(
        &self,
        model_name: &str,
        version: Option<&str>,
    ) -> Result<pb::ModelStatisticsResponse> {
        self.with_root_client(|mut client| async move {
            let result = client
                .model_statistics(pb::ModelStatisticsRequest {
                    name: model_name.to_string(),
                    version: version.unwrap_or("").to_string(),
                })
                .await?;
            Ok(result.into_inner())
        })
        .await
    }

    pub async fn repository_index(
        &self,
        repository_name: &str,
        ready: bool,
    ) -> Result<pb::RepositoryIndexResponse> {
        self.with_root_client(|mut client| async move {
            let result = client
                .repository_index(pb::RepositoryIndexRequest {
                    repository_name: repository_name.to_string(),
                    ready,
                })
                .await?;
            Ok(result.into_inner())
        })
        .await
    }

    pub async fn repository_model_load(
        &self,
        repository_name: &str,
        model_name: &str,
        parameters: Option<&HashMap<String, pb::ModelRepositoryParameter>>,
    ) -> Result<()> {
        self.with_root_client(|mut client| async move {
            client
                .repository_model_load(pb::RepositoryModelLoadRequest {
                    repository_name: repository_name.to_string(),
                    model_name: model_name.to_string(),
                    parameters: parameters.unwrap_or(&HashMap::new()).clone(),
                })
                .await?;
            Ok(())
        })
        .await
    }

    pub async fn repository_model_unload(
        &self,
        repository_name: &str,
        model_name: &str,
        parameters: Option<&HashMap<String, pb::ModelRepositoryParameter>>,
    ) -> Result<()> {
        self.with_root_client(|mut client| async move {
            client
                .repository_model_unload(pb::RepositoryModelUnloadRequest {
                    repository_name: repository_name.to_string(),
                    model_name: model_name.to_string(),
                    parameters: parameters.unwrap_or(&HashMap::new()).clone(),
                })
                .await?;
            Ok(())
        })
        .await
    }

    pub async fn system_shared_memory_status(
        &self,
        name: &str,
    ) -> Result<pb::SystemSharedMemoryStatusResponse> {
        self.with_root_client(|mut client| async move {
            let result = client
                .system_shared_memory_status(pb::SystemSharedMemoryStatusRequest {
                    name: name.to_string(),
                })
                .await?;
            Ok(result.into_inner())
        })
        .await
    }

    pub async fn system_shared_memory_register(
        &self,
        name: &str,
        key: &str,
        offset: u64,
        byte_size: u64,
    ) -> Result<()> {
        self.with_root_client(|mut client| async move {
            client
                .system_shared_memory_register(pb::SystemSharedMemoryRegisterRequest {
                    name: name.to_string(),
                    key: key.to_string(),
                    offset,
                    byte_size,
                })
                .await?;
            Ok(())
        })
        .await
    }

    pub async fn system_shared_memory_unregister(&self, name: &str) -> Result<()> {
        self.with_root_client(|mut client| async move {
            client
                .system_shared_memory_unregister(pb::SystemSharedMemoryUnregisterRequest {
                    name: name.to_string(),
                })
                .await?;
            Ok(())
        })
        .await
    }

    pub async fn cuda_shared_memory_status(
        &self,
        name: &str,
    ) -> Result<pb::CudaSharedMemoryStatusResponse> {
        self.with_root_client(|mut client| async move {
            let result = client
                .cuda_shared_memory_status(pb::CudaSharedMemoryStatusRequest {
                    name: name.to_string(),
                })
                .await?;
            Ok(result.into_inner())
        })
        .await
    }

    pub async fn cuda_shared_memory_register(
        &self,
        name: &str,
        raw_handle: &Bytes,
        device_id: i64,
        byte_size: u64,
    ) -> Result<()> {
        self.with_root_client(|mut client| async move {
            client
                .cuda_shared_memory_register(pb::CudaSharedMemoryRegisterRequest {
                    name: name.to_string(),
                    raw_handle: raw_handle.clone(),
                    device_id,
                    byte_size,
                })
                .await?;
            Ok(())
        })
        .await
    }

    pub async fn cuda_shared_memory_unregister(&self, name: &str) -> Result<()> {
        self.with_root_client(|mut client| async move {
            client
                .cuda_shared_memory_unregister(pb::CudaSharedMemoryUnregisterRequest {
                    name: name.to_string(),
                })
                .await?;
            Ok(())
        })
        .await
    }

    pub async fn trace_setting(
        &self,
        model_name: &str,
        settings: Option<&HashMap<String, pb::TraceSettingValue>>,
    ) -> Result<pb::TraceSettingResponse> {
        self.with_root_client(|mut client| async move {
            let result = client
                .trace_setting(pb::TraceSettingRequest {
                    settings: settings.unwrap_or(&HashMap::new()).clone(),
                    model_name: model_name.to_string(),
                })
                .await?;
            Ok(result.into_inner())
        })
        .await
    }

    pub async fn log_settings(
        &self,
        settings: Option<&HashMap<String, pb::LogSettingValue>>,
    ) -> Result<pb::LogSettingsResponse> {
        self.with_root_client(|mut client| async move {
            let result = client
                .log_settings(pb::LogSettingsRequest {
                    settings: settings.unwrap_or(&HashMap::new()).clone(),
                })
                .await?;
            Ok(result.into_inner())
        })
        .await
    }
}
