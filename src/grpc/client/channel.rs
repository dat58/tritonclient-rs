use super::InferenceServerClientConfig;
use super::{Error, Result};
use std::future::Future;
use std::sync::RwLock;
use std::time::Duration;
use tonic::transport::{Channel, ClientTlsConfig, Uri};
use tonic::{Code, Status};

pub struct ChannelPool {
    channel: RwLock<Option<Channel>>,
    uri: Uri,
    tls: bool,
    timeout: Duration,
    connection_timeout: Duration,
    keep_alive_while_idle: bool,
    keep_alive_timeout: Duration,
}

impl ChannelPool {
    pub fn new(
        uri: Uri,
        tls: bool,
        timeout: Duration,
        connection_timeout: Duration,
        keep_alive_while_idle: bool,
        keep_alive_timeout: Duration,
    ) -> Self {
        Self {
            channel: RwLock::new(None),
            uri,
            tls,
            timeout,
            connection_timeout,
            keep_alive_while_idle,
            keep_alive_timeout,
        }
    }

    async fn make_channel(&self) -> Result<Channel> {
        let mut tls = self.tls;
        let mut uri = self.uri.clone();
        match uri.scheme_str() {
            Some(scheme) => {
                let scheme = scheme.to_lowercase();
                if scheme == "https" {
                    tls = true;
                } else if scheme == "http" {
                    tls = false;
                } else {
                    return Err(Status::invalid_argument(format!(
                        "Invalid scheme `{scheme}`."
                    )))?;
                }
            }
            None => {
                uri = format!("http://{}", uri.to_string()).parse().unwrap();
            }
        };
        let endpoint = Channel::builder(uri)
            .timeout(self.timeout)
            .connect_timeout(self.connection_timeout)
            .keep_alive_while_idle(self.keep_alive_while_idle)
            .keep_alive_timeout(self.keep_alive_timeout);

        let endpoint = if tls {
            endpoint
                .tls_config(ClientTlsConfig::new())
                .map_err(|e| Status::internal(format!("Failed to create TLS config: {}", e)))?
        } else {
            endpoint
        };

        let channel = endpoint
            .connect()
            .await
            .map_err(|e| Status::internal(format!("Failed to connect to {}: {}", self.uri, e)))?;
        let mut self_channel = self.channel.write().unwrap();

        *self_channel = Some(channel.clone());

        Ok(channel)
    }

    pub async fn get_channel(&self) -> Result<Channel> {
        if let Some(channel) = &*self.channel.read()? {
            return Ok(channel.clone());
        }

        let channel = self.make_channel().await?;
        Ok(channel)
    }

    pub async fn drop_channel(&self) -> Result<()> {
        let mut channel = self.channel.write()?;
        *channel = None;
        Ok(())
    }

    // Allow to retry request if channel is broken
    pub async fn with_channel<T, O: Future<Output = Result<T>>>(
        &self,
        f: impl Fn(Channel) -> O,
        allow_retry: bool,
    ) -> Result<T> {
        let channel = self.get_channel().await?;

        let result: Result<T> = f(channel).await;

        // Reconnect on failure to handle the case with domain name change.
        match result {
            Ok(res) => Ok(res),
            Err(err) => match err {
                Error::ResponseError { ref status } => match status.code() {
                    Code::Internal | Code::Unavailable | Code::Cancelled | Code::Unknown => {
                        self.drop_channel().await?;
                        if allow_retry {
                            let channel = self.get_channel().await?;
                            Ok(f(channel).await?)
                        } else {
                            Err(err)
                        }
                    }
                    _ => Err(err),
                },
                _ => Err(err),
            },
        }
    }
}

impl From<InferenceServerClientConfig> for ChannelPool {
    fn from(value: InferenceServerClientConfig) -> Self {
        Self::new(
            value.uri,
            value.tls,
            value.timeout,
            value.connect_timeout,
            value.keep_alive_while_idle,
            value.keep_alive_timeout,
        )
    }
}
