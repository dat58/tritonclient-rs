use super::{Error, Result};
use std::str::FromStr;
use std::time::Duration;
use tonic::transport::Uri;

#[derive(Clone, Debug)]
pub struct InferenceServerClientConfig {
    /// Triton server URI to connect to
    pub uri: Uri,

    /// Timeout for API requests
    pub timeout: Duration,

    /// Secure connection
    pub tls: bool,

    /// Timeout for connecting to the Qdrant server
    pub connect_timeout: Duration,

    /// Whether to keep idle connections active
    pub keep_alive_while_idle: bool,

    /// Duration an idle connection remains open before closing
    pub keep_alive_timeout: Duration,

    /// Optional compression schema to use for API requests
    pub compression: Option<CompressionEncoding>,
}

impl InferenceServerClientConfig {
    pub fn from_uri<S: AsRef<str>>(uri: S) -> Result<Self> {
        Ok(Self {
            uri: Uri::from_str(uri.as_ref()).map_err(|e| Error::InvalidUri(e.to_string()))?,
            ..Self::default()
        })
    }

    pub fn timeout<T: AsTimeout>(mut self, timeout: T) -> Self {
        self.timeout = AsTimeout::timeout(timeout);
        self
    }

    pub fn tls(mut self, tls: bool) -> Self {
        self.tls = tls;
        self
    }

    pub fn connection_timeout<T: AsTimeout>(mut self, timeout: T) -> Self {
        self.connect_timeout = AsTimeout::timeout(timeout);
        self
    }

    pub fn keep_alive_while_idle(mut self, keep_alive_while_idle: bool) -> Self {
        self.keep_alive_while_idle = keep_alive_while_idle;
        self
    }

    pub fn keep_alive_timeout<T: AsTimeout>(mut self, keep_alive_timeout: T) -> Self {
        self.keep_alive_timeout = keep_alive_timeout.timeout();
        self
    }

    pub fn compression(mut self, compression: Option<CompressionEncoding>) -> Self {
        self.compression = compression;
        self
    }

    pub fn set_timeout<T: AsTimeout>(&mut self, timeout: T) {
        self.timeout = AsTimeout::timeout(timeout);
    }

    pub fn set_tls(&mut self, tls: bool) {
        self.tls = tls;
    }

    pub fn set_connection_timeout<T: AsTimeout>(&mut self, timeout: T) {
        self.connect_timeout = AsTimeout::timeout(timeout);
    }

    pub fn set_keep_alive_while_idle(&mut self, keep_alive_while_idle: bool) {
        self.keep_alive_while_idle = keep_alive_while_idle;
    }

    pub fn set_keep_alive_timeout<T: AsTimeout>(&mut self, keep_alive_timeout: T) {
        self.keep_alive_timeout = keep_alive_timeout.timeout();
    }

    pub fn set_compression(&mut self, compression: Option<CompressionEncoding>) {
        self.compression = compression;
    }
}

impl Default for InferenceServerClientConfig {
    fn default() -> Self {
        Self {
            uri: Uri::from_str("localhost:8001").unwrap(),
            timeout: Duration::from_secs(30),
            tls: false,
            connect_timeout: Duration::from_secs(5),
            keep_alive_while_idle: true,
            keep_alive_timeout: Duration::from_secs(20),
            compression: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionEncoding {
    Gzip,
    Zstd,
}

impl From<CompressionEncoding> for tonic::codec::CompressionEncoding {
    fn from(encoding: CompressionEncoding) -> Self {
        match encoding {
            CompressionEncoding::Gzip => tonic::codec::CompressionEncoding::Gzip,
            CompressionEncoding::Zstd => tonic::codec::CompressionEncoding::Zstd,
        }
    }
}

pub trait AsTimeout {
    fn timeout(self) -> Duration;
}

impl AsTimeout for Duration {
    fn timeout(self) -> Duration {
        self
    }
}

impl AsTimeout for u64 {
    fn timeout(self) -> Duration {
        Duration::from_secs(self)
    }
}
