use thiserror::Error as ThisError;

#[derive(ThisError, Debug)]
pub enum Error {
    #[error("Error in the response: {}", .status.message())]
    ResponseError { status: tonic::Status },

    #[error("Invalid Uri: {}", .0)]
    InvalidUri(String),

    #[error("Resource was poisoned: {}", .0)]
    ResourcePoisoned(String),

    #[error("Error in conversion: {}", .0)]
    ConversionError(String),
}

impl From<tonic::Status> for Error {
    fn from(status: tonic::Status) -> Self {
        Self::ResponseError { status }
    }
}

impl<T> From<std::sync::PoisonError<T>> for Error {
    fn from(error: std::sync::PoisonError<T>) -> Self {
        Self::ResourcePoisoned(error.to_string())
    }
}

impl From<ndarray::ShapeError> for Error {
    fn from(error: ndarray::ShapeError) -> Self {
        Self::ConversionError(error.to_string())
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
