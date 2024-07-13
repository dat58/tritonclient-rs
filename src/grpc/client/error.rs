use std::fmt::{Debug, Display, Formatter};

#[derive(Debug)]
pub struct Error {}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl std::error::Error for Error {}

impl From<tonic::Status> for Error {
    fn from(value: tonic::Status) -> Self {
        Self {}
    }
}

pub type Result<T> = std::result::Result<T, Error>;
