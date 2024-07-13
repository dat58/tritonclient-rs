use std::fmt::{Display, Formatter};

#[derive(Eq, PartialEq, Clone, Debug)]
pub enum TritonDataTypes {
    BOOL,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    FP16,
    BF16,
    FP32,
    FP64,
    BYTES,
}

impl Display for TritonDataTypes {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BOOL => write!(f, "BOOL"),
            Self::BYTES => write!(f, "BYTES"),
            Self::INT8 => write!(f, "INT8"),
            Self::INT16 => write!(f, "INT16"),
            Self::INT32 => write!(f, "INT32"),
            Self::INT64 => write!(f, "INT64"),
            Self::UINT8 => write!(f, "UINT8"),
            Self::UINT16 => write!(f, "UINT16"),
            Self::UINT32 => write!(f, "UINT32"),
            Self::UINT64 => write!(f, "UINT64"),
            Self::FP16 => write!(f, "FP16"),
            Self::BF16 => write!(f, "BF16"),
            Self::FP32 => write!(f, "FP32"),
            Self::FP64 => write!(f, "FP64"),
        }
    }
}

impl<T> From<T> for TritonDataTypes
where
    T: AsRef<str>,
{
    fn from(value: T) -> Self {
        match value.as_ref() {
            "BOOL" => Self::BOOL,
            "BYTES" => Self::BYTES,
            "INT8" => Self::INT8,
            "INT16" => Self::INT16,
            "INT32" => Self::INT32,
            "INT64" => Self::INT64,
            "UINT8" => Self::UINT8,
            "UINT16" => Self::UINT16,
            "UINT32" => Self::UINT32,
            "UINT64" => Self::UINT64,
            "FP16" => Self::FP16,
            "BF16" => Self::BF16,
            "FP32" => Self::FP32,
            "FP64" => Self::FP64,
            _ => panic!("Invalid data type!"),
        }
    }
}

pub type Bytes = Vec<u8>;
