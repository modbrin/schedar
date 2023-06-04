use thiserror::Error;

#[derive(Error, Debug)]
pub enum EngineError {
    #[error("Error while loading asset ({file_path}): {message}")]
    AssetError { file_path: String, message: String },
}

impl EngineError {
    pub fn asset(file_path: &str, message: &str) -> Self {
        Self::AssetError {
            file_path: file_path.to_string(),
            message: message.to_string(),
        }
    }
}
