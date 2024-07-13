fn main() {
    let output_dir = "src/grpc/proto";
    if !std::path::Path::new(output_dir).is_dir() {
        std::fs::create_dir(output_dir).unwrap();
    }
    tonic_build::configure()
        .out_dir(output_dir)
        .build_server(false)
        .build_client(true)
        .compile(
            &[
                "protobuf/grpc_service.proto",
                "protobuf/health.proto",
                "protobuf/model_config.proto",
            ],
            &["protobuf"],
        )
        .unwrap();
}
