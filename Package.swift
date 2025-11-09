// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "SweetLlama",
    platforms: [
        .macOS(.v12),
        .macCatalyst(.v15),
        .iOS(.v15),
    ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "SweetLlama",
            targets: ["SweetLlama"]),
    ],
    targets: [
        .target(
            name: "SweetLlama",
            dependencies: [
                "LlamaFramework"
            ]),
        .binaryTarget(
            name: "LlamaFramework",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b6862/llama-b6862-xcframework.zip",
            checksum: "d89290f1a495ba53bb483363529558896af41db35620a5d3542bbf6af6a163fd"
        )
    ]
)
