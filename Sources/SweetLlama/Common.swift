import Foundation
import llama

public struct CommonParams {
    public enum FlashAttentionType: Int32, RawRepresentable{
        case auto = -1
        case disabled = 0
        case enabled = 1
    }
    
    public var nPredict: Int = -1
    public var nCTX: Int = 4096
    public var nBatch: Int = 512
    public var nUBatch: Int = 512
    public var nKeep: Int = 0
    public var nGpuLayers: Int = -1
    public var nThreads: Int = -1
    public var sparams: SamplerParams = .init()

    public var antiprompt: [String] = []
    public var flashAttnType: FlashAttentionType = .auto
    public var noPerf: Bool = false
    public var useMmap: Bool = true
    public var useMlock: Bool = false

    public var seed: UInt32

    public init(seed: UInt32 = 42) {
        self.seed = seed
    }
}

public struct CommonInitResult {
    public let model: OpaquePointer?
    public let ctx: OpaquePointer?

    public init() {
        model = nil
        ctx = nil
    }

    public init(_ model: OpaquePointer, _ ctx: OpaquePointer) {
        self.model = model
        self.ctx = ctx
    }
}

public struct LlamaCommon {
    public static func initFrom(_ modelPath: String, _ params: CommonParams)
        -> CommonInitResult
    {
        let mparams = modelParamsFrom(params)
        let model = llama_model_load_from_file(modelPath, mparams)
        guard let model = model else {
            print("Failed to load model")
            return .init()
        }

        let cparams = contextParamsFrom(params)
        let ctx = llama_init_from_model(model, cparams)
        guard let ctx = ctx else {
            print("Failed to create context")
            llama_model_free(model)
            return .init()
        }

        return .init(model, ctx)
    }

    public static func modelParamsFrom(_ params: CommonParams)
        -> llama_model_params
    {
        var mparams = llama_model_default_params()

        if params.nGpuLayers != -1 {
            mparams.n_gpu_layers = Int32(params.nGpuLayers)
#if targetEnvironment(simulator)
            mparams.n_gpu_layers = 0
#endif
        }
        mparams.use_mmap = params.useMmap
        mparams.use_mlock = params.useMlock
        
        return mparams
    }

    public static func contextParamsFrom(_ params: CommonParams)
        -> llama_context_params
    {
        var cparams = llama_context_default_params()

        cparams.n_ctx = UInt32(params.nCTX)
        cparams.n_batch = UInt32(params.nBatch)
        cparams.n_ubatch = UInt32(params.nUBatch)
        if params.nThreads == -1 {
            cparams.n_threads = Int32(ProcessInfo.processInfo.processorCount - 2)
            cparams.n_threads_batch = cparams.n_threads
        } else {
            cparams.n_threads = Int32(params.nThreads)
            cparams.n_threads_batch = Int32(params.nThreads)
        }
        cparams.flash_attn_type = llama_flash_attn_type(params.flashAttnType.rawValue)
        cparams.no_perf = params.noPerf

        return cparams
    }

    public static func samplerInit(
        _ model: OpaquePointer, _ params: SamplerParams, seed: UInt32
    ) -> UnsafeMutablePointer<llama_sampler>? {

        var lparams = llama_sampler_chain_default_params()
        lparams.no_perf = false

        guard let chain = llama_sampler_chain_init(lparams) else {
            print("failed to init sampler chain")
            return nil
        }

        llama_sampler_chain_add(
            chain,
            llama_sampler_init_penalties(
                Int32(params.penaltyLastN),
                params.penaltyRepeat,
                params.penaltyFrequency,
                params.penaltyPresent))

        if params.mirostat == .none {
            do {
                let defaultDryBreakers = ["\n", ":", "\"", "*"]
                var breakers: [UnsafePointer<CChar>?] = []
                for str in defaultDryBreakers {
                    str.withCString { cStr in
                        breakers.append(cStr)
                    }
                }
                breakers.withUnsafeMutableBufferPointer { buffer in
                    llama_sampler_chain_add(
                        chain,
                        llama_sampler_init_dry(
                            model,
                            llama_model_n_ctx_train(model),
                            params.dryMultiplier,
                            params.dryBase,
                            Int32(params.dryAllowedLength),
                            Int32(params.dryPenaltyLastN),
                            buffer.baseAddress,
                            defaultDryBreakers.count))
                }
            }

            llama_sampler_chain_add(
                chain,
                llama_sampler_init_top_k(Int32(params.topK)))

            llama_sampler_chain_add(
                chain,
                llama_sampler_init_top_p(params.topP, params.minKeep))

            llama_sampler_chain_add(
                chain,
                llama_sampler_init_min_p(params.minP, params.minKeep))

            llama_sampler_chain_add(
                chain,
                llama_sampler_init_xtc(
                    params.xtcProbability, params.xtcThreshold, params.minKeep,
                    seed))

            llama_sampler_chain_add(
                chain,
                llama_sampler_init_typical(params.typicalP, params.minKeep))

            llama_sampler_chain_add(
                chain,
                llama_sampler_init_temp(params.temperature))

            llama_sampler_chain_add(
                chain,
                llama_sampler_init_dist(seed))
        } else if params.mirostat == .v1 {
            llama_sampler_chain_add(
                chain,
                llama_sampler_init_temp(params.temperature))
            llama_sampler_chain_add(
                chain,
                llama_sampler_init_mirostat(
                    llama_vocab_n_tokens(model),
                    seed,
                    params.mirostatTau,
                    params.mirostatEta,
                    100))
        } else {
            llama_sampler_chain_add(
                chain,
                llama_sampler_init_temp(params.temperature))
            llama_sampler_chain_add(
                chain,
                llama_sampler_init_mirostat_v2(
                    seed,
                    params.mirostatTau,
                    params.mirostatEta))
        }

        return chain
    }

    public static func tokenToPiece(
        _ ctx: OpaquePointer, _ token: llama_token, _ special: Bool
    ) -> [CChar] {
        var piece: [CChar] = Array(repeating: 0, count: 16)
        var nChars: Int = 0
        piece.withUnsafeMutableBytes { buffer in
            nChars = Int(
                llama_token_to_piece(
                    llama_model_get_vocab(llama_get_model(ctx)), token, buffer.baseAddress,
                    Int32(buffer.count), 0, special))
        }
        if nChars < 0 {
            piece = Array(repeating: 0, count: -nChars)
            var check: Int = 0
            piece.withUnsafeMutableBytes { buffer in
                check = Int(
                    llama_token_to_piece(
                        llama_model_get_vocab(llama_get_model(ctx)), token, buffer.baseAddress,
                        Int32(buffer.count), 0, special))
            }
            guard check == -nChars else {
                return []
            }
        } else {
            piece.removeSubrange(nChars..<piece.count)
        }

        return piece
    }

    public static func tokenize(
        _ model: OpaquePointer, _ text: String, _ addSpecial: Bool,
        _ parseSpecial: Bool
    ) -> [llama_token] {
        let utf8Count = text.utf8.count
        let n_tokens = utf8Count + (addSpecial ? 1 : 0) + 1

        return Array(unsafeUninitializedCapacity: n_tokens) {
            buffer, initializedCount in
            initializedCount = Int(
                llama_tokenize(
                    llama_model_get_vocab(model), text, Int32(utf8Count), buffer.baseAddress,
                    Int32(n_tokens), addSpecial, parseSpecial)
            )
        }
    }
}
