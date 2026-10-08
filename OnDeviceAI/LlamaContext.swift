//
//  LlamaContext.swift
//  OnDeviceAI
//
//  Created by Aryan Singh on 08.10.26.
//

import Foundation
import llama

enum LlamaError: Error {
    case modelLoadFailed
    case contextCreationFailed
}

func llama_batch_clear(_ batch: inout llama_batch) {
    batch.n_tokens = 0
}

func llama_batch_add(
    _ batch: inout llama_batch,
    _ id: llama_token,
    _ pos: llama_pos,
    _ seq_ids: [llama_seq_id],
    _ logits: Bool
) {
    batch.token[Int(batch.n_tokens)] = id
    batch.pos[Int(batch.n_tokens)] = pos
    batch.n_seq_id[Int(batch.n_tokens)] = Int32(seq_ids.count)

    for i in 0..<seq_ids.count {
        batch.seq_id[Int(batch.n_tokens)]![Int(i)] = seq_ids[i]
    }

    batch.logits[Int(batch.n_tokens)] = logits ? 1 : 0
    batch.n_tokens += 1
}

actor LlamaContext {
    private var model: OpaquePointer
    private var context: OpaquePointer
    
    private var vocab: OpaquePointer
    private var sampling: UnsafeMutablePointer<llama_sampler>
    private var batch: llama_batch

    private var tokensList: [llama_token] = []
    private var currentPosition: Int32 = 0
    private var maxTokens: Int32 = 128

    init(model: OpaquePointer, context: OpaquePointer) {
        self.model = model
        self.context = context
        self.vocab = llama_model_get_vocab(model)
        self.batch = llama_batch_init(512, 0, 1)

        let samplerParams = llama_sampler_chain_default_params()
        self.sampling = llama_sampler_chain_init(samplerParams)

        llama_sampler_chain_add(
            self.sampling,
            llama_sampler_init_temp(0.4)
        )

        llama_sampler_chain_add(
            self.sampling,
            llama_sampler_init_dist(1234)
        )
    }

    deinit {
        llama_sampler_free(sampling)
        llama_batch_free(batch)
        llama_model_free(model)
        llama_free(context)
        llama_backend_free()
    }

    static func create(path: String) throws -> LlamaContext {
        // Initialize llama.cpp
        llama_backend_init()

        // Model configuration
        var modelParams = llama_model_default_params()

        #if targetEnvironment(simulator)
        // GPU layers are disabled for the simulator
        modelParams.n_gpu_layers = 0
        #endif

        // Load GGUF model
        guard let model = llama_model_load_from_file(path, modelParams) else {
            llama_backend_free()
            throw LlamaError.modelLoadFailed
        }

        // Context configuration
        var contextParams = llama_context_default_params()
        contextParams.n_ctx = 2048

        let context = llama_init_from_model(model, contextParams)

        guard let context else {
            llama_model_free(model)
            llama_backend_free()
            throw LlamaError.contextCreationFailed
        }

        return LlamaContext(
            model: model,
            context: context
        )
    }
    
    private func tokenize(text: String) -> [llama_token] {
        let utf8Count = text.utf8.count
        let capacity = utf8Count + 2

        let tokens = UnsafeMutablePointer<llama_token>.allocate(
            capacity: capacity
        )

        let count = llama_tokenize(
            vocab,
            text,
            Int32(utf8Count),
            tokens,
            Int32(capacity),
            true,
            false
        )

        var result: [llama_token] = []

        if count > 0 {
            for i in 0..<Int(count) {
                result.append(tokens[i])
            }
        }

        tokens.deallocate()
        return result
    }
    
    private func tokenToString(_ token: llama_token) -> String {
        var buffer = [CChar](repeating: 0, count: 256)

        let count = llama_token_to_piece(
            vocab,
            token,
            &buffer,
            Int32(buffer.count),
            0,
            false
        )

        guard count > 0 else {
            return ""
        }

        return String(
            bytes: buffer.prefix(Int(count)).map {
                UInt8(bitPattern: $0)
            },
            encoding: .utf8
        ) ?? ""
    }
    
    func generate(prompt: String) -> String {
        let tokens = tokenize(text: prompt)

        llama_batch_clear(&batch)

        for (i, token) in tokens.enumerated() {
            llama_batch_add(
                &batch,
                token,
                Int32(i),
                [0],
                false
            )
        }

        batch.logits[Int(batch.n_tokens) - 1] = 1

        if llama_decode(context, batch) != 0 {
            return "Prompt decoding failed."
        }

        currentPosition = batch.n_tokens

        var result = ""

        for _ in 0..<maxTokens {
            let token = llama_sampler_sample(
                sampling,
                context,
                batch.n_tokens - 1
            )

            if llama_vocab_is_eog(vocab, token) {
                break
            }

            let piece = tokenToString(token)
            result += piece

            llama_batch_clear(&batch)

            llama_batch_add(
                &batch,
                token,
                currentPosition,
                [0],
                true
            )

            currentPosition += 1

            if llama_decode(context, batch) != 0 {
                break
            }
        }

        return result
    }
}
