//
//  LlamaContext.swift
//  OnDeviceAI
//
//  Created by Aryan Singh on 08.10.26.
//

import SwiftUI
import llama

struct ContentView: View {
    @State private var llamaContext: LlamaContext?
    @State private var status = "Model not loaded"
    @State private var response = ""

    var body: some View {
        VStack(spacing: 20) {
            Text("OnDeviceAI")
                .font(.largeTitle)
                .bold()

            Text(status)

            Button("Load Model") {
                loadModel()
            }

            Button("Ask Llama") {
                generate()
            }
            .disabled(llamaContext == nil)

            ScrollView {
                Text(response)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding()
    }

    private func loadModel() {
        guard let path = Bundle.main.path(
            forResource: "qwen2.5-0.5b-instruct-q4_k_m",
            ofType: "gguf"
        ) else {
            status = "Model not found"
            return
        }

        status = "Loading..."

        Task {
            do {
                llamaContext = try LlamaContext.create(path: path)
                status = "Model loaded successfully!"
            } catch {
                status = "Failed: \(error)"
            }
        }
    }

    private func generate() {
        guard let llamaContext else { return }

        Task {
            let result = await llamaContext.generate(
                prompt: "What is quantization?"
            )

            response = result
        }
    }
}

#Preview {
    ContentView()
}
