
//  ContentView.swift
//  AgeGenderEmotionApp
//  Benchmark: Step 6.2 (20 runs)


import SwiftUI
import CoreML

struct ContentView: View {
    @State private var statusText: String = "Ready"
    @State private var selectedImage: UIImage? = nil
    @State private var showPicker: Bool = false

    // Update these to match YOUR training label order
    private let ageLabels = ["Young", "Adult", "Senior"] // 3 classes
    private let genderLabels = ["Female", "Male"]        // 2 classes
    private let emotionLabels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"] // 7 classes

    var body: some View {
        VStack(spacing: 16) {
            Text("AgeGenderEmotion")
                .font(.title2)

            Button("Pick image") {
                showPicker = true
            }

            Button("Run inference on selected image") {
                runImageInference()
            }
            .disabled(selectedImage == nil)

            // Step 6.2: Benchmark over 20 runs (report-ready)
            Button("Benchmark image inference (20 runs)") {
                benchmarkImageInference(runs: 20)
            }
            .disabled(selectedImage == nil)

            if let img = selectedImage {
                Image(uiImage: img)
                    .resizable()
                    .scaledToFit()
                    .frame(height: 220)
                    .cornerRadius(12)
            } else {
                Text("No image selected yet.")
                    .font(.footnote)
            }

            Text(statusText)
                .font(.footnote)
                .multilineTextAlignment(.center)
                .padding()
        }
        .padding()
        .sheet(isPresented: $showPicker) {
            ImagePicker(image: $selectedImage)
        }
    }

    /// Step 6.2: Benchmark inference time over N runs on the currently selected image.
    /// - Warms up 3 runs to stabilize caching/allocations.
    /// - Measures N runs and reports avg/min/max in ms.
    private func benchmarkImageInference(runs: Int) {
        guard let img = selectedImage else {
            statusText = "No image selected."
            return
        }

        do {
            statusText = "Benchmarking… (\(runs) runs)"

            let model = try AgeGenderEmotion(configuration: MLModelConfiguration())

            // IMPORTANT:
            // - If you trained with ImageNet normalization, keep normalizeImageNet=true.
            // - If you trained WITHOUT normalization, set it to false.
            let input = try uiImageToCHWTensor(img, size: 224, normalizeImageNet: true)

            // Warmup (recommended)
            for _ in 0..<3 { _ = try model.prediction(image: input) }

            var times: [Double] = []
            times.reserveCapacity(runs)

            for _ in 0..<runs {
                let start = CFAbsoluteTimeGetCurrent()
                _ = try model.prediction(image: input)
                let end = CFAbsoluteTimeGetCurrent()
                times.append((end - start) * 1000.0) // ms
            }

            let avg = times.reduce(0, +) / Double(times.count)
            let minT = times.min() ?? avg
            let maxT = times.max() ?? avg

            print("Benchmark (\(runs) runs): avg \(String(format: "%.2f", avg)) ms | min \(String(format: "%.2f", minT)) ms | max \(String(format: "%.2f", maxT)) ms")

            statusText = "avg \(String(format: "%.1f", avg)) ms (min \(String(format: "%.1f", minT)), max \(String(format: "%.1f", maxT)))"
        } catch {
            statusText = "Benchmark error: \(error.localizedDescription)"
            print("Benchmark failed:", error)
        }
    }

    // --- helpers below are required for the benchmark ---

    private func runImageInference() {
        guard let img = selectedImage else {
            statusText = "No image selected."
            return
        }

        do {
            statusText = "Running image inference…"

            let model = try AgeGenderEmotion(configuration: MLModelConfiguration())
            let input = try uiImageToCHWTensor(img, size: 224, normalizeImageNet: true)
            let output = try model.prediction(image: input)

            let ageIdx = argmax(output.var_820)
            let genderIdx = argmax(output.var_823)
            let emotionIdx = argmax(output.var_826)

            let age = ageIdx < ageLabels.count ? ageLabels[ageIdx] : "\(ageIdx)"
            let gender = genderIdx < genderLabels.count ? genderLabels[genderIdx] : "\(genderIdx)"
            let emotion = emotionIdx < emotionLabels.count ? emotionLabels[emotionIdx] : "\(emotionIdx)"

            print("Image inference successful")
            print("Age logits:", output.var_820, "->", ageIdx, age)
            print("Gender logits:", output.var_823, "->", genderIdx, gender)
            print("Emotion logits:", output.var_826, "->", emotionIdx, emotion)

            statusText = "Image: Age=\(age), Gender=\(gender), Emotion=\(emotion)"
        } catch {
            statusText = "Error: \(error.localizedDescription)"
            print("Image inference failed:", error)
        }
    }

    /// Convert a UIImage into a Float32 Core ML tensor shaped (1,3,H,W) in CHW order.
    private func uiImageToCHWTensor(_ image: UIImage, size: Int, normalizeImageNet: Bool) throws -> MLMultiArray {
        guard let cgImage = image.cgImage else {
            throw NSError(domain: "ImageConversion", code: -1, userInfo: [NSLocalizedDescriptionKey: "UIImage has no CGImage"])
        }

        let width = size
        let height = size
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8

        var rawData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)

        guard let context = CGContext(
            data: &rawData,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw NSError(domain: "ImageConversion", code: -2, userInfo: [NSLocalizedDescriptionKey: "Failed to create CGContext"])
        }

        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        let tensor = try MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)], dataType: .float32)

        let mean: [Float] = [0.485, 0.456, 0.406]
        let std:  [Float] = [0.229, 0.224, 0.225]

        for y in 0..<height {
            for x in 0..<width {
                let base = (y * width + x) * bytesPerPixel
                let r = Float(rawData[base]) / 255.0
                let g = Float(rawData[base + 1]) / 255.0
                let b = Float(rawData[base + 2]) / 255.0

                let rr: Float
                let gg: Float
                let bb: Float

                if normalizeImageNet {
                    rr = (r - mean[0]) / std[0]
                    gg = (g - mean[1]) / std[1]
                    bb = (b - mean[2]) / std[2]
                } else {
                    rr = r
                    gg = g
                    bb = b
                }

                tensor[[0, 0, y as NSNumber, x as NSNumber]] = NSNumber(value: rr)
                tensor[[0, 1, y as NSNumber, x as NSNumber]] = NSNumber(value: gg)
                tensor[[0, 2, y as NSNumber, x as NSNumber]] = NSNumber(value: bb)
            }
        }

        return tensor
    }

    /// Argmax over a 1D MLMultiArray.
    private func argmax(_ arr: MLMultiArray) -> Int {
        var bestIndex = 0
        var bestValue = Double.leastNormalMagnitude

        for i in 0..<arr.count {
            let v = arr[i].doubleValue
            if v > bestValue {
                bestValue = v
                bestIndex = i
            }
        }
        return bestIndex
    }
}

#Preview {
    ContentView()
}
