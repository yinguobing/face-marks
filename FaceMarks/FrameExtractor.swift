//
//  FrameExtractor.swift
//  Created by yinguobing on 2018/1/4.
//

import UIKit
import AVFoundation
import CoreML
import Vision

protocol FrameExtractorDelegate: class {
    func captured(image: UIImage)
}

class FrameExtractor: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    private var position = AVCaptureDevice.Position.front
    private let quality = AVCaptureSession.Preset.medium
    
    private var permissionGranted = false
    private let sessionQueue = DispatchQueue(label: "session queue")
    private let captureSession = AVCaptureSession()
    private let context = CIContext()
        
    weak var delegate: FrameExtractorDelegate?
    
    var marks: MLMultiArray!
    
    /// Vision and CoreML
    // - Tag: MLModelSetup, make a request object.
    lazy var landmarkRequest: VNCoreMLRequest = {
        do {
            /*
             Use the Swift class `landmark` Core ML generates from the model.
             */
            let model = try VNCoreMLModel(for: landmark().model)
            
            let request = VNCoreMLRequest(model: model, completionHandler: { [weak self] request, error in
                self?.processLandmarks(for: request, error: error)
            })
            request.imageCropAndScaleOption = .centerCrop
            return request
        } catch {
            fatalError("Failed to load Vision ML model: \(error)")
        }
    }()
    
    /// - Tag: PerformRequests
    func updateLandmarks(for ciImage: CIImage) {
        
        let orientation = CGImagePropertyOrientation(rawValue: 1)
        
        DispatchQueue.global(qos: .userInitiated).async {
            let handler = VNImageRequestHandler(ciImage: ciImage, orientation: orientation!)
            do {
                try handler.perform([self.landmarkRequest])
            } catch {
                print("Failed to perform landmarks.\n\(error.localizedDescription)")
            }
        }
    }
    
    /// Updates the UI with the results of the classification.
    /// - Tag: ProcessClassifications
    func processLandmarks(for request: VNRequest, error: Error?) {
        DispatchQueue.main.async {
            /*
            guard let results = request.results else {
                fatalError("Unable to process image.")
            }
            
            // The `results` will always be `VNCoreMLFeatureValueObservation`s, as specified by the Core ML model in this project.
            let observations = results as! [VNCoreMLFeatureValueObservation]
            
            if observations.isEmpty {
                print("Nothing recognized.")
            } else {
                // MARKS GOT.
                guard let faceMarks = observations.first?.featureValue.multiArrayValue
                    else { fatalError("can't get best result") }
                self.marks = faceMarks
            }
            */
            guard let observations = request.results as? [VNCoreMLFeatureValueObservation]
                else { fatalError("unexpected result type from VNCoreMLRequest") }
            guard let faceMarks = observations.first?.featureValue.multiArrayValue
                else { fatalError("can't get best result") }
            self.marks = faceMarks
        }
    }
    
    override init() {
        super.init()
        checkPermission()
        sessionQueue.async { [unowned self] in
            self.configureSession()
            self.captureSession.startRunning()
        }
        do {
            try marks = MLMultiArray(shape: [136], dataType: MLMultiArrayDataType.double)
        } catch {
            print("MulitiArray initialization failed.")
        }
        
        
    }
    
    public func flipCamera() {
        sessionQueue.async { [unowned self] in
            self.captureSession.beginConfiguration()
            guard let currentCaptureInput = self.captureSession.inputs.first else { return }
            self.captureSession.removeInput(currentCaptureInput)
            guard let currentCaptureOutput = self.captureSession.outputs.first else { return }
            self.captureSession.removeOutput(currentCaptureOutput)
            self.position = self.position == .front ? .back : .front
            self.configureSession()
            self.captureSession.commitConfiguration()
        }
    }
    
    // MARK: AVSession configuration
    private func checkPermission() {
        switch AVCaptureDevice.authorizationStatus(for: AVMediaType.video) {
        case .authorized:
            permissionGranted = true
        case .notDetermined:
            requestPermission()
        default:
            permissionGranted = false
        }
    }
    
    private func requestPermission() {
        sessionQueue.suspend()
        AVCaptureDevice.requestAccess(for: AVMediaType.video) { [unowned self] granted in
            self.permissionGranted = granted
            self.sessionQueue.resume()
        }
    }
    
    private func configureSession() {
        guard permissionGranted else { return }
        captureSession.sessionPreset = quality
        guard let captureDevice = selectCaptureDevice() else { return }
        guard let captureDeviceInput = try? AVCaptureDeviceInput(device: captureDevice) else { return }
        guard captureSession.canAddInput(captureDeviceInput) else { return }
        captureSession.addInput(captureDeviceInput)
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "sample buffer"))
        guard captureSession.canAddOutput(videoOutput) else { return }
        captureSession.addOutput(videoOutput)
        guard let connection = videoOutput.connection(with: AVFoundation.AVMediaType.video) else { return }
        guard connection.isVideoOrientationSupported else { return }
        guard connection.isVideoMirroringSupported else { return }
        connection.videoOrientation = .portrait
        connection.isVideoMirrored = position == .front
    }
    
    private func selectCaptureDevice() -> AVCaptureDevice? {
        return AVCaptureDevice.devices().filter {
            ($0 as AnyObject).hasMediaType(AVMediaType.video) &&
            ($0 as AnyObject).position == position
            }.first
    }
    
    // MARK: Sample buffer to UIImage conversion
    private func imageFromSampleBuffer(sampleBuffer: CMSampleBuffer) -> UIImage? {
        // Get image from buffer.
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return nil }
        let ciImage = CIImage(cvPixelBuffer: imageBuffer)
        let imgCropped = ciImage.cropped(to: CGRect(x: 0, y: 0, width: 360, height: 360))
        
        updateLandmarks(for: imgCropped)
        
        // Try to draw the marks.
        UIGraphicsBeginImageContextWithOptions(imgCropped.extent.size, true, 0)
        let context = UIGraphicsGetCurrentContext()!
        context.setLineWidth(5.0)
        context.setStrokeColor(UIColor.green.cgColor)
        let uiImage = UIImage(ciImage: imgCropped)
        uiImage.draw(at: CGPoint(x: 0, y: 0))
        
        context.beginPath()
        for idx in 0...67 {
            let x = CGFloat(truncating: self.marks[idx * 2]) * imgCropped.extent.width
            let y = CGFloat(truncating: self.marks[idx * 2 + 1]) * imgCropped.extent.width
            context.addEllipse(in: CGRect(x: x, y: y, width: 1, height: 1))
        }
        
        context.strokePath()
        let resultImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resultImage
    }
    
    // MARK: AVCaptureVideoDataOutputSampleBufferDelegate
    func captureOutput(_ captureOutput: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let uiImage = imageFromSampleBuffer(sampleBuffer: sampleBuffer) else { return }
        DispatchQueue.main.async { [unowned self] in
            self.delegate?.captured(image: uiImage)
        }
    }
}
