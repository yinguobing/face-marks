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
    var facebox: CGRect!
    
    /// Vision and CoreML
    // TODO: return facebox
    func updateFacebox(from ciImage: CIImage) {
        let orientation = CGImagePropertyOrientation(rawValue: 1)
        DispatchQueue.global(qos: .userInitiated).async {
            let handler = VNImageRequestHandler(ciImage: ciImage, orientation: orientation!)
            do {
                try handler.perform([self.faceRequest])
            } catch {
                print("Failed to perform face detection.\n\(error.localizedDescription)")
            }
        }
    }
    
    // Make a face detection request object.
    lazy var faceRequest = VNDetectFaceRectanglesRequest(completionHandler: self.processFacebox)
    
    func processFacebox(for request: VNRequest, error: Error?) {
        guard let observations = request.results as? [VNFaceObservation]
            else { fatalError("unexpected result type from VNCoreMLRequest") }
        guard let face = observations.first
            else {return}
        let w = face.boundingBox.size.width
        let h = face.boundingBox.size.height
        let x = face.boundingBox.origin.x
        let y = face.boundingBox.origin.y
        self.facebox = CGRect(x: x, y: y, width: w, height: h)
    }
    
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
//            request.imageCropAndScaleOption = .scaleFill
            return request
        } catch {
            fatalError("Failed to load Vision ML model: \(error)")
        }
    }()
    
    /// - Tag: Perform landmark requests
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
    
    /// Updates the marks with the results of the detection.
    /// - Tag: processLandmarks
    func processLandmarks(for request: VNRequest, error: Error?) {
        DispatchQueue.main.async {
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
        facebox = CGRect(x: 0, y: 0, width: 360, height: 360)
        
        
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
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return nil}
        let ciImage = CIImage(cvPixelBuffer: imageBuffer)
        
        // TODO: Try to get face box
        /*
        updateFacebox(from: ciImage)
        let box = CGRect(x: self.facebox.origin.x * ciImage.extent.width,
                         y: (self.facebox.origin.y) * ciImage.extent.height,
                         width: self.facebox.width  * ciImage.extent.width,
                         height: self.facebox.height * ciImage.extent.height)
        */
        let shiftX = 30
        let shiftY = 90
        let width = 300
        let height = 300
        let box = CGRect(x: shiftX, y: shiftY, width: width, height: height)
        let imgCropped = ciImage.cropped(to: box)
        
        updateLandmarks(for: imgCropped)
        
        let targetSize = ciImage.extent.size
        
        // Setup a CGContext for drawing.
        UIGraphicsBeginImageContextWithOptions(targetSize, true, 0)
        let context = UIGraphicsGetCurrentContext()!
        context.setLineWidth(3.0)
        context.setStrokeColor(UIColor.white.cgColor)
        context.setFillColor(UIColor.white.cgColor)

        // Draw image
        let uiImage = UIImage(ciImage: ciImage)
        uiImage.draw(in: CGRect(x: 0, y: 0, width: targetSize.width, height: targetSize.height))
        
        // Draw marks
        for idx in 0...67 {
            let x = CGFloat(truncating: self.marks[idx * 2]) * imgCropped.extent.width + CGFloat(shiftX)
            let y = CGFloat(truncating: self.marks[idx * 2 + 1]) * imgCropped.extent.width + CGFloat(shiftY)
            context.fillEllipse(in: CGRect(x: x, y: y, width: 3, height: 3))
        }
        
        // Draw facebox
        context.addRect(box)
        context.setStrokeColor(UIColor.green.cgColor)
        context.strokePath()
        
        // Generate image.
        let resultImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return (resultImage)
    }
    
    // MARK: AVCaptureVideoDataOutputSampleBufferDelegate
    func captureOutput(_ captureOutput: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let uiImage = imageFromSampleBuffer(sampleBuffer: sampleBuffer) else { return }
        DispatchQueue.main.async { [unowned self] in
            self.delegate?.captured(image: uiImage)
        }
    }
}
