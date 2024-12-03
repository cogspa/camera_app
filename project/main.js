import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';

let detector = null;
let video = null;
let canvas = null;
let ctx = null;
let isTracking = false;
let currentStream = null;

function updateStatus(message) {
    const status = document.getElementById('status');
    status.textContent = message;
}

async function getConnectedCameras() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        return devices.filter(device => device.kind === 'videoinput');
    } catch (error) {
        console.error('Error getting cameras:', error);
        updateStatus('Error: Could not get camera list');
        return [];
    }
}

async function populateCameraSelect() {
    const cameraSelect = document.getElementById('cameraSelect');
    const cameras = await getConnectedCameras();
    
    // Clear existing options
    cameraSelect.innerHTML = '';
    
    if (cameras.length === 0) {
        cameraSelect.innerHTML = '<option value="">No cameras found</option>';
        return;
    }

    // Add cameras to select
    cameras.forEach(camera => {
        const option = document.createElement('option');
        option.value = camera.deviceId;
        option.text = camera.label || `Camera ${camera.deviceId.slice(0, 8)}`;
        cameraSelect.appendChild(option);
        
        // Auto-select Logitech C615 if found
        if (option.text.includes('C615')) {
            option.selected = true;
            // Trigger camera switch
            switchCamera(camera.deviceId);
        }
    });

    // Add change event listener
    cameraSelect.addEventListener('change', (e) => {
        const selectedDeviceId = e.target.value;
        if (selectedDeviceId) {
            switchCamera(selectedDeviceId);
        }
    });
}

async function switchCamera(deviceId) {
    try {
        // Stop current stream if it exists
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
        }
        
        updateStatus('Switching camera...');
        const constraints = {
            video: {
                deviceId: deviceId ? { exact: deviceId } : undefined,
                width: { ideal: 1280 },
                height: { ideal: 720 },
                frameRate: { ideal: 30 }
            }
        };
        
        currentStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = currentStream;
        updateStatus('Camera switched successfully');
        
        // Ensure video element is properly configured
        video.onloadedmetadata = () => {
            video.play();
        };
    } catch (error) {
        console.error('Error switching camera:', error);
        updateStatus('Error: Could not switch camera');
    }
}

async function setupCamera() {
    video = document.getElementById('video');
    
    try {
        updateStatus('Requesting camera access...');
        await populateCameraSelect();
        
        // Use first available camera
        const cameras = await getConnectedCameras();
        if (cameras.length > 0) {
            await switchCamera(cameras[0].deviceId);
        } else {
            throw new Error('No cameras found');
        }
        
        updateStatus('Camera access granted');
    } catch (error) {
        console.error('Error accessing camera:', error);
        updateStatus('Error: Could not access camera');
        throw error;
    }
    
    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            video.play();
            resolve(video);
        };
    });
}

async function setupCanvas() {
    try {
        // Match canvas size to video
        canvas.width = video.width;
        canvas.height = video.height;
        
        // Set canvas on top of video
        canvas.style.position = 'absolute';
        canvas.style.left = '0';
        canvas.style.top = '0';
        
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1); // Mirror the context horizontally
        
        updateStatus('Canvas setup complete');
    } catch (error) {
        console.error('Error setting up canvas:', error);
        updateStatus('Error: Canvas setup failed');
    }
}

async function setupFaceDetector() {
    try {
        updateStatus('Loading face detection model...');
        const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
        const detectorConfig = {
            runtime: 'tfjs',
            maxFaces: 1,
            refineLandmarks: false,
            shouldLoadIrisModel: false,
            enableFaceGeometry: false
        };
        
        console.log('Creating face detector with config:', detectorConfig);
        detector = await faceLandmarksDetection.createDetector(model, detectorConfig);
        console.log('Face detector created successfully');
        updateStatus('Face detection ready');
        
        // Start tracking immediately after setup
        startTracking();
    } catch (error) {
        console.error('Error loading face detector:', error);
        updateStatus('Error: Could not load face detector - ' + error.message);
        throw error;
    }
}

async function detectFaces() {
    if (!detector || !video || !isTracking) return;
    
    try {
        // Make sure video is playing and ready
        if (video.readyState !== 4) {
            requestAnimationFrame(detectFaces);
            return;
        }

        // Get the predictions
        const predictions = await detector.estimateFaces(video, {
            flipHorizontal: false,
            staticImageMode: false,
            predictIrises: false
        });
        
        // Clear previous drawings
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Save the current transformation matrix
        ctx.save();
        
        // Draw the face mesh
        if (predictions.length > 0) {
            predictions.forEach(prediction => {
                const keypoints = prediction.keypoints;
                
                // Draw points
                ctx.fillStyle = '#00FF00';
                ctx.strokeStyle = '#00FF00';
                ctx.lineWidth = 1;
                
                keypoints.forEach(point => {
                    ctx.beginPath();
                    ctx.arc(point.x, point.y, 2, 0, 2 * Math.PI);
                    ctx.fill();
                });
                
                // Draw face contours
                const contours = [
                    keypoints.slice(0, 17),    // Jaw
                    keypoints.slice(17, 22),   // Left eyebrow
                    keypoints.slice(22, 27),   // Right eyebrow
                    keypoints.slice(36, 42),   // Left eye
                    keypoints.slice(42, 48),   // Right eye
                    keypoints.slice(48, 60)    // Mouth outer
                ];
                
                contours.forEach(contour => {
                    ctx.beginPath();
                    contour.forEach((point, i) => {
                        if (i === 0) {
                            ctx.moveTo(point.x, point.y);
                        } else {
                            ctx.lineTo(point.x, point.y);
                        }
                    });
                    if (contour === contours[0]) { // Close the jaw line
                        ctx.closePath();
                    }
                    ctx.stroke();
                });
            });
            updateStatus('Face detected');
        } else {
            updateStatus('No face detected - try adjusting your position');
        }
        
        // Restore the transformation matrix
        ctx.restore();
        
        if (isTracking) {
            requestAnimationFrame(detectFaces);
        }
    } catch (error) {
        console.error('Error during face detection:', error);
        updateStatus('Error during face detection: ' + error.message);
        // Try to recover
        if (isTracking) {
            setTimeout(detectFaces, 1000);
        }
    }
}

function startTracking() {
    if (!isTracking) {
        isTracking = true;
        console.log('Starting face tracking');
        try {
            detectFaces();
        } catch (error) {
            console.error('Error starting tracking:', error);
            updateStatus('Error starting tracking: ' + error.message);
        }
    }
}

async function init() {
    try {
        updateStatus('Initializing...');
        await tf.setBackend('webgl');
        console.log('TensorFlow.js backend initialized:', tf.getBackend());
        
        await setupCamera();
        console.log('Camera setup complete');
        
        canvas = document.getElementById('canvas');
        ctx = canvas.getContext('2d');
        const WIDTH = 640;
        const HEIGHT = 480;
        video.width = WIDTH;
        video.height = HEIGHT;
        canvas.width = WIDTH;
        canvas.height = HEIGHT;
        
        await setupCanvas();
        console.log('Canvas setup complete');
        
        await setupFaceDetector();
        console.log('Face detector setup complete');
    } catch (error) {
        console.error('Initialization error:', error);
        updateStatus('Error: ' + error.message);
    }
}

// Start the application when the page loads
window.addEventListener('load', init);
