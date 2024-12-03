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
    
    cameraSelect.innerHTML = cameras.length === 0 
        ? '<option value="">No cameras found</option>'
        : cameras.map(camera => 
            `<option value="${camera.deviceId}">${camera.label || `Camera ${camera.deviceId.slice(0, 8)}`}</option>`
        ).join('');
    
    cameraSelect.addEventListener('change', () => switchCamera(cameraSelect.value));
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
                width: 640,
                height: 480
            }
        };
        
        currentStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = currentStream;
        updateStatus('Camera switched successfully');
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

function setupCanvas() {
    canvas = document.getElementById('canvas');
    canvas.width = 640;
    canvas.height = 480;
    ctx = canvas.getContext('2d');
    // Mirror the canvas to match video
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
}

async function setupFaceDetector() {
    try {
        updateStatus('Loading face detection model...');
        const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
        const detectorConfig = {
            runtime: 'tfjs',
            maxFaces: 1,
            refineLandmarks: true,
        };
        detector = await faceLandmarksDetection.createDetector(model, detectorConfig);
        updateStatus('Face detection ready');
        console.log('Face detector loaded successfully');
    } catch (error) {
        console.error('Error loading face detector:', error);
        updateStatus('Error: Could not load face detector');
        throw error;
    }
}

function drawFaceMesh(predictions) {
    if (!ctx) return;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (predictions.length > 0) {
        predictions.forEach(prediction => {
            const keypoints = prediction.keypoints;
            
            // Draw dots for each landmark
            ctx.fillStyle = '#00FF00';
            keypoints.forEach(keypoint => {
                ctx.beginPath();
                ctx.arc(keypoint.x, keypoint.y, 2, 0, 2 * Math.PI);
                ctx.fill();
            });
            
            // Draw face contour
            ctx.strokeStyle = '#00FF00';
            ctx.lineWidth = 1;
            
            // Draw facial features (eyes, nose, mouth)
            const features = [
                keypoints.slice(33, 46),    // Left eye
                keypoints.slice(362, 375),  // Right eye
                keypoints.slice(0, 17),     // Face contour
            ];
            
            features.forEach(feature => {
                ctx.beginPath();
                ctx.moveTo(feature[0].x, feature[0].y);
                feature.forEach((point, i) => {
                    if (i > 0) ctx.lineTo(point.x, point.y);
                });
                if (feature.length > 2) ctx.closePath();
                ctx.stroke();
            });
        });
        updateStatus('Tracking face...');
    } else {
        updateStatus('No face detected');
    }
}

async function detectFaces() {
    if (!detector || !video || !isTracking) return;
    
    try {
        const predictions = await detector.estimateFaces(video);
        drawFaceMesh(predictions);
    } catch (error) {
        console.error('Error during face detection:', error);
        updateStatus('Error during face detection');
    }
    
    if (isTracking) {
        requestAnimationFrame(detectFaces);
    }
}

async function startTracking() {
    if (!isTracking) {
        isTracking = true;
        detectFaces();
    }
}

async function init() {
    try {
        updateStatus('Initializing...');
        await tf.setBackend('webgl');
        console.log('TensorFlow.js backend initialized:', tf.getBackend());
        
        await setupCamera();
        console.log('Camera setup complete');
        
        setupCanvas();
        console.log('Canvas setup complete');
        
        await setupFaceDetector();
        console.log('Face detector setup complete');
        
        startTracking();
    } catch (error) {
        console.error('Initialization error:', error);
        updateStatus('Error: ' + error.message);
    }
}

// Start the application when the page loads
window.addEventListener('load', init);
