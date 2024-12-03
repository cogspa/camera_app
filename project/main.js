import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';
import * as handPoseDetection from '@tensorflow-models/hand-pose-detection';

let detector = null;
let handDetector = null;
let video = null;
let canvas = null;
let ctx = null;
let isTracking = false;
let currentStream;

// Overlay images
let faceOverlay = null;
let handOverlay = null;
let faceScale = 1.0;
let handScale = 1.0;

// Add rate limiting variables
let lastDetectionTime = 0;
const DETECTION_INTERVAL = 50; // Minimum milliseconds between detections

// Add offset variables
let faceOffsetX = 0;
let faceOffsetY = 0;
let handOffsetX = 0;
let handOffsetY = 0;

// Setup image upload handlers
function setupImageUploads() {
    const faceInput = document.getElementById('faceOverlayUpload');
    const handInput = document.getElementById('handOverlayUpload');
    const facePreview = document.getElementById('facePreview');
    const handPreview = document.getElementById('handPreview');
    const faceScaleInput = document.getElementById('faceScale');
    const handScaleInput = document.getElementById('handScale');

    faceInput.addEventListener('change', async (e) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            const img = new Image();
            img.src = URL.createObjectURL(file);
            await new Promise(resolve => img.onload = resolve);
            faceOverlay = img;
            facePreview.src = img.src;
            facePreview.style.display = 'block';
        }
    });

    handInput.addEventListener('change', async (e) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            const img = new Image();
            img.src = URL.createObjectURL(file);
            await new Promise(resolve => img.onload = resolve);
            handOverlay = img;
            handPreview.src = img.src;
            handPreview.style.display = 'block';
        }
    });

    faceScaleInput.addEventListener('input', (e) => {
        faceScale = parseFloat(e.target.value);
    });

    handScaleInput.addEventListener('input', (e) => {
        handScale = parseFloat(e.target.value);
    });
}

function drawImageOnLandmarks(img, landmarks, scale, offsetX = 0, offsetY = 0) {
    if (!img || !landmarks || landmarks.length < 2) return;

    // Calculate bounding box
    let minX = Infinity, minY = Infinity;
    let maxX = -Infinity, maxY = -Infinity;

    landmarks.forEach(point => {
        // Flip X coordinate since video is mirrored
        const x = canvas.width - point.x;
        const y = point.y;
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
    });

    const width = (maxX - minX) * scale;
    const height = (maxY - minY) * scale;
    const centerX = (minX + maxX) / 2 + offsetX;  // Apply X offset
    const centerY = (minY + maxY) / 2 + offsetY;  // Apply Y offset

    // Save current context state
    ctx.save();
    
    // Scale and draw the image
    ctx.translate(centerX, centerY);
    ctx.scale(-1, 1); // Mirror the image horizontally
    ctx.drawImage(
        img,
        -width / 2,
        -height / 2,
        width,
        height
    );
    
    // Restore context state
    ctx.restore();
}

async function detectHands() {
    if (!handDetector || !video || !isTracking) return;
    
    try {
        const hands = await handDetector.estimateHands(video, {
            flipHorizontal: false // Changed to false since we handle flipping manually
        });
        
        if (hands.length > 0) {
            hands.forEach(hand => {
                // Draw hand landmarks
                ctx.fillStyle = '#00FFFF';
                ctx.strokeStyle = '#00FFFF';
                ctx.lineWidth = 2;
                
                // Draw keypoints
                hand.keypoints.forEach(point => {
                    ctx.beginPath();
                    ctx.arc(point.x, point.y, 3, 0, 2 * Math.PI);
                    ctx.fill();
                });
                
                // Draw connections
                const fingers = [
                    [0, 1, 2, 3, 4],          // thumb
                    [0, 5, 6, 7, 8],          // index finger
                    [0, 9, 10, 11, 12],       // middle finger
                    [0, 13, 14, 15, 16],      // ring finger
                    [0, 17, 18, 19, 20]       // pinky
                ];
                
                fingers.forEach(finger => {
                    ctx.beginPath();
                    finger.forEach((pointIndex, i) => {
                        const point = hand.keypoints[pointIndex];
                        if (i === 0) {
                            ctx.moveTo(point.x, point.y);
                        } else {
                            ctx.lineTo(point.x, point.y);
                        }
                    });
                    ctx.stroke();
                });
            });
        }
    } catch (error) {
        console.error('Error during hand detection:', error);
    }
}

async function detectFaces() {
    if (!detector || !video || !isTracking) return;
    
    try {
        // Rate limiting
        const now = Date.now();
        if (now - lastDetectionTime < DETECTION_INTERVAL) {
            requestAnimationFrame(detectFaces);
            return;
        }
        lastDetectionTime = now;

        if (video.readyState !== 4) {
            requestAnimationFrame(detectFaces);
            return;
        }

        // Memory cleanup
        if (tf.memory().numTensors > 1000) {
            tf.disposeVariables();
            await tf.ready();
        }

        const predictions = await detector.estimateFaces(video, {
            flipHorizontal: false,
            staticImageMode: false,
            predictIrises: false
        });
        
        // Dispose of tensors after each detection
        tf.tidy(() => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            if (predictions.length > 0) {
                predictions.forEach(prediction => {
                    const keypoints = prediction.keypoints;
                    
                    if (faceOverlay) {
                        drawImageOnLandmarks(faceOverlay, keypoints, faceScale, faceOffsetX, faceOffsetY);
                    } else {
                        ctx.save();
                        ctx.scale(-1, 1);
                        ctx.translate(-canvas.width, 0);
                        
                        ctx.fillStyle = '#00FF00';
                        ctx.strokeStyle = '#00FF00';
                        ctx.lineWidth = 1.5;
                        
                        keypoints.forEach(point => {
                            ctx.beginPath();
                            ctx.arc(point.x, point.y, 2.5, 0, 2 * Math.PI);
                            ctx.fill();
                        });
                        
                        ctx.restore();
                    }
                });
            }
            
            // Hand detection with rate limiting
            if (now - lastDetectionTime >= DETECTION_INTERVAL) {
                handDetector.estimateHands(video, {
                    flipHorizontal: false
                }).then(hands => {
                    if (hands.length > 0) {
                        hands.forEach(hand => {
                            if (handOverlay) {
                                drawImageOnLandmarks(handOverlay, hand.keypoints, handScale, handOffsetX, handOffsetY);
                            } else {
                                ctx.save();
                                ctx.scale(-1, 1);
                                ctx.translate(-canvas.width, 0);
                                
                                ctx.fillStyle = '#00FFFF';
                                ctx.strokeStyle = '#00FFFF';
                                ctx.lineWidth = 2;
                                
                                hand.keypoints.forEach(point => {
                                    ctx.beginPath();
                                    ctx.arc(point.x, point.y, 3, 0, 2 * Math.PI);
                                    ctx.fill();
                                });
                                
                                ctx.restore();
                            }
                        });
                    }
                });
            }
        });
        
        if (isTracking) {
            requestAnimationFrame(detectFaces);
        }
    } catch (error) {
        console.error('Error during detection:', error);
        updateStatus('Error during detection: ' + error.message);
        // Add delay before retry on error
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
        
        setupImageUploads();
        setupOffsetControls();
        console.log('Image upload handlers setup');
        
        await setupFaceDetector();
        await initHandDetector();
        console.log('Face and hand detectors setup complete');
        
        startTracking();
    } catch (error) {
        console.error('Initialization error:', error);
        updateStatus('Error: ' + error.message);
    }
}

// Start the application when the page loads
window.addEventListener('load', init);

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
        
        // Reset any previous transformations
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        
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
        // startTracking();
    } catch (error) {
        console.error('Error loading face detector:', error);
        updateStatus('Error: Could not load face detector - ' + error.message);
        throw error;
    }
}

async function initHandDetector() {
    try {
        const model = handPoseDetection.SupportedModels.MediaPipeHands;
        const detectorConfig = {
            runtime: 'tfjs',
            modelType: 'full',  // Use full model for better accuracy
            maxHands: 2,
            scoreThreshold: 0.5,  // Lower threshold for more sensitive detection
            refineLandmarks: true  // Enable landmark refinement for better precision
        };
        handDetector = await handPoseDetection.createDetector(model, detectorConfig);
        updateStatus('Hand detector initialized');
    } catch (error) {
        console.error('Error initializing hand detector:', error);
        updateStatus('Error: Hand detector initialization failed');
    }
}

function updateStatus(message) {
    const status = document.getElementById('status');
    status.textContent = message;
}

// Add cleanup function
function cleanup() {
    isTracking = false;
    if (detector) {
        detector.dispose();
    }
    if (handDetector) {
        handDetector.dispose();
    }
    tf.disposeVariables();
}

// Call cleanup when stopping tracking
function stopTracking() {
    isTracking = false;
    cleanup();
    updateStatus('Tracking stopped');
}

// Add event listener for page unload
window.addEventListener('unload', cleanup);

function setupOffsetControls() {
    const faceOffsetXInput = document.getElementById('faceOffsetX');
    const faceOffsetYInput = document.getElementById('faceOffsetY');
    const handOffsetXInput = document.getElementById('handOffsetX');
    const handOffsetYInput = document.getElementById('handOffsetY');

    faceOffsetXInput.addEventListener('input', (e) => {
        faceOffsetX = parseInt(e.target.value) || 0;
    });

    faceOffsetYInput.addEventListener('input', (e) => {
        faceOffsetY = parseInt(e.target.value) || 0;
    });

    handOffsetXInput.addEventListener('input', (e) => {
        handOffsetX = parseInt(e.target.value) || 0;
    });

    handOffsetYInput.addEventListener('input', (e) => {
        handOffsetY = parseInt(e.target.value) || 0;
    });
}
