import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';
import * as handPoseDetection from '@tensorflow-models/hand-pose-detection';

// Global variables
let detector = null;
let handDetector = null;
let video = null;
let canvas = null;
let ctx = null;
let isTracking = false;
let currentStream = null;

// Overlay images
let faceOverlay = null;
let handOverlay = null;
let faceScale = 1.0;
let handScale = 1.0;

// Simplified drawing function without manual flipping
function drawImageOnLandmarks(img, landmarks, scale) {
    if (!img || !landmarks || landmarks.length < 2) return;

    let minX = Infinity, minY = Infinity;
    let maxX = -Infinity, maxY = -Infinity;

    // Use raw coordinates since video and canvas are both flipped in CSS
    landmarks.forEach(point => {
        const x = point.x;
        const y = point.y;
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
    });

    const width = (maxX - minX) * scale;
    const height = (maxY - minY) * scale;
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;

    ctx.save();
    ctx.translate(centerX, centerY);
    ctx.drawImage(img, -width / 2, -height / 2, width, height);
    ctx.restore();
}

async function detectFaces() {
    if (!detector || !video || !canvas || !ctx || !isTracking) return;
    
    try {
        if (video.readyState !== 4) {
            requestAnimationFrame(detectFaces);
            return;
        }

        // Since video is flipped in CSS, use flipHorizontal: true
        const predictions = await detector.estimateFaces(video, {
            flipHorizontal:false,
            staticImageMode: false,
            predictIrises: false
        });
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (predictions.length > 0) {
            predictions.forEach(prediction => {
                const keypoints = prediction.keypoints;
                
                if (faceOverlay) {
                    // Use keypoints directly since video and canvas are both flipped
                    drawImageOnLandmarks(faceOverlay, keypoints, faceScale);
                } else {
                    // Draw keypoints directly
                    ctx.fillStyle = '#00FF00';
                    ctx.strokeStyle = '#00FF00';
                    ctx.lineWidth = 1.5;
                    
                    keypoints.forEach(point => {
                        ctx.beginPath();
                        ctx.arc(point.x, point.y, 2.5, 0, 2 * Math.PI);
                        ctx.fill();
                    });
                }
            });
        }
        
        // Handle hands the same way
        const hands = await handDetector.estimateHands(video, {
            flipHorizontal: true
        });
        
        if (hands.length > 0) {
            hands.forEach(hand => {
                if (handOverlay) {
                    drawImageOnLandmarks(handOverlay, hand.keypoints, handScale);
                } else {
                    ctx.fillStyle = '#00FFFF';
                    ctx.strokeStyle = '#00FFFF';
                    ctx.lineWidth = 2;
                    
                    hand.keypoints.forEach(point => {
                        ctx.beginPath();
                        ctx.arc(point.x, point.y, 3, 0, 2 * Math.PI);
                        ctx.fill();
                    });
                }
            });
        }
        
        if (isTracking) {
            requestAnimationFrame(detectFaces);
        }
    } catch (error) {
        console.error('Error during detection:', error);
        updateStatus('Error during detection: ' + error.message);
        if (isTracking) {
            setTimeout(detectFaces, 1000);
        }
    }
}

async function init() {
    try {
        updateStatus('Initializing...');
        
        // Optimize TensorFlow.js initialization
        await tf.ready();
        await tf.setBackend('webgl');
        tf.env().set('WEBGL_FORCE_F16_TEXTURES', true);
        tf.env().set('WEBGL_PACK', true);
        console.log('TensorFlow.js backend initialized:', tf.getBackend());
        
        // Initialize video element
        video = document.getElementById('video');
        if (!video) throw new Error('Video element not found');
        
        // Initialize canvas
        canvas = document.getElementById('canvas');
        if (!canvas) throw new Error('Canvas element not found');
        ctx = canvas.getContext('2d');
        if (!ctx) throw new Error('Could not get canvas context');
        
        // Set up camera stream
        await setupCamera();
        
        // Set up canvas dimensions
        canvas.width = video.width;
        canvas.height = video.height;
        
        // Load models in parallel
        updateStatus('Loading detection models...');
        const [faceModel, handModel] = await Promise.all([
            setupFaceDetector(),
            setupHandDetector()
        ]);
        
        console.log('Models loaded successfully');
        updateStatus('Ready!');
        
        // Start tracking
        isTracking = true;
        detectFaces();
        
    } catch (error) {
        console.error('Initialization error:', error);
        updateStatus('Error: ' + error.message);
    }
}

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
    try {
        const cameraSelect = document.getElementById('cameraSelect');
        const cameras = await getConnectedCameras();
        
        // Clear existing options
        cameraSelect.innerHTML = '';
        
        if (cameras.length === 0) {
            const option = document.createElement('option');
            option.text = 'No cameras found';
            cameraSelect.add(option);
            cameraSelect.disabled = true;
            throw new Error('No cameras found');
        }

        // Add options for each camera
        cameras.forEach(camera => {
            const option = document.createElement('option');
            option.value = camera.deviceId;
            option.text = camera.label || `Camera ${cameraSelect.length + 1}`;
            cameraSelect.add(option);
        });

        // Enable select and add change listener
        cameraSelect.disabled = false;
        cameraSelect.onchange = async (event) => {
            try {
                updateStatus('Switching camera...');
                await switchCamera(event.target.value);
            } catch (error) {
                console.error('Error in camera switch:', error);
                updateStatus('Failed to switch camera. Please try again.');
            }
        };

        console.log('Camera select populated with', cameras.length, 'cameras');
    } catch (error) {
        console.error('Error populating camera select:', error);
        updateStatus('Error: Could not list cameras - ' + error.message);
    }
}

async function switchCamera(deviceId) {
    try {
        // Stop tracking before switching camera
        isTracking = false;

        // Stop all tracks of the current stream if it exists
        if (currentStream) {
            const tracks = currentStream.getTracks();
            tracks.forEach(track => {
                track.stop();
                currentStream.removeTrack(track);
            });
            currentStream = null;
        }

        // Clear video source
        if (video.srcObject) {
            video.srcObject = null;
        }

        // Wait a moment for cleanup
        await new Promise(resolve => setTimeout(resolve, 500));

        // Configure new camera
        const constraints = {
            video: {
                deviceId: deviceId ? { exact: deviceId } : undefined,
                width: 640,
                height: 480
            }
        };

        // Get new stream
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        
        // Update video source
        video.srcObject = stream;
        currentStream = stream;

        // Wait for video to be ready
        await new Promise((resolve) => {
            video.onloadedmetadata = () => {
                video.play()
                    .then(() => {
                        console.log('New camera stream started');
                        resolve();
                    })
                    .catch(error => {
                        console.error('Error playing video:', error);
                        throw error;
                    });
            };
        });

        // Restart tracking
        isTracking = true;
        detectFaces();
        
        updateStatus('Camera switched successfully');
    } catch (error) {
        console.error('Error switching camera:', error);
        updateStatus('Error: Could not switch camera - ' + error.message);
        
        // Try to recover the previous stream if possible
        if (!currentStream) {
            try {
                const cameras = await getConnectedCameras();
                if (cameras.length > 0) {
                    await switchCamera(cameras[0].deviceId);
                }
            } catch (recoveryError) {
                console.error('Recovery failed:', recoveryError);
            }
        }
    }
}

async function setupCamera() {
    try {
        // Initialize video element if not already done
        if (!video) {
            video = document.getElementById('video');
            if (!video) {
                throw new Error('Video element not found');
            }
        }

        // Get list of cameras
        const cameras = await getConnectedCameras();
        if (cameras.length === 0) {
            throw new Error('No cameras found');
        }

        // Set up camera selection
        await populateCameraSelect();

        // Start with first camera if no stream exists
        if (!currentStream) {
            await switchCamera(cameras[0].deviceId);
        }

        console.log('Camera setup complete');
        updateStatus('Camera ready');
        return true;
    } catch (error) {
        console.error('Error setting up camera:', error);
        updateStatus('Error: Camera setup failed - ' + error.message);
        throw error;
    }
}

// Helper function to check if a device is in use
async function isDeviceInUse(deviceId) {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { deviceId: { exact: deviceId } }
        });
        stream.getTracks().forEach(track => track.stop());
        return false;
    } catch (error) {
        return error.name === 'NotReadableError' || error.name === 'TrackStartError';
    }
}

async function setupCanvas() {
    try {
        // Make sure canvas and video elements exist
        canvas = document.getElementById('canvas');
        if (!canvas) {
            throw new Error('Canvas element not found');
        }

        ctx = canvas.getContext('2d');
        if (!ctx) {
            throw new Error('Could not get canvas context');
        }

        // Make sure video element exists and is properly sized
        if (!video) {
            throw new Error('Video element not initialized');
        }

        // Set canvas dimensions
        canvas.width = video.width || 640;
        canvas.height = video.height || 480;
        
        // Set canvas position
        canvas.style.position = 'absolute';
        canvas.style.left = '0';
        canvas.style.top = '0';
        
        // Reset any previous transformations
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        
        console.log('Canvas setup complete with dimensions:', canvas.width, 'x', canvas.height);
        updateStatus('Canvas setup complete');
        return true;
    } catch (error) {
        console.error('Error setting up canvas:', error);
        updateStatus('Error: Canvas setup failed - ' + error.message);
        throw error;
    }
}

async function setupFaceDetector() {
    try {
        const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
        const detectorConfig = {
            runtime: 'tfjs',
            maxFaces: 1,
            refineLandmarks: false,
            shouldLoadIrisModel: false,
            enableFaceGeometry: false,
            modelType: 'lite'  // Use lite model for faster loading
        };
        
        detector = await faceLandmarksDetection.createDetector(model, detectorConfig);
        console.log('Face detector created');
    } catch (error) {
        console.error('Error loading face detector:', error);
        throw error;
    }
}

async function setupHandDetector() {
    try {
        const model = handPoseDetection.SupportedModels.MediaPipeHands;
        const detectorConfig = {
            runtime: 'tfjs',
            maxHands: 2,
            modelType: 'lite'  // Use lite model for faster loading
        };
        
        handDetector = await handPoseDetection.createDetector(model, detectorConfig);
        console.log('Hand detector created');
    } catch (error) {
        console.error('Error loading hand detector:', error);
        throw error;
    }
}

function updateStatus(message) {
    const status = document.getElementById('status');
    status.textContent = message;
}

// Start the application when the page loads
window.addEventListener('load', init);
