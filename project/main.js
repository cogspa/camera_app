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

// Drawing functionality
let isDrawing = false;
let drawingCanvas;
let drawingCtx;
let currentColor = '#000000';
let currentSize = 5;

// Add background canvas variables
let bgCanvas;
let bgCtx;

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
        
        // Add CSS variable for background image
        document.documentElement.style.setProperty('--bg-image', 'none');
        document.body.style.setProperty('background-image', 'var(--bg-image)');
        
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
        setupDrawingCanvas();
        console.log('Controls setup complete');
        
        await setupFaceDetector();
        await initHandDetector();
        console.log('Face and hand detectors setup complete');
        
        // Start background update
        requestAnimationFrame(updateBackground);
        
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

// Add background update function
function posterize(imageData, levels) {
    const data = imageData.data;
    const step = Math.floor(255 / (levels - 1));
    
    for (let i = 0; i < data.length; i += 4) {
        // Get grayscale value
        const gray = (data[i] + data[i + 1] + data[i + 2]) / 3;
        const posterized = Math.round(gray / step) * step;
        
        // Apply pink and blue tinting based on brightness
        if (posterized > 128) {
            // Brighter areas get pink tint
            data[i] = posterized + 50;     // R: increase red
            data[i + 1] = posterized - 30;  // G: decrease green
            data[i + 2] = posterized + 20;  // B: slight increase blue
        } else {
            // Darker areas get blue tint
            data[i] = posterized - 30;      // R: decrease red
            data[i + 1] = posterized - 30;  // G: decrease green
            data[i + 2] = posterized + 50;  // B: increase blue
        }
    }
    return imageData;
}

function updateBackground() {
    if (!video || !isTracking) return;
    
    // Create a temporary canvas to capture the current video frame
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.width * 0.5;  // 50% of video width
    tempCanvas.height = video.height * 0.5; // 50% of video height
    const tempCtx = tempCanvas.getContext('2d');
    
    // Draw the video frame at 50% size
    tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
    
    // Get image data for posterize effect
    const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
    
    // Apply posterize effect with pink and blue tinting
    const processedImageData = posterize(imageData, 5); // 5 levels of posterization
    
    // Put the processed image data back
    tempCtx.putImageData(processedImageData, 0, 0);
    
    // Convert the canvas to a data URL
    const backgroundImage = tempCanvas.toDataURL();
    
    // Update the body's background
    document.body.style.setProperty('--bg-image', `url(${backgroundImage})`);
    
    // Clean up
    tempCanvas.remove();
    
    // Request next frame
    requestAnimationFrame(updateBackground);
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
    // Reset background
    document.body.style.setProperty('--bg-image', 'none');
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

function setupDrawingCanvas() {
    drawingCanvas = document.getElementById('drawingCanvas');
    drawingCtx = drawingCanvas.getContext('2d', { alpha: true });
    
    // Clear with transparency
    clearDrawing();
    
    // Setup drawing controls
    const clearBtn = document.getElementById('clearDrawing');
    const flipVertBtn = document.getElementById('flipVertical');
    const flipHorizBtn = document.getElementById('flipHorizontal');
    const useAsOverlayBtn = document.getElementById('useAsOverlay');
    const colorPicker = document.getElementById('drawingColor');
    const brushSize = document.getElementById('brushSize');
    
    // Drawing event listeners
    drawingCanvas.addEventListener('mousedown', startDrawing);
    drawingCanvas.addEventListener('mousemove', draw);
    drawingCanvas.addEventListener('mouseup', stopDrawing);
    drawingCanvas.addEventListener('mouseout', stopDrawing);
    
    drawingCanvas.addEventListener('pointerdown', startDrawing);
    drawingCanvas.addEventListener('pointermove', draw);
    drawingCanvas.addEventListener('pointerup', stopDrawing);
    drawingCanvas.addEventListener('pointerout', stopDrawing);
    
    drawingCanvas.addEventListener('touchstart', (e) => e.preventDefault());
    drawingCanvas.addEventListener('touchmove', (e) => e.preventDefault());
    drawingCanvas.addEventListener('touchend', (e) => e.preventDefault());
    
    // Control event listeners
    clearBtn.addEventListener('click', clearDrawing);
    flipVertBtn.addEventListener('click', flipVertical);
    flipHorizBtn.addEventListener('click', flipHorizontal);
    useAsOverlayBtn.addEventListener('click', useDrawingAsOverlay);
    colorPicker.addEventListener('input', (e) => currentColor = e.target.value);
    brushSize.addEventListener('input', (e) => currentSize = parseInt(e.target.value));
}

function startDrawing(e) {
    isDrawing = true;
    draw(e); // Draw a single point on click/touch
}

function draw(e) {
    if (!isDrawing) return;
    
    // Prevent scrolling
    e.preventDefault();
    
    const rect = drawingCanvas.getBoundingClientRect();
    let x, y;
    
    if (e.type.startsWith('touch')) {
        x = e.touches[0].clientX - rect.left;
        y = e.touches[0].clientY - rect.top;
    } else if (e.type.startsWith('pointer')) {
        x = e.clientX - rect.left;
        y = e.clientY - rect.top;
    } else {
        x = e.clientX - rect.left;
        y = e.clientY - rect.top;
    }
    
    // Pressure sensitivity for pen input
    let pressure = e.pressure !== undefined ? e.pressure : 1;
    let size = currentSize * (pressure || 1);
    
    drawingCtx.fillStyle = currentColor;
    drawingCtx.strokeStyle = currentColor;
    drawingCtx.lineWidth = size;
    
    drawingCtx.lineTo(x, y);
    drawingCtx.stroke();
    drawingCtx.beginPath();
    drawingCtx.arc(x, y, size/2, 0, Math.PI * 2);
    drawingCtx.fill();
    drawingCtx.beginPath();
    drawingCtx.moveTo(x, y);
}

function stopDrawing(e) {
    if (e) e.preventDefault();
    isDrawing = false;
    drawingCtx.beginPath(); // Start a new path for next drawing
}

function clearDrawing() {
    // Clear the entire canvas with transparency
    drawingCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    drawingCtx.beginPath();
}

function flipVertical() {
    // Create a temporary canvas to store the flipped image
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = drawingCanvas.width;
    tempCanvas.height = drawingCanvas.height;
    const tempCtx = tempCanvas.getContext('2d');
    
    // Draw the current canvas content onto the temporary canvas, flipped
    tempCtx.scale(1, -1);
    tempCtx.translate(0, -drawingCanvas.height);
    tempCtx.drawImage(drawingCanvas, 0, 0);
    
    // Clear the original canvas
    drawingCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    
    // Draw the flipped image back onto the original canvas
    drawingCtx.drawImage(tempCanvas, 0, 0);
    
    // Clean up
    tempCanvas.remove();
}

function flipHorizontal() {
    // Create a temporary canvas
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = drawingCanvas.width;
    tempCanvas.height = drawingCanvas.height;
    const tempCtx = tempCanvas.getContext('2d');
    
    // Draw the current canvas content flipped horizontally
    tempCtx.translate(tempCanvas.width, 0);
    tempCtx.scale(-1, 1);
    tempCtx.drawImage(drawingCanvas, 0, 0);
    
    // Clear original canvas
    drawingCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    
    // Draw the flipped image back
    drawingCtx.drawImage(tempCanvas, 0, 0);
    
    // Clean up
    tempCanvas.remove();
}

function useDrawingAsOverlay() {
    // Create a new image from the drawing canvas
    const drawingImage = new Image();
    drawingImage.onload = () => {
        faceOverlay = drawingImage;
        updateStatus('Drawing set as face overlay');
    };
    // Explicitly use PNG format to preserve transparency
    drawingImage.src = drawingCanvas.toDataURL('image/png');
}
