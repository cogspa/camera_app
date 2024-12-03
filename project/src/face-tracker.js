import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import '@mediapipe/face_mesh';
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';

export class FaceTracker {
  constructor(video, canvas) {
    this.video = video;
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.model = null;
    this.isTracking = false;
  }

  async initialize() {
    try {
      // Ensure TensorFlow.js is properly initialized with WebGL backend
      await tf.setBackend('webgl');
      await tf.ready();
      
      // Load the MediaPipe Facemesh model
      this.model = await faceLandmarksDetection.load(
        faceLandmarksDetection.SupportedPackages.mediapipeFacemesh,
        {
          maxFaces: 1,
          refineLandmarks: true,
          shouldLoadIrisModel: false
        }
      );
      
      return true;
    } catch (error) {
      console.error('Error loading face tracking model:', error);
      throw new Error('Failed to initialize face tracking model');
    }
  }

  async startTracking() {
    if (!this.model) {
      throw new Error('Face tracking model not initialized');
    }
    
    this.isTracking = true;
    this.track();
  }

  stopTracking() {
    this.isTracking = false;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  async track() {
    if (!this.isTracking) return;

    try {
      const faces = await this.model.estimateFaces({
        input: this.video,
        returnTensors: false,
        flipHorizontal: false
      });

      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      
      faces.forEach(face => {
        // Draw face mesh points
        this.ctx.fillStyle = '#32CD32';
        face.keypoints.forEach(point => {
          this.ctx.beginPath();
          this.ctx.arc(point.x, point.y, 2, 0, 2 * Math.PI);
          this.ctx.fill();
        });

        // Draw face bounding box
        if (face.box) {
          this.ctx.strokeStyle = '#00ff00';
          this.ctx.lineWidth = 2;
          this.ctx.strokeRect(
            face.box.xMin,
            face.box.yMin,
            face.box.xMax - face.box.xMin,
            face.box.yMax - face.box.yMin
          );
        }
      });
    } catch (error) {
      console.error('Error during face tracking:', error);
    }

    if (this.isTracking) {
      requestAnimationFrame(() => this.track());
    }
  }
}