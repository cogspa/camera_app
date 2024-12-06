# Face and Hand Tracking Web Application

A real-time web application for face and hand tracking with custom image overlays and interactive drawing capabilities.

## System Requirements

- Node.js (v14.0.0 or higher)
- Modern web browser with WebGL support (Chrome, Firefox, Edge recommended)
- Webcam
- GPU with WebGL support (for TensorFlow.js)

## Features

- Real-time face tracking
- Hand pose detection
- Custom image overlays
- Drawing canvas
- Green screen (chroma key) background removal
- Dynamic video backgrounds
- Responsive dark-themed UI

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/face-tracking-app.git
cd face-tracking-app
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

4. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:5173)

## Browser Permissions

The application requires the following permissions:
- Camera access
- JavaScript enabled
- WebGL enabled

## Development Setup

To modify or enhance the application:

1. Install development dependencies:
```bash
npm install --save-dev vite
```

2. Start development server with hot reload:
```bash
npm run dev
```

3. Build for production:
```bash
npm run build
```

4. Preview production build:
```bash
npm run preview
```

## Project Structure

```
project/
├── index.html          # Main HTML file
├── main.js            # Core application logic
├── package.json       # Dependencies and scripts
└── public/            # Static assets
```

## Dependencies

Core dependencies:
- @tensorflow/tfjs: ^4.11.0
- @tensorflow/tfjs-backend-webgl: ^4.11.0
- @tensorflow/tfjs-core: ^4.11.0
- @tensorflow-models/face-landmarks-detection: ^1.0.5
- @tensorflow-models/hand-pose-detection: ^2.0.1

Development dependencies:
- vite: ^4.4.9

## Troubleshooting

1. Camera not working:
   - Ensure camera permissions are granted in browser
   - Check if another application is using the camera
   - Try refreshing the page

2. Performance issues:
   - Ensure you have a GPU with WebGL support
   - Close other resource-intensive applications
   - Try reducing video resolution in settings

3. TensorFlow.js errors:
   - Clear browser cache and reload
   - Check for browser compatibility
   - Ensure WebGL is enabled

## Browser Compatibility

Tested and supported on:
- Chrome (latest)
- Firefox (latest)
- Edge (latest)

Note: Performance may vary based on hardware capabilities and browser implementation.

## Known Limitations

- Requires modern browser with WebGL support
- Performance dependent on client hardware
- May experience higher CPU/GPU usage during tracking
- Mobile performance may be limited

## License

[Your chosen license]

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request
