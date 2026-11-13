# NutriVision CV — Production Local AI Food Detection Web Application

NutriVision CV is a client-side Computer Vision web application that detects, localizes, and classifies multiple food items and dining containers in uploaded images using **YOLOv8** and **ONNX Runtime Web (WebGPU + WASM)**.

## Strict Technical Guarantees
- **Zero External AI Inference APIs:** No OpenAI, Gemini, Claude, or Hugging Face Inference endpoints.
- **Zero API Keys Required:** Runs 100% locally in the user's browser.
- **Vercel-Ready Static Architecture:** Avoids Vercel Python serverless function 250 MB bundle limits and 10-second timeouts by executing inference on the client edge via WebGPU / WebAssembly SIMD.

---

## Quick Start (Local Development)

### 1. Install Frontend Dependencies
```bash
npm install