/**
 * TextScanner — OCR + LLM Summarization Module
 * Uses Tesseract.js for text extraction and Ollama for summarization.
 */

export class TextScanner {
  constructor() {
    this._worker = null;
    this._workerReady = false;
    this._loading = false;
    this._state = 'idle'; // idle | scanning | summarizing | done | error
    this._rawText = '';
    this._summary = '';
    this._error = '';
    this._progress = 0; // OCR progress 0-100
    this._abortController = null;
    
    // Ollama config
    this._ollamaUrl = 'http://localhost:11434/api/generate';
    this._ollamaModel = 'qwen2.5:3b';
  }

  get state() { return this._state; }
  get rawText() { return this._rawText; }
  get summary() { return this._summary; }
  get error() { return this._error; }
  get progress() { return this._progress; }
  get isReady() { return this._workerReady; }

  /** Initialize the Tesseract.js worker. Call once on first use. */
  async initWorker() {
    if (this._workerReady || this._loading) return;
    this._loading = true;
    
    try {
      // Dynamically load Tesseract.js from CDN if not already loaded
      if (!window.Tesseract) {
        await this._loadScript('https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js');
      }
      
      // Create a worker with English language
      this._worker = await window.Tesseract.createWorker('eng', 1, {
        logger: (m) => {
          if (m.status === 'recognizing text') {
            this._progress = Math.round((m.progress || 0) * 100);
          }
        },
      });
      
      this._workerReady = true;
      this._loading = false;
      console.log('[TextScanner] Tesseract.js worker ready');
    } catch (err) {
      this._loading = false;
      console.error('[TextScanner] Failed to init Tesseract worker:', err);
      throw err;
    }
  }

  /** Load an external script dynamically */
  _loadScript(src) {
    return new Promise((resolve, reject) => {
      if (document.querySelector(`script[src="${src}"]`)) {
        resolve();
        return;
      }
      const script = document.createElement('script');
      script.src = src;
      script.onload = resolve;
      script.onerror = reject;
      document.head.appendChild(script);
    });
  }

  /**
   * Capture a frame from the video element, run OCR, return the frozen canvas.
   * @param {HTMLVideoElement} video
   * @returns {HTMLCanvasElement} the frozen frame canvas (for display)
   */
  async scan(video) {
    if (!this._workerReady) {
      await this.initWorker();
    }
    
    this._state = 'scanning';
    this._rawText = '';
    this._summary = '';
    this._error = '';
    this._progress = 0;
    
    // Freeze frame to a canvas
    const freezeCanvas = document.createElement('canvas');
    freezeCanvas.width = video.videoWidth || video.width || 1280;
    freezeCanvas.height = video.videoHeight || video.height || 720;
    const ctx = freezeCanvas.getContext('2d');
    ctx.drawImage(video, 0, 0, freezeCanvas.width, freezeCanvas.height);
    
    try {
      // Run OCR
      const result = await this._worker.recognize(freezeCanvas);
      const text = (result.data.text || '').trim();
      
      if (!text || text.length < 3) {
        this._rawText = '';
        this._state = 'done';
        this._error = 'No readable text detected in the frame. Try capturing a frame with visible text.';
        return { canvas: freezeCanvas, words: [], text: '' };
      }
      
      this._rawText = text;
      
      // Extract word bounding boxes for highlighting
      const words = (result.data.words || []).map(w => ({
        text: w.text,
        bbox: w.bbox, // { x0, y0, x1, y1 }
        confidence: w.confidence,
      }));
      
      return { canvas: freezeCanvas, words, text };
    } catch (err) {
      this._state = 'error';
      this._error = `OCR failed: ${err.message}`;
      return { canvas: freezeCanvas, words: [], text: '' };
    }
  }

  /**
   * Send text to Ollama for summarization, streaming the response.
   * @param {string} text - the OCR-extracted text
   * @param {function} onChunk - callback(partialSummary) called as each word arrives
   * @returns {string} the full summary
   */
  async summarize(text, onChunk) {
    if (!text || text.length < 3) {
      this._state = 'done';
      return '';
    }
    
    this._state = 'summarizing';
    this._summary = '';
    
    // Abort any previous request
    if (this._abortController) {
      this._abortController.abort();
    }
    this._abortController = new AbortController();
    
    const prompt = `Summarize the following text concisely in 3-5 sentences:\n\n${text}`;
    
    try {
      const response = await fetch(this._ollamaUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: this._ollamaModel,
          prompt: prompt,
          stream: true,
        }),
        signal: this._abortController.signal,
      });
      
      if (!response.ok) {
        throw new Error(`Ollama returned ${response.status}: ${response.statusText}. Is Ollama running at ${this._ollamaUrl}?`);
      }
      
      // Stream the response
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        
        // Ollama sends newline-delimited JSON
        const lines = buffer.split('\n');
        buffer = lines.pop(); // keep incomplete line in buffer
        
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const json = JSON.parse(line);
            if (json.response) {
              this._summary += json.response;
              if (onChunk) onChunk(this._summary);
            }
            if (json.done) {
              this._state = 'done';
            }
          } catch {
            // skip malformed lines
          }
        }
      }
      
      // Process remaining buffer
      if (buffer.trim()) {
        try {
          const json = JSON.parse(buffer);
          if (json.response) {
            this._summary += json.response;
            if (onChunk) onChunk(this._summary);
          }
        } catch {
          // skip
        }
      }
      
      this._state = 'done';
      return this._summary;
    } catch (err) {
      if (err.name === 'AbortError') {
        this._state = 'idle';
        return this._summary;
      }
      this._state = 'error';
      this._error = `Summarization failed: ${err.message}`;
      if (onChunk) onChunk(this._summary);
      return this._summary;
    }
  }

  /** Reset all state for a new scan */
  reset() {
    if (this._abortController) {
      this._abortController.abort();
      this._abortController = null;
    }
    this._state = 'idle';
    this._rawText = '';
    this._summary = '';
    this._error = '';
    this._progress = 0;
  }

  /** Cleanup worker */
  async destroy() {
    this.reset();
    if (this._worker) {
      await this._worker.terminate();
      this._worker = null;
      this._workerReady = false;
    }
  }
}
