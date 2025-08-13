import React, { useState, useRef } from 'react';
import { Upload, Camera, Loader2, X } from 'lucide-react';
import { AnalysisService } from '../services/AnalysisService';
import type { AnalysisResult } from '../types';

interface ImageUploadProps {
  onAnalysisComplete: (result: AnalysisResult, imageFile?: File) => void;
  isAnalyzing: boolean;
  setIsAnalyzing: (analyzing: boolean) => void;
}

const ImageUpload: React.FC<ImageUploadProps> = ({
  onAnalysisComplete,
  isAnalyzing,
  setIsAnalyzing
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [context, setContext] = useState<string>('');
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (file: File) => {
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    try {
      const analysisService = AnalysisService.getInstance();
      const result = await analysisService.analyzeImage(selectedFile, context);
      onAnalysisComplete(result, selectedFile);
    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      setIsAnalyzing(false);
    }
  }; 
 const clearImage = () => {
    setSelectedFile(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="image-upload">
      <div className="upload-section">
        <h2>Upload Food Image</h2>
        <p>Upload a clear image of your food for AI-powered nutritional analysis</p>

        <div
          className={`upload-area ${dragActive ? 'drag-active' : ''} ${selectedFile ? 'has-file' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          {previewUrl ? (
            <div className="preview-container">
              <img src={previewUrl} alt="Food preview" className="preview-image" />
              <button
                className="clear-button"
                onClick={(e) => {
                  e.stopPropagation();
                  clearImage();
                }}
              >
                <X size={20} />
              </button>
            </div>
          ) : (
            <div className="upload-placeholder">
              <Upload size={48} />
              <h3>Drop your food image here</h3>
              <p>or click to browse files</p>
              <span className="file-types">Supports: PNG, JPG, JPEG</span>
            </div>
          )}
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
          style={{ display: 'none' }}
        />
      </div>

      <div className="context-section">
        <h3>Additional Context (Optional)</h3>
        <textarea
          value={context}
          onChange={(e) => setContext(e.target.value)}
          placeholder="Provide additional details about the meal, cooking method, portion size, or special ingredients..."
          className="context-input"
          rows={4}
        />
      </div>

      <div className="action-section">
        <button
          className="analyze-button"
          onClick={handleAnalyze}
          disabled={!selectedFile || isAnalyzing}
        >
          {isAnalyzing ? (
            <>
              <Loader2 className="spinner" size={20} />
              Analyzing...
            </>
          ) : (
            <>
              <Camera size={20} />
              Analyze Food
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default ImageUpload;