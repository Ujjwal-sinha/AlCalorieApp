import React, { useState, useRef } from 'react';
import { Upload, Camera, Loader2, X, Image as ImageIcon, Sparkles } from 'lucide-react';
import { AnalysisService } from '../services/AnalysisService';
import type { AnalysisResult } from '../types';
import './ImageUpload.css';

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
    <div className="image-upload-container">
      <div className="upload-frame">
        <div className="upload-header-section">
          <div className="upload-icon-wrapper">
            <ImageIcon size={32} />
          </div>
          <h3>Upload Your Food Image</h3>
          <p>Drag & drop or click to select an image</p>
        </div>

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
              <div className="preview-overlay">
                <button
                  className="clear-button"
                  onClick={(e) => {
                    e.stopPropagation();
                    clearImage();
                  }}
                >
                  <X size={20} />
                </button>
                <div className="preview-badge">
                  <Sparkles size={16} />
                  <span>Ready for Analysis</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="upload-placeholder">
              <div className="upload-icon-large">
                <Upload size={64} />
              </div>
              <h4>Drop your food image here</h4>
              <p>or click to browse files</p>
              <div className="file-types">
                <span>Supports: PNG, JPG, JPEG</span>
              </div>
              <div className="upload-tips">
                <div className="tip">
                  <Camera size={16} />
                  <span>Good lighting recommended</span>
                </div>
                <div className="tip">
                  <ImageIcon size={16} />
                  <span>Clear, focused images work best</span>
                </div>
              </div>
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
        <div className="context-header">
          <h4>Additional Context (Optional)</h4>
          <p>Help improve analysis accuracy</p>
        </div>
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
          className={`analyze-button ${!selectedFile ? 'disabled' : ''} ${isAnalyzing ? 'analyzing' : ''}`}
          onClick={handleAnalyze}
          disabled={!selectedFile || isAnalyzing}
        >
          {isAnalyzing ? (
            <>
              <Loader2 className="spinner" size={20} />
              Analyzing with AI...
            </>
          ) : (
            <>
              <Camera size={20} />
              Analyze Food
            </>
          )}
        </button>
        
        {selectedFile && (
          <div className="file-info">
            <div className="file-details">
              <span className="file-name">{selectedFile.name}</span>
              <span className="file-size">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageUpload;