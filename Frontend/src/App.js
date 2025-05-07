import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { FaCut, FaTshirt, FaUpload, FaFilePdf, FaSpinner, FaExclamationTriangle, 
         FaDownload, FaRuler, FaCheck, FaRecycle } from 'react-icons/fa';
import './App.css';

// Use environment variable or fallback to localhost for development
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001/api';

function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [file, setFile] = useState(null);
  const [settings, setSettings] = useState({
    fabricWidth: 1000,
    allowRotation: true,
    arbitraryRotation: true,
    margin: 5,
    rotationStep: 15
  });
  const [result, setResult] = useState(null);
  const [svgContent, setSvgContent] = useState('');

  const onDrop = (acceptedFiles) => {
    const uploadedFile = acceptedFiles[0];
    if (uploadedFile && uploadedFile.type === 'application/pdf') {
      setFile(uploadedFile);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf']
    },
    maxFiles: 1
  });

  const handleSettingChange = (e) => {
    const { name, value, type, checked } = e.target;
    setSettings(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : type === 'number' ? Number(value) : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    setIsLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('fabric_width', settings.fabricWidth.toString());
    formData.append('allow_rotation', settings.allowRotation.toString());
    formData.append('arbitrary_rotation', settings.arbitraryRotation.toString());
    formData.append('margin', settings.margin.toString());
    formData.append('rotation_step', settings.rotationStep.toString());

    try {
      const response = await axios.post(`${API_URL}/optimize`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResult(response.data);
      
      // Fetch SVG content
      const svgResponse = await axios.get(`${API_URL}/files/${response.data.svg_path}`, {
        responseType: 'text',
      });
      setSvgContent(svgResponse.data);
    } catch (err) {
      setError(err.message || 'An error occurred during optimization');
      console.error('Optimization error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <header>
        <div className="container header-content">
          <div className="logo">
            <FaCut className="logo-icon" />
            <h1>Pattern Optimizer</h1>
          </div>
          <div className="badge">
            <FaTshirt className="badge-icon" />
            <span className="badge-text">Fabric Scrap Analysis</span>
          </div>
        </div>
      </header>
      
      <main>
        <div className="container">
          {/* Hero Section */}
          {!result && (
            <div className="hero">
              <h1>Pattern Optimization with Scrap Analysis</h1>
              <p>Upload your pattern PDF and optimize fabric usage while identifying usable scrap areas.</p>
            </div>
          )}
          
          {/* Upload Form */}
          <div className="card">
            <h2>Upload Pattern</h2>
            
            <div 
              {...getRootProps()} 
              className={`dropzone ${isDragActive ? 'active' : ''}`}
            >
              <input {...getInputProps()} />
              
              {file ? (
                <div className="file-info">
                  <FaFilePdf className="file-icon" />
                  <p className="file-name">{file.name}</p>
                  <p className="file-size">{(file.size / 1024).toFixed(1)} KB</p>
                </div>
              ) : (
                <div className="file-info">
                  <FaUpload className="upload-icon" />
                  <p className="file-name">Drag & drop a PDF pattern file here</p>
                  <p className="file-size">or click to browse files</p>
                </div>
              )}
            </div>

            <form onSubmit={handleSubmit}>
              <div className="form-grid">
                <div className="form-group">
                  <label className="form-label">Fabric Width (pixels)</label>
                  <input
                    type="number"
                    name="fabricWidth"
                    value={settings.fabricWidth}
                    onChange={handleSettingChange}
                    className="form-input"
                    min="100"
                    max="5000"
                  />
                </div>
                
                <div className="form-group">
                  <label className="form-label">Margin Between Pieces (pixels)</label>
                  <input
                    type="number"
                    name="margin"
                    value={settings.margin}
                    onChange={handleSettingChange}
                    className="form-input"
                    min="0"
                    max="50"
                  />
                </div>
                
                <div className="form-group">
                  <label className="form-label">Rotation Step (degrees)</label>
                  <input
                    type="number"
                    name="rotationStep"
                    value={settings.rotationStep}
                    onChange={handleSettingChange}
                    className="form-input"
                    min="1"
                    max="90"
                    disabled={!settings.allowRotation || !settings.arbitraryRotation}
                  />
                </div>
                
                <div className="form-group">
                  <div className="checkbox-group">
                    <input
                      type="checkbox"
                      name="allowRotation"
                      checked={settings.allowRotation}
                      onChange={handleSettingChange}
                      className="checkbox"
                      id="allowRotation"
                    />
                    <label className="checkbox-label" htmlFor="allowRotation">Allow Rotation</label>
                  </div>
                  
                  <div className="checkbox-group">
                    <input
                      type="checkbox"
                      name="arbitraryRotation"
                      checked={settings.arbitraryRotation}
                      onChange={handleSettingChange}
                      disabled={!settings.allowRotation}
                      className="checkbox"
                      id="arbitraryRotation"
                    />
                    <label className="checkbox-label" htmlFor="arbitraryRotation">Allow Arbitrary Rotation</label>
                  </div>
                </div>
              </div>

              <button
                type="submit"
                disabled={!file || isLoading}
                className={`button button-primary button-full ${!file || isLoading ? 'disabled' : ''}`}
              >
                {isLoading ? (
                  <>
                    <FaSpinner className="spinner" />
                    Optimizing Pattern...
                  </>
                ) : (
                  'Optimize Pattern'
                )}
              </button>
            </form>
          </div>
          
          {/* Error Message */}
          {error && (
            <div className="error-message">
              <div className="error-content">
                <FaExclamationTriangle className="error-icon" />
                <p className="error-text">{error}</p>
              </div>
            </div>
          )}
          
          {/* Results Display */}
          {result && svgContent && (
            <div className="results">
              <div className="results-grid">
                {/* SVG Visualization */}
                <div className="card">
                  <h2>Optimized Pattern Layout</h2>
                  <div className="svg-container" dangerouslySetInnerHTML={{ __html: svgContent }} />
                  <div className="download-buttons">
                    <a 
                      href={`${API_URL}/files/${result.svg_path}`} 
                      download 
                      className="button button-secondary"
                    >
                      <FaDownload className="download-icon" />
                      Download SVG
                    </a>
                    <a 
                      href={`${API_URL}/files/${result.pdf_path}`} 
                      download 
                      className="button button-primary"
                    >
                      <FaDownload className="download-icon" />
                      Download PDF
                    </a>
                  </div>
                </div>

                {/* Metrics */}
                <div className="card">
                  <h2>Pattern Metrics</h2>
                  
                  <div className="metrics-group">
                    <h3 className="metrics-heading">
                      <FaRuler className="metrics-icon" />
                      Fabric Dimensions
                    </h3>
                    <div className="metrics-grid">
                      <div className="metrics-item">
                        <p className="metrics-label">Width</p>
                        <p className="metrics-value">{result.metrics.fabric_width.toFixed(1)} px</p>
                        <p className="metrics-subvalue">≈ {(result.metrics.fabric_width / 96).toFixed(1)} in</p>
                      </div>
                      <div className="metrics-item">
                        <p className="metrics-label">Height</p>
                        <p className="metrics-value">{result.metrics.fabric_height.toFixed(1)} px</p>
                        <p className="metrics-subvalue">≈ {(result.metrics.fabric_height / 96).toFixed(1)} in</p>
                      </div>
                    </div>
                  </div>
                  
                  <div className="metrics-group">
                    <h3 className="metrics-heading">
                      <FaCheck className="metrics-icon" />
                      Efficiency
                    </h3>
                    <div className="metrics-grid">
                      <div className="metrics-item">
                        <p className="metrics-label">Utilization</p>
                        <p className="metrics-value">{result.metrics.utilization.toFixed(1)}%</p>
                      </div>
                      <div className="metrics-item">
                        <p className="metrics-label">Fabric Saved</p>
                        <p className="metrics-value">{result.metrics.fabric_saved_percent.toFixed(1)}%</p>
                      </div>
                    </div>
                  </div>
                  
                  <div className="metrics-group">
                    <h3 className="metrics-heading">
                      <FaRecycle className="metrics-icon" />
                      Scrap Analysis
                    </h3>
                    <div className="metrics-stack">
                      <div className="metrics-item">
                        <p className="metrics-label">Total Scrap Area</p>
                        <p className="metrics-value">{result.metrics.scrap_analysis.total_scrap_area.toFixed(0)} px²</p>
                        <p className="metrics-subvalue">
                          ≈ {(result.metrics.scrap_analysis.total_scrap_area / (96 * 96)).toFixed(2)} in²
                        </p>
                      </div>
                      <div className="metrics-item">
                        <p className="metrics-label">Usable Rectangle Area</p>
                        <p className="metrics-value">{result.metrics.scrap_analysis.usable_rectangle_area.toFixed(0)} px²</p>
                        <p className="metrics-subvalue">
                          {(result.metrics.scrap_analysis.usable_rectangle_area / result.metrics.scrap_analysis.total_scrap_area * 100).toFixed(1)}% of scrap
                        </p>
                      </div>
                      <div className="metrics-item">
                        <p className="metrics-label">Practical Usable Area</p>
                        <p className="metrics-value">{result.metrics.scrap_analysis.practical_usable_area.toFixed(0)} px²</p>
                        <p className="metrics-subvalue">
                          {(result.metrics.scrap_analysis.practical_usable_area / result.metrics.scrap_analysis.total_scrap_area * 100).toFixed(1)}% of scrap
                        </p>
                      </div>
                      <div className="metrics-item">
                        <p className="metrics-label">Scrap Quality Index</p>
                        <p className="metrics-value">{result.metrics.scrap_analysis.scrap_quality_index.toFixed(1)}%</p>
                        <div className="progress-bar">
                          <div 
                            className="progress-fill" 
                            style={{ width: `${result.metrics.scrap_analysis.scrap_quality_index}%` }} 
                          ></div>
                        </div>
                      </div>
                      <div className="metrics-item">
                        <p className="metrics-label">Usable Rectangles</p>
                        <p className="metrics-value">{result.metrics.scrap_analysis.usable_rectangles_count}</p>
                        {result.metrics.scrap_analysis.truly_usable_rectangles_count && (
                          <p className="metrics-subvalue">
                            {result.metrics.scrap_analysis.truly_usable_rectangles_count} truly usable
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
      
      <footer>
        <div className="container footer-content">
          <div className="footer-info">
            <h3 className="footer-title">Pattern Optimizer</h3>
            <p className="footer-subtitle">Optimize fabric usage and analyze scrap</p>
          </div>
          <div className="footer-copyright">
            <p className="footer-year"> 2025 Pattern Optimizer</p>
            <p className="footer-project">CSCI 566 Final Project</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
