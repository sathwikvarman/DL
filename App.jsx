import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import LiveSensorStream from './components/LiveSensorStream';
import SampleTester from './components/SampleTester';
import FileUploadAnalyzer from './components/FileUploadAnalyzer';
import ModelInspector from './components/ModelInspector';
import PredictionHistory from './components/PredictionHistory';
import { checkHealth, fetchStats } from './services/api';
import { 
  Activity, 
  Cpu, 
  Layers, 
  Database, 
  Radio, 
  Sparkles,
  Zap
} from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('live-stream');
  const [health, setHealth] = useState(null);
  const [stats, setStats] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(Date.now());

  // Polling / fetching health & stats
  useEffect(() => {
    async function loadSystemStatus() {
      try {
        const [h, s] = await Promise.all([checkHealth(), fetchStats()]);
        setHealth(h);
        setStats(s);
      } catch (err) {
        console.warn('Backend not currently reachable:', err);
      }
    }
    loadSystemStatus();
    const interval = setInterval(loadSystemStatus, 6000);
    return () => clearInterval(interval);
  }, [lastUpdate]);

  const handlePredictionTriggered = () => {
    setLastUpdate(Date.now());
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-['Plus_Jakarta_Sans',sans-serif]">
      {/* Top Navbar */}
      <Navbar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        health={health} 
        stats={stats} 
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        
        {/* Dynamic Tab Views */}
        {activeTab === 'live-stream' && (
          <LiveSensorStream onPredictionUpdate={handlePredictionTriggered} />
        )}

        {activeTab === 'samples' && (
          <SampleTester onPredictionUpdate={handlePredictionTriggered} />
        )}

        {activeTab === 'upload' && (
          <FileUploadAnalyzer onPredictionUpdate={handlePredictionTriggered} />
        )}

        {activeTab === 'model' && (
          <ModelInspector />
        )}

        {activeTab === 'history' && (
          <PredictionHistory />
        )}

      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/80 py-6 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center space-x-2">
            <Activity className="w-4 h-4 text-indigo-400" />
            <span className="text-slate-400 font-medium">HAR Deep Learning Platform</span>
            <span>•</span>
            <span>UCI Smartphone Dataset Benchmark</span>
          </div>
          <div className="flex items-center space-x-4 font-mono text-[11px] text-slate-500">
            <span>Window: 128 Timesteps (2.56s)</span>
            <span>Rate: 50 Hz</span>
            <span>Input: 9 Channels</span>
            <span>Loss: Categorical Crossentropy</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
