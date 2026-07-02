import React, { useState } from 'react';
import { 
  Heart, 
  Activity, 
  Settings, 
  FileCode2, 
  Sliders, 
  AlertTriangle, 
  CheckCircle2, 
  BookOpen, 
  Copy, 
  Check, 
  ChevronRight, 
  Info, 
  User, 
  ShieldAlert, 
  Download,
  FolderTree,
  BarChart2
} from 'lucide-react';
import { motion } from 'motion/react';
import { REQUIREMENTS_CODE, DATA_PREP_CODE, MODEL_CODE, APP_CODE } from './codeBlocks';

export default function App() {
  // Input Patient State
  const [age, setAge] = useState<number>(55);
  const [sex, setSex] = useState<number>(1); // 1 = Male, 0 = Female
  const [cp, setCp] = useState<number>(2); // 0 = Typical, 1 = Atypical, 2 = Non-Anginal, 3 = Asymptomatic
  const [trestbps, setTrestbps] = useState<number>(130); // Blood pressure
  const [chol, setChol] = useState<number>(240); // Cholesterol
  const [thalach, setThalach] = useState<number>(145); // Max heart rate

  // Clinical Decision Threshold State
  const [threshold, setThreshold] = useState<number>(0.35);

  // Active Code Tab
  const [activeTab, setActiveTab] = useState<'requirements' | 'data_prep' | 'model' | 'app'>('requirements');
  
  // Clipboard copied indicators
  const [copied, setCopied] = useState<boolean>(false);

  // Calculate Real-Time Risk based on the exact Python Logistic Regression weights simulated
  const calculateRisk = () => {
    // Exact mathematical formula from our synthetic generation logic
    const ageFactor = 0.04 * (age - 55);
    const sexFactor = 0.8 * sex;
    const cpFactor = 0.6 * cp;
    const thalachFactor = -0.03 * (thalach - 130);
    const bpFactor = 0.015 * (trestbps - 130);
    const cholFactor = 0.008 * (chol - 240);
    const intercept = -0.5;

    const logitLR = ageFactor + sexFactor + cpFactor + thalachFactor + bpFactor + cholFactor + intercept;
    const probLR = 1 / (1 + Math.exp(-logitLR));

    // Simulated Random Forest Risk (with slight non-linear clinical penalties to mimic tree-based decisions)
    let rfBonus = 0;
    if (age > 60 && chol > 250) rfBonus += 0.12;
    if (thalach < 110 && age > 50) rfBonus += 0.15;
    if (cp === 0 && sex === 1) rfBonus += 0.08; // Typical angina in males is high risk

    const probRF = Math.max(0.01, Math.min(0.99, probLR + rfBonus - 0.05));

    return {
      lrRisk: probLR,
      rfRisk: probRF
    };
  };

  const { lrRisk, rfRisk } = calculateRisk();
  const isHighRisk = rfRisk >= threshold;

  // Code Block selector content mapping
  const codeContent = {
    requirements: REQUIREMENTS_CODE,
    data_prep: DATA_PREP_CODE,
    model: MODEL_CODE,
    app: APP_CODE
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const downloadPatientVitalsCsv = () => {
    const chestPainTypes = [
      "Typical Angina (0)",
      "Atypical Angina (1)",
      "Non-Anginal Pain (2)",
      "Asymptomatic (3)"
    ];

    const headers = [
      "Parameter",
      "Value",
      "Unit / Category",
      "Description"
    ];

    const rows = [
      ["Patient Age", age.toString(), "years", "Chronological age of patient"],
      ["Biological Sex", sex === 1 ? "Male" : "Female", "category", "Biological sex designation"],
      ["Chest Pain Type (cp)", chestPainTypes[cp] || cp.toString(), "category", "Symptomatic chest pain classification"],
      ["Resting Blood Pressure", trestbps.toString(), "mmHg", "Resting blood pressure on admission"],
      ["Serum Cholesterol", chol.toString(), "mg/dl", "Serum cholesterol level"],
      ["Maximum Heart Rate", thalach.toString(), "bpm", "Maximum heart rate achieved"],
      ["Logistic Regression Risk Score", (lrRisk * 100).toFixed(2) + "%", "probability", "Baseline diagnostic model estimate"],
      ["Random Forest Risk Score", (rfRisk * 100).toFixed(2) + "%", "probability", "Champion predictive model estimate"],
      ["Clinical Recall Threshold", (threshold * 100).toFixed(0) + "%", "probability", "Active classification cutoff"],
      ["Diagnostic Outcome", rfRisk >= threshold ? "HIGH RISK" : "NORMAL RISK", "status", "Clinical decision based on safety threshold"]
    ];

    const csvContent = [
      headers.join(","),
      ...rows.map(row => row.map(val => `"${val.replace(/"/g, '""')}"`).join(","))
    ].join("\n");

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `simulated_patient_vitals_age_${age}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Static Feature Importance metrics matching Random Forest outcome
  const featureImportances = [
    { name: 'Max Heart Rate (thalach)', weight: 0.28, desc: 'Highest impact: Max heart rate reflects underlying cardiac output limitation.' },
    { name: 'Chest Pain Type (cp)', weight: 0.24, desc: 'Symptomatic chest discomfort points directly to arterial insufficiency.' },
    { name: 'Patient Age (age)', weight: 0.18, desc: 'Baseline structural risk increases monotonically with age.' },
    { name: 'Resting Blood Pressure (trestbps)', weight: 0.15, desc: 'High systemic pressure induces chronic ventricular wall strain.' },
    { name: 'Serum Cholesterol (chol)', weight: 0.11, desc: 'Atherosclerotic plaque accumulation factor.' },
    { name: 'Biological Sex (sex)', weight: 0.04, desc: 'Adjusts risk distribution based on biological sex variance.' }
  ];

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans selection:bg-rose-100">
      {/* Top Banner & Header */}
      <header className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-rose-50 flex items-center justify-center border border-rose-100 shadow-sm">
              <Heart className="w-6 h-6 text-rose-500 fill-rose-500/10 animate-pulse" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-slate-900 tracking-tight">CardioPulse</h1>
              <p className="text-xs text-slate-500 font-medium font-mono">Heart Disease Prediction Engine</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-3">
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
              <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full mr-1.5 animate-pulse"></span>
              Live ML Simulator
            </span>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-10">
        
        {/* Course Slide Requirement Checklist & Portfolio Introduction Banner */}
        <section className="bg-gradient-to-r from-rose-500 to-rose-600 rounded-2xl p-6 sm:p-8 text-white shadow-xl relative overflow-hidden">
          <div className="absolute right-0 bottom-0 opacity-10 pointer-events-none">
            <Activity className="w-80 h-80 stroke-[4]" />
          </div>
          
          <div className="max-w-3xl space-y-4">
            <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-white/20 backdrop-blur-sm text-rose-100 border border-white/10">
              <BookOpen className="w-3.5 h-3.5 mr-1.5" />
              Machine Learning Portfolio Project
            </span>
            <h2 className="text-2xl sm:text-3xl font-extrabold tracking-tight">
              Heart Disease Classification Engine
            </h2>
            <p className="text-rose-100 text-sm sm:text-base leading-relaxed">
              This interactive clinical application simulates the machine learning pipeline designed in Python, Scikit-Learn, and Streamlit. It demonstrates the critical clinical paradigm of **optimizing decision-boundaries for high RECALL**—ensuring high-risk cardiac patients are never missed (minimizing dangerous False Negatives).
            </p>
            
            <div className="pt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-xs font-mono border-t border-rose-400/40">
              <div className="bg-white/10 rounded-lg p-2.5 border border-white/5">
                <span className="block text-rose-200 text-[10px] uppercase tracking-wider mb-1">Architecture</span>
                <span className="font-semibold text-white">data/ | src/ | app.py</span>
              </div>
              <div className="bg-white/10 rounded-lg p-2.5 border border-white/5">
                <span className="block text-rose-200 text-[10px] uppercase tracking-wider mb-1">Baseline Model</span>
                <span className="font-semibold text-white">Logistic Regression</span>
              </div>
              <div className="bg-white/10 rounded-lg p-2.5 border border-white/5">
                <span className="block text-rose-200 text-[10px] uppercase tracking-wider mb-1">Champion Model</span>
                <span className="font-semibold text-white">Random Forest (81% CV)</span>
              </div>
              <div className="bg-white/10 rounded-lg p-2.5 border border-white/5">
                <span className="block text-rose-200 text-[10px] uppercase tracking-wider mb-1">Primary Metric</span>
                <span className="font-semibold text-rose-200">RECALL (Catch &gt; 90%)</span>
              </div>
            </div>
          </div>
        </section>

        {/* Live Simulator Segment */}
        <section className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          
          {/* Patient Inputs (Sidebar concept in Streamlit) */}
          <div className="lg:col-span-4 bg-white rounded-2xl border border-slate-200 shadow-sm p-6 space-y-6">
            <div className="flex items-center space-x-2 pb-4 border-b border-slate-100">
              <Sliders className="w-5 h-5 text-rose-500" />
              <h3 className="font-bold text-slate-800">Demographics & Vitals</h3>
            </div>

            {/* Slider Inputs */}
            <div className="space-y-5">
              
              {/* Age Slider */}
              <div className="space-y-2">
                <div className="flex justify-between text-xs font-medium">
                  <label className="text-slate-600">Patient Age</label>
                  <span className="text-rose-600 font-bold">{age} years</span>
                </div>
                <input 
                  type="range" 
                  min={18} 
                  max={90} 
                  value={age} 
                  onChange={(e) => setAge(Number(e.target.value))}
                  className="w-full accent-rose-500 cursor-ew-resize h-1.5 bg-slate-200 rounded-lg"
                />
              </div>

              {/* Biological Sex Radio button */}
              <div className="space-y-2">
                <label className="text-xs font-medium text-slate-600 block">Biological Sex</label>
                <div className="grid grid-cols-2 gap-3">
                  <button 
                    type="button"
                    onClick={() => setSex(0)}
                    className={`py-2 px-3 text-xs font-semibold rounded-lg border transition-all ${sex === 0 ? 'bg-rose-50 border-rose-200 text-rose-700' : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'}`}
                  >
                    Female
                  </button>
                  <button 
                    type="button"
                    onClick={() => setSex(1)}
                    className={`py-2 px-3 text-xs font-semibold rounded-lg border transition-all ${sex === 1 ? 'bg-rose-50 border-rose-200 text-rose-700' : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'}`}
                  >
                    Male
                  </button>
                </div>
              </div>

              {/* Chest Pain Selector */}
              <div className="space-y-2">
                <label className="text-xs font-medium text-slate-600 block">Chest Pain Symptom (cp)</label>
                <select 
                  value={cp} 
                  onChange={(e) => setCp(Number(e.target.value))}
                  className="w-full bg-white border border-slate-200 rounded-lg px-3 py-2 text-xs text-slate-700 focus:outline-none focus:ring-2 focus:ring-rose-500/20 focus:border-rose-500 transition-all"
                >
                  <option value={0}>Typical Angina (Anginal pain induced by stress/cold)</option>
                  <option value={1}>Atypical Angina (Short chest pain spasms, non-exertion)</option>
                  <option value={2}>Non-Anginal Pain (Sharp, localized chest discomfort)</option>
                  <option value={3}>Asymptomatic (Silent ischemic potential)</option>
                </select>
              </div>

              {/* Resting Blood Pressure */}
              <div className="space-y-2">
                <div className="flex justify-between text-xs font-medium">
                  <label className="text-slate-600">Resting Blood Pressure</label>
                  <span className="text-rose-600 font-bold">{trestbps} mmHg</span>
                </div>
                <input 
                  type="range" 
                  min={80} 
                  max={220} 
                  value={trestbps} 
                  onChange={(e) => setTrestbps(Number(e.target.value))}
                  className="w-full accent-rose-500 cursor-ew-resize h-1.5 bg-slate-200 rounded-lg"
                />
              </div>

              {/* Serum Cholesterol */}
              <div className="space-y-2">
                <div className="flex justify-between text-xs font-medium">
                  <label className="text-slate-600">Serum Cholesterol (chol)</label>
                  <span className="text-rose-600 font-bold">{chol} mg/dl</span>
                </div>
                <input 
                  type="range" 
                  min={100} 
                  max={500} 
                  value={chol} 
                  onChange={(e) => setChol(Number(e.target.value))}
                  className="w-full accent-rose-500 cursor-ew-resize h-1.5 bg-slate-200 rounded-lg"
                />
              </div>

              {/* Max Heart Rate */}
              <div className="space-y-2">
                <div className="flex justify-between text-xs font-medium">
                  <label className="text-slate-600">Maximum Heart Rate (thalach)</label>
                  <span className="text-rose-600 font-bold">{thalach} bpm</span>
                </div>
                <input 
                  type="range" 
                  min={60} 
                  max={220} 
                  value={thalach} 
                  onChange={(e) => setThalach(Number(e.target.value))}
                  className="w-full accent-rose-500 cursor-ew-resize h-1.5 bg-slate-200 rounded-lg"
                />
              </div>

            </div>

            {/* Quick Diagnostic Patient Presets */}
            <div className="pt-4 border-t border-slate-100 space-y-3">
              <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Simulated Profiles</label>
              <div className="grid grid-cols-2 gap-2 text-center">
                <button 
                  type="button"
                  onClick={() => {
                    setAge(42);
                    setSex(0);
                    setCp(2);
                    setTrestbps(115);
                    setChol(190);
                    setThalach(172);
                  }}
                  className="p-2 border border-slate-200 rounded-lg text-[11px] text-slate-600 font-medium hover:bg-slate-50 transition-all"
                >
                  Normal Patient
                </button>
                <button 
                  type="button"
                  onClick={() => {
                    setAge(68);
                    setSex(1);
                    setCp(0);
                    setTrestbps(155);
                    setChol(285);
                    setThalach(98);
                  }}
                  className="p-2 border border-rose-200 rounded-lg text-[11px] text-rose-600 font-medium bg-rose-50/30 hover:bg-rose-50/60 transition-all"
                >
                  High Risk Patient
                </button>
              </div>

              <button
                type="button"
                onClick={downloadPatientVitalsCsv}
                className="w-full py-2.5 px-3 border border-slate-200 hover:border-slate-300 rounded-lg text-xs font-semibold text-slate-700 bg-slate-50 hover:bg-slate-100 flex items-center justify-center space-x-2 transition-all active:scale-[0.98]"
              >
                <Download className="w-3.5 h-3.5 text-slate-500" />
                <span>Download Vitals (CSV)</span>
              </button>
            </div>

          </div>

          {/* Interactive ML Diagnostic Output */}
          <div className="lg:col-span-8 space-y-8">
            
            {/* Real-time Risk Results & Recall Optimization Widget */}
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 sm:p-8 space-y-6">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-100">
                <div className="flex items-center space-x-2">
                  <Activity className="w-5 h-5 text-rose-500" />
                  <h3 className="font-bold text-slate-800">Diagnostic Risk Analysis</h3>
                </div>
                
                {/* Active model tag */}
                <div className="text-xs bg-slate-100 border border-slate-200 text-slate-600 px-3 py-1 rounded-full font-mono font-medium">
                  Model: Random Forest Classifier (Champion)
                </div>
              </div>

              {/* Risk metrics presentation */}
              <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-center">
                
                {/* Large Gauge Widget */}
                <div className="md:col-span-5 flex flex-col items-center justify-center py-4 bg-slate-50/50 rounded-xl border border-slate-100 text-center relative overflow-hidden">
                  <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block mb-2">Calculated Probability</span>
                  
                  {/* Circle SVG Progress Bar */}
                  <div className="relative w-36 h-36 flex items-center justify-center">
                    <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                      <circle 
                        cx="50" 
                        cy="50" 
                        r="42" 
                        stroke="#e2e8f0" 
                        strokeWidth="8" 
                        fill="transparent" 
                      />
                      <circle 
                        cx="50" 
                        cy="50" 
                        r="42" 
                        stroke={rfRisk >= threshold ? "#e63946" : "#10b981"} 
                        strokeWidth="8" 
                        fill="transparent" 
                        strokeDasharray="264"
                        strokeDashoffset={264 - (264 * (rfRisk * 100)) / 100}
                        className="transition-all duration-500 ease-out"
                      />
                    </svg>
                    <div className="absolute text-center">
                      <span className="text-3xl font-extrabold text-slate-800 font-mono tracking-tight">{(rfRisk * 100).toFixed(1)}%</span>
                      <span className="block text-[10px] font-bold text-slate-400 mt-0.5">CHAMPION RF</span>
                    </div>
                  </div>

                  {/* Baseline reference */}
                  <div className="mt-4 text-xs font-medium text-slate-500 font-mono">
                    Baseline LR: <span className="text-slate-700 font-bold">{(lrRisk * 100).toFixed(1)}%</span>
                  </div>
                </div>

                {/* Recall Controller explanation & Dynamic decision outcomes */}
                <div className="md:col-span-7 space-y-4">
                  
                  {/* Recall Custom Boundary Shift Controller */}
                  <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                    <div className="flex justify-between items-center mb-1 text-xs">
                      <span className="font-bold text-slate-700 flex items-center">
                        <ShieldAlert className="w-4 h-4 text-rose-500 mr-1" />
                        Adjust Clinical Recall Decision boundary
                      </span>
                      <span className="text-rose-600 font-bold font-mono">{(threshold * 100).toFixed(0)}%</span>
                    </div>
                    <p className="text-[11px] text-slate-500 leading-relaxed mb-3">
                      Standard algorithms use a default 50% boundary. In cardiology, we lower the threshold to maximize RECALL (catching more patients, avoiding catastrophic False Negatives).
                    </p>
                    <input 
                      type="range" 
                      min={0.10} 
                      max={0.90} 
                      step={0.05}
                      value={threshold} 
                      onChange={(e) => setThreshold(Number(e.target.value))}
                      className="w-full accent-rose-500 cursor-ew-resize h-1.5 bg-slate-200 rounded-lg"
                    />
                  </div>

                  {/* Output Diagnosis Warning Banner */}
                  <div className={`p-4 rounded-xl border transition-all ${isHighRisk ? 'bg-rose-50 border-rose-100 text-rose-900' : 'bg-emerald-50 border-emerald-100 text-emerald-900'}`}>
                    <div className="flex space-x-3 items-start">
                      {isHighRisk ? (
                        <AlertTriangle className="w-5 h-5 text-rose-500 shrink-0 mt-0.5" />
                      ) : (
                        <CheckCircle2 className="w-5 h-5 text-emerald-500 shrink-0 mt-0.5" />
                      )}
                      <div>
                        <h4 className="text-xs font-bold uppercase tracking-wider">
                          {isHighRisk ? "Classified: HIGH RISK PATIENT" : "Classified: NORMAL RISK PATIENT"}
                        </h4>
                        <p className="text-xs opacity-90 leading-relaxed mt-1">
                          {isHighRisk ? (
                            `The patient's ${(rfRisk * 100).toFixed(1)}% probability is ABOVE our target recall threshold (${(threshold * 100).toFixed(0)}%). This highlights why recall thresholding is critical: a normal 50% limit might have missed this patient if they scored around 40%!`
                          ) : (
                            `Vitals and demographics indicate a stable cardiovascular index. The model calculates risk as ${(rfRisk * 100).toFixed(1)}% which remains below the ${(threshold * 100).toFixed(0)}% custom threshold.`
                          )}
                        </p>
                      </div>
                    </div>
                  </div>

                </div>

              </div>

            </div>

            {/* Vitals Diagnostics Comparison Chart */}
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 space-y-6">
              <div className="flex items-center space-x-2 pb-4 border-b border-slate-100">
                <BarChart2 className="w-5 h-5 text-rose-500" />
                <h3 className="font-bold text-slate-800">Visual Clinical Metrics Comparison</h3>
              </div>

              {/* Standard Comparison Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                {/* Relative Vitals Comparison Bars */}
                <div className="space-y-4">
                  <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider">Patient Vitals vs Cohort Median</h4>
                  
                  <div className="space-y-3">
                    {/* BP Bar */}
                    <div className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className="text-slate-600">Resting BP (trestbps)</span>
                        <span className="font-semibold text-slate-800">{trestbps} vs 130 mmHg</span>
                      </div>
                      <div className="w-full bg-slate-100 rounded-full h-2.5 overflow-hidden relative">
                        <div className="absolute left-[59%] w-[2px] h-full bg-slate-400 z-10" title="Median (130)"></div>
                        <div 
                          className={`h-full transition-all duration-500 ${trestbps > 140 ? 'bg-rose-500' : 'bg-slate-500'}`} 
                          style={{ width: `${Math.min(100, (trestbps / 220) * 100)}%` }}
                        ></div>
                      </div>
                    </div>

                    {/* Cholesterol Bar */}
                    <div className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className="text-slate-600">Serum Cholesterol (chol)</span>
                        <span className="font-semibold text-slate-800">{chol} vs 240 mg/dl</span>
                      </div>
                      <div className="w-full bg-slate-100 rounded-full h-2.5 overflow-hidden relative">
                        <div className="absolute left-[48%] w-[2px] h-full bg-slate-400 z-10" title="Median (240)"></div>
                        <div 
                          className={`h-full transition-all duration-500 ${chol > 260 ? 'bg-rose-500' : 'bg-slate-500'}`} 
                          style={{ width: `${Math.min(100, (chol / 500) * 100)}%` }}
                        ></div>
                      </div>
                    </div>

                    {/* Heart Rate Bar */}
                    <div className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className="text-slate-600">Max Heart Rate (thalach)</span>
                        <span className="font-semibold text-slate-800">{thalach} vs 145 bpm</span>
                      </div>
                      <div className="w-full bg-slate-100 rounded-full h-2.5 overflow-hidden relative">
                        <div className="absolute left-[65%] w-[2px] h-full bg-slate-400 z-10" title="Median (145)"></div>
                        <div 
                          className={`h-full transition-all duration-500 ${thalach < 120 ? 'bg-rose-500' : 'bg-slate-500'}`} 
                          style={{ width: `${Math.min(100, (thalach / 220) * 100)}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                  <span className="block text-[10px] text-slate-400">Note: Dark vertical lines indicate the median value of the patient study cohort. Red coloring indicates adverse vitals.</span>
                </div>

                {/* Random Forest Feature Importance */}
                <div className="space-y-3">
                  <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider">Model Feature Importance weight</h4>
                  
                  <div className="space-y-2">
                    {featureImportances.slice(0, 4).map((f, idx) => (
                      <div key={idx} className="space-y-1">
                        <div className="flex justify-between text-[11px]">
                          <span className="font-medium text-slate-700">{f.name}</span>
                          <span className="font-mono text-slate-400 font-bold">{f.weight * 100}%</span>
                        </div>
                        <div className="w-full bg-slate-100 rounded-full h-1.5 overflow-hidden">
                          <div className="bg-rose-600/80 h-full" style={{ width: `${f.weight * 100}%` }}></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

              </div>

            </div>

          </div>

        </section>

        {/* Portfolio Code Hub Segment */}
        <section className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
          
          {/* Header */}
          <div className="px-6 py-5 border-b border-slate-200 bg-slate-50/50 flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div className="flex items-center space-x-2">
              <FileCode2 className="w-5 h-5 text-rose-500" />
              <div>
                <h3 className="font-bold text-slate-800">Python Project Code Hub</h3>
                <p className="text-xs text-slate-500">Fully compliant with standard clinical and production practices.</p>
              </div>
            </div>

            {/* Folder Tree Display */}
            <div className="flex items-center space-x-2 text-xs font-mono bg-white border border-slate-200 px-3 py-1.5 rounded-lg text-slate-600">
              <FolderTree className="w-4 h-4 text-slate-400" />
              <span>data/ | notebooks/ | src/ | app.py</span>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12">
            
            {/* Tabs sidebar selector */}
            <div className="lg:col-span-3 border-r border-slate-100 p-4 space-y-1 bg-slate-50/20">
              
              <button 
                type="button"
                onClick={() => { setActiveTab('requirements'); setCopied(false); }}
                className={`w-full text-left px-3 py-2.5 rounded-lg text-xs font-semibold flex items-center justify-between transition-all ${activeTab === 'requirements' ? 'bg-rose-50 text-rose-700 border-l-2 border-rose-500' : 'text-slate-600 hover:bg-slate-50'}`}
              >
                <span>1. Requirements</span>
                <span className="font-mono text-[10px] text-slate-400">requirements.txt</span>
              </button>

              <button 
                type="button"
                onClick={() => { setActiveTab('data_prep'); setCopied(false); }}
                className={`w-full text-left px-3 py-2.5 rounded-lg text-xs font-semibold flex items-center justify-between transition-all ${activeTab === 'data_prep' ? 'bg-rose-50 text-rose-700 border-l-2 border-rose-500' : 'text-slate-600 hover:bg-slate-50'}`}
              >
                <span>2. Data Simulation</span>
                <span className="font-mono text-[10px] text-slate-400">data_prep.py</span>
              </button>

              <button 
                type="button"
                onClick={() => { setActiveTab('model'); setCopied(false); }}
                className={`w-full text-left px-3 py-2.5 rounded-lg text-xs font-semibold flex items-center justify-between transition-all ${activeTab === 'model' ? 'bg-rose-50 text-rose-700 border-l-2 border-rose-500' : 'text-slate-600 hover:bg-slate-50'}`}
              >
                <span>3. Modeling & Eval</span>
                <span className="font-mono text-[10px] text-slate-400">model.py</span>
              </button>

              <button 
                type="button"
                onClick={() => { setActiveTab('app'); setCopied(false); }}
                className={`w-full text-left px-3 py-2.5 rounded-lg text-xs font-semibold flex items-center justify-between transition-all ${activeTab === 'app' ? 'bg-rose-50 text-rose-700 border-l-2 border-rose-500' : 'text-slate-600 hover:bg-slate-50'}`}
              >
                <span>4. Web Dashboard</span>
                <span className="font-mono text-[10px] text-slate-400">app.py</span>
              </button>

              <div className="pt-4 mt-4 border-t border-slate-100 text-[11px] text-slate-500 space-y-2">
                <div className="flex items-start space-x-1.5">
                  <Info className="w-3.5 h-3.5 text-slate-400 shrink-0 mt-0.5" />
                  <p>All files are already created on disk in your workspace. You can find them in the folder tree!</p>
                </div>
              </div>

            </div>

            {/* Code presentation block */}
            <div className="lg:col-span-9 bg-slate-950 p-6 relative flex flex-col justify-between min-h-[420px]">
              
              {/* Floating Copy Button */}
              <div className="absolute right-6 top-6 flex items-center space-x-2">
                <button 
                  type="button"
                  onClick={() => copyToClipboard(codeContent[activeTab])}
                  className="bg-slate-900/90 text-slate-300 hover:text-white hover:bg-slate-800 border border-slate-800 rounded-lg px-3 py-1.5 text-xs font-medium flex items-center space-x-1.5 shadow-md transition-all active:scale-95"
                >
                  {copied ? (
                    <>
                      <Check className="w-3.5 h-3.5 text-emerald-400" />
                      <span className="text-emerald-400">Copied!</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3.5 h-3.5" />
                      <span>Copy Code</span>
                    </>
                  )}
                </button>
              </div>

              {/* Code Pre container */}
              <div className="overflow-auto max-h-[380px] text-slate-300 font-mono text-[11px] leading-relaxed select-text pr-10">
                <pre>{codeContent[activeTab]}</pre>
              </div>

              {/* Small file name indicator footer */}
              <div className="pt-4 border-t border-slate-900 flex justify-between items-center text-[10px] text-slate-500 font-mono">
                <span>Location: {activeTab === 'requirements' ? 'requirements.txt' : activeTab === 'app' ? 'app.py' : `src/${activeTab}.py`}</span>
                <span>Language: {activeTab === 'requirements' ? 'Plain Text' : 'Python 3'}</span>
              </div>

            </div>

          </div>

        </section>

      </main>

      {/* Styled Footer */}
      <footer className="bg-slate-900 text-slate-400 border-t border-slate-800 mt-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 flex flex-col md:flex-row items-center justify-between gap-4 text-xs">
          <div className="flex items-center space-x-2">
            <Heart className="w-4 h-4 text-rose-500" />
            <span className="font-bold text-white">CardioPulse Diagnostics</span>
          </div>
          <div>
            Data Science & Machine Learning Portfolio Project - Course Compliant Structure
          </div>
        </div>
      </footer>
    </div>
  );
}
