<template>
  <div class="noise-overlay"></div>
  <div class="bg-grid-pattern"></div>

  <main class="w-[95%] max-w-[1400px] mx-auto min-h-screen flex flex-col gap-6 py-6 md:h-screen md:py-8">
    <Navbar />
    
    <div class="flex-1 grid grid-cols-1 md:grid-cols-[2fr_1.2fr] gap-6 min-h-0">
      <VisionPanel 
        :analysis-state="state" 
        :heatmap-url="overlayUrl"
        :raw-data="rawMaskData"
        @image-selected="handleImageSelect"
        @probe-update="handleProbeUpdate"
        @clear="handleClear"
      />
      
      <DataPanel 
        :analysis-state="state"
        :file="selectedFile"
        :report="reportText"
        :probe-val="probeValue"
        :danger-level="dangerLevel"
        @execute="handleExecute"
        @api-changed="updateApi"
      />
    </div>
  </main>
</template>

<script setup>
import { ref } from 'vue';
import { client } from "@gradio/client";
import { useI18n } from 'vue-i18n';
import Navbar from './components/Navbar.vue';
import VisionPanel from './components/VisionPanel.vue';
import DataPanel from './components/DataPanel.vue';

const { t } = useI18n();

// States
// 'awaiting' | 'ready' | 'analyzing' | 'done' | 'error'
const state = ref('awaiting'); 
const selectedFile = ref(null);
const overlayUrl = ref('');
const rawMaskData = ref(null);
const reportText = ref('');
const probeValue = ref('--.--%');
const dangerLevel = ref('safe'); // 'safe' | 'warning' | 'danger'
const apiUrl = ref('https://feiyang0102-ani-foregery-detector.hf.space');

const handleImageSelect = (file) => {
  selectedFile.value = file;
  state.value = 'ready';
  overlayUrl.value = '';
  rawMaskData.value = null;
  reportText.value = '';
  probeValue.value = '--.--%';
  dangerLevel.value = 'safe';
};

const handleClear = () => {
  selectedFile.value = null;
  state.value = 'awaiting';
  overlayUrl.value = '';
  rawMaskData.value = null;
  reportText.value = '';
  probeValue.value = '--.--%';
  dangerLevel.value = 'safe';
};

const handleProbeUpdate = (val) => {
  probeValue.value = val;
};

const updateApi = (url) => {
  apiUrl.value = url;
};

const handleExecute = async () => {
  if (!selectedFile.value) return;
  
  state.value = 'analyzing';
  reportText.value = `Establishing uplink to Remote Node [${apiUrl.value}]...\nUploading image tensor...`;
  
  try {
    const app = await client(apiUrl.value);
    reportText.value += "\nUplink established. Executing Shearlet Deep-Forgery Analysis...";
    
    const result = await app.predict("/predict", [
      selectedFile.value,
    ]);
    
    const overlayData = result.data[0];
    const reportRaw = result.data[1];
    rawMaskData.value = result.data[2];
    
    if (overlayData && overlayData.url) {
      overlayUrl.value = overlayData.url;
    } else if (typeof overlayData === 'string') {
      overlayUrl.value = overlayData; 
    }
    
    reportText.value = reportRaw;
    state.value = 'done';
    
    if (reportRaw.includes("🚨")) {
      dangerLevel.value = 'danger';
    } else if (reportRaw.includes("⚠️")) {
      dangerLevel.value = 'warning';
    } else {
      dangerLevel.value = 'safe';
    }
    
  } catch (err) {
    console.error(err);
    reportText.value = t('report.error') + `${err.message}\n\nPlease verify that the Remote API Node (${apiUrl.value}) is online and CORS is configured.`;
    state.value = 'error';
    dangerLevel.value = 'danger';
  }
};
</script>
