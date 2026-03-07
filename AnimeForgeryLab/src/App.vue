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
        :report-data="reportData"
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
const reportData = ref(null);
const probeValue = ref('--.--%');
const dangerLevel = ref('safe'); // 'safe' | 'warning' | 'danger'
const apiUrl = ref('https://feiyang0102-ani-foregery-detector.hf.space');

const handleImageSelect = (file) => {
  selectedFile.value = file;
  state.value = 'ready';
  overlayUrl.value = '';
  rawMaskData.value = null;
  reportData.value = null;
  probeValue.value = '--.--%';
  dangerLevel.value = 'safe';
};

const handleClear = () => {
  selectedFile.value = null;
  state.value = 'awaiting';
  overlayUrl.value = '';
  rawMaskData.value = null;
  reportData.value = null;
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
  reportData.value = null;
  
  try {
    const app = await client(apiUrl.value);
    
    const result = await app.predict("/predict", [
      selectedFile.value,
    ]);
    
    const overlayData = result.data[0];
    const reportObj = result.data[1];
    rawMaskData.value = result.data[2];
    
    if (overlayData && overlayData.url) {
      overlayUrl.value = overlayData.url;
    } else if (typeof overlayData === 'string') {
      overlayUrl.value = overlayData; 
    }
    
    // Parse if it came back as string, else use direct
    if (typeof reportObj === 'string') {
        try {
            reportData.value = JSON.parse(reportObj);
        } catch(e) {
            console.error("Failed to parse report JSON", e);
            reportData.value = reportObj; // fallback
        }
    } else {
        reportData.value = reportObj;
    }

    state.value = 'done';
    
    if (reportData.value && typeof reportData.value === 'object') {
        dangerLevel.value = reportData.value.danger_level || 'safe';
    } else {
        dangerLevel.value = 'safe';
    }
    
  } catch (err) {
    console.error(err);
    state.value = 'error';
    dangerLevel.value = 'danger';
  }
};
</script>
