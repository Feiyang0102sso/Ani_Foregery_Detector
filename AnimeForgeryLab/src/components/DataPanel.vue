<template>
  <div class="p-6 flex flex-col gap-6 glass-panel rounded-lg flex-1">
    
    <!-- Action Button -->
    <button 
      @click="$emit('execute')"
      :disabled="!file || analysisState === 'analyzing'"
      class="cyber-btn w-full bg-transparent border-[3px] border-[var(--color-cyber-cyan)] text-[var(--color-cyber-cyan)] p-5 font-display text-2xl font-extrabold tracking-widest uppercase flex justify-between items-center relative overflow-hidden transition-all duration-300 disabled:border-[var(--border-panel)] disabled:text-[var(--text-muted)] disabled:cursor-not-allowed group shadow-[0_4px_15px_rgba(0,0,0,0.1)] dark:shadow-none"
    >
      <div class="absolute inset-0 bg-[var(--color-cyber-cyan)] -translate-x-full transition-transform duration-300 group-hover:translate-x-0 -z-10 disabled:hidden"></div>
      <span class="btn-text group-hover:text-white dark:group-hover:text-black transition-colors disabled:group-hover:text-[var(--text-muted)]">{{ $t('data.executeBtn') }}</span>
      <i class="ri-arrow-right-double-line group-hover:text-white dark:group-hover:text-black transition-colors disabled:group-hover:text-[var(--text-muted)]"></i>
    </button>

    <!-- Separator -->
    <div class="h-[1px] w-full bg-[repeating-linear-gradient(90deg,var(--border-panel),var(--border-panel)_4px,transparent_4px,transparent_8px)]"></div>

    <!-- Report Box -->
    <div 
      class="flex-1 flex flex-col bg-black/5 dark:bg-black/50 border border-[var(--border-panel)] border-l-4 rounded-r-lg min-h-[200px]"
      :class="{
        'border-l-[var(--color-cyber-cyan)]': dangerLevel === 'safe',
        'border-l-[var(--color-cyber-warning)]': dangerLevel === 'warning',
        'border-l-[var(--color-cyber-red)]': dangerLevel === 'danger'
      }"
    >
      <div class="text-xs font-bold text-[var(--text-muted)] p-3 border-b border-[var(--border-panel)] bg-black/5 dark:bg-white/5 tracking-wider font-mono uppercase">
        {{ $t('data.reportLabel') }}
      </div>
      <!-- When no data yet -->
      <div v-if="!reportData" class="flex flex-col justify-center items-center p-8 gap-4 opacity-50">
        <i class="ri-pie-chart-line text-4xl"></i>
        <span class="font-mono text-sm tracking-widest uppercase">{{ $t('data.reportWait') }}</span>
      </div>
      
      <!-- Visualization Report -->
      <div v-else class="flex flex-col md:flex-row gap-6 p-6 animate-fade-in">
        
        <!-- Left: Overall Probability (Circular Progress) -->
        <div class="flex-1 shrink-0 flex flex-col items-center justify-center gap-4 border-r-0 md:border-r border-[var(--border-panel)] pr-0 md:pr-6">
          <div class="relative w-40 h-40">
            <!-- SVG Circular Progress Bar -->
            <svg class="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
              <!-- Background Circle -->
              <circle 
                cx="50" cy="50" r="45" 
                fill="none" 
                stroke="currentColor" 
                stroke-width="8" 
                class="text-[var(--border-panel)] opacity-30"
              />
              <!-- Foreground Progress Circle -->
              <circle 
                cx="50" cy="50" r="45" 
                fill="none" 
                stroke="currentColor" 
                stroke-width="8" 
                stroke-linecap="round"
                :stroke-dasharray="283"
                :stroke-dashoffset="283 - (283 * (reportData.overall_ai_prob || 0)) / 100"
                class="transition-all duration-1000 ease-out"
                :class="{
                  'text-[var(--color-cyber-cyan)]': reportData.danger_level === 'safe',
                  'text-[var(--color-cyber-warning)]': reportData.danger_level === 'warning',
                  'text-[var(--color-cyber-red)]': reportData.danger_level === 'danger'
                }"
              />
            </svg>
            <!-- Center Text -->
            <div class="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
              <span class="text-3xl font-display font-extrabold"
                :class="{
                  'text-[var(--color-cyber-cyan)]': reportData.danger_level === 'safe',
                  'text-[var(--color-cyber-warning)]': reportData.danger_level === 'warning',
                  'text-[var(--color-cyber-red)]': reportData.danger_level === 'danger'
                }">
                {{ reportData.overall_ai_prob ? reportData.overall_ai_prob.toFixed(1) : 0 }}%
              </span>
              <span class="text-xs font-bold text-[var(--text-muted)] tracking-widest uppercase mt-1">AI PROB</span>
            </div>
          </div>
          
          <div class="text-center">
            <p class="text-[10px] text-[var(--text-muted)] mt-2 font-mono uppercase tracking-[0.15em] leading-relaxed max-w-[160px] mx-auto opacity-80">
              <span class="text-[var(--color-cyber-cyan)] font-extrabold block" v-if="reportData.danger_level === 'safe'">We are highly confident this is a Human made image</span>
              <span class="text-[var(--color-cyber-red)] font-extrabold block" v-else-if="reportData.danger_level === 'danger'">We are highly confident this is an AI image</span>
              <span class="text-[var(--color-cyber-warning)] font-extrabold block" v-else>We are moderately confident this is an AI image</span>
            </p>
          </div>
        </div>

        <!-- Right: Detail Metrics -->
        <div class="flex-1 flex flex-col gap-4 justify-center">
          
          <!-- Local Tampering Risk -->
          <div class="flex flex-col gap-1">
            <div class="flex justify-between items-end mb-1">
              <span class="text-xs font-mono font-bold tracking-widest text-[var(--text-muted)] uppercase">Local Tampering Risk</span>
              <span class="text-sm font-display font-bold text-[var(--text-primary)]">{{ reportData.local_tamper_prob ? reportData.local_tamper_prob.toFixed(1) : 0 }}%</span>
            </div>
            <div class="h-2 w-full bg-black/20 dark:bg-white/10 rounded-full overflow-hidden">
              <div 
                class="h-full transition-all duration-1000 rounded-full" 
                :class="{
                  'bg-[var(--color-cyber-cyan)]': (reportData.local_tamper_prob || 0) < 45,
                  'bg-[var(--color-cyber-warning)]': (reportData.local_tamper_prob || 0) >= 45 && (reportData.local_tamper_prob || 0) < 70,
                  'bg-[var(--color-cyber-red)]': (reportData.local_tamper_prob || 0) >= 70
                }"
                :style="`width: ${reportData.local_tamper_prob || 0}%`"
              ></div>
            </div>
          </div>

          <!-- Source Prediction (Donut Chart) -->
          <div class="bg-black/5 dark:bg-white/5 p-4 rounded-md border border-[var(--border-panel)] mt-2 flex items-center justify-between gap-4">
            <div class="flex flex-col gap-2 w-[140px]">
              <div class="text-[10px] font-mono font-bold tracking-[0.15em] text-[var(--text-muted)] uppercase mb-1">
                Generator Source
              </div>
              <div v-for="(prob, sourceStr) in (reportData.all_source_probs || {})" :key="sourceStr" class="flex items-center gap-2">
                 <div class="w-2 h-2 rounded-full shrink-0" 
                      :class="{
                        'bg-[var(--color-cyber-cyan)]': sourceStr.includes('Human') || sourceStr.includes('Real'),
                        'bg-[var(--color-cyber-red)]': sourceStr === 'SDXL' || sourceStr.includes('SDXL'),
                        'bg-[var(--color-cyber-warning)]': sourceStr.includes('FLUX'),
                        'bg-[#a855f7]': sourceStr === 'SD' || (!sourceStr.includes('Human') && sourceStr !== 'SDXL' && !sourceStr.includes('FLUX'))
                      }">
                 </div>
                 <div class="flex-1 text-[10px] font-bold truncate text-[var(--text-primary)]" :title="sourceStr">
                    {{ sourceStr }}
                 </div>
                 <div class="text-right text-[10px] font-mono text-[var(--text-muted)]">{{ prob.toFixed(1) }}%</div>
              </div>
            </div>
            
            <!-- CSS Donut Chart -->
            <div class="relative w-24 h-24 shrink-0 rounded-full flex items-center justify-center bg-[var(--bg-primary)]"
                 :style="getDonutGradient(reportData.all_source_probs)">
               <!-- Inner cut out -->
               <div class="absolute w-16 h-16 bg-[var(--bg-primary)] rounded-full border-4 border-black/5 dark:border-white/5"></div>
            </div>
          </div>

        </div>
      </div>
    </div>

    <!-- Probe Reading -->
    <div class="bg-black/5 dark:bg-white/5 p-5 border-2 border-[var(--border-panel)] flex flex-col gap-2 rounded-lg">
      <div class="flex items-center gap-2 text-[var(--color-cyber-cyan)] text-sm font-mono font-bold tracking-wider uppercase">
        <i class="ri-crosshair-2-line text-lg"></i>
        <span>{{ $t('data.probeTitle') }}</span>
      </div>
      <div class="text-5xl font-extrabold font-display text-[var(--text-primary)] tracking-[2px]">
        {{ probeVal }}
      </div>
      <div class="text-xs font-bold text-[var(--text-muted)] font-mono leading-tight">
        {{ $t('data.probeDesc') }}
      </div>
    </div>

    <!-- API Config (Click to open modal) -->
    <div class="flex items-center gap-3 text-sm font-bold text-[var(--text-muted)] cursor-pointer transition-colors hover:text-[var(--color-cyber-cyan)] bg-black/5 dark:bg-white/5 p-3 rounded-lg border border-transparent hover:border-[var(--color-cyber-cyan)]" @click="showModal = true">
      <i class="ri-server-line text-xl"></i>
      <div class="flex flex-col gap-1">
        <strong class="font-mono tracking-wider">{{ $t('data.apiConfig') }}</strong>
        <span class="opacity-90 tracking-wide font-mono">{{ currentApiUrl }}</span>
      </div>
    </div>

    <!-- Teleport Modal to Body end to avoid stacking context issues -->
    <Teleport to="body">
      <div v-if="showModal" class="fixed inset-0 bg-black/80 backdrop-blur-md flex justify-center items-center z-[1000]">
        <div class="bg-[var(--bg-primary)] p-10 w-[90%] max-w-[700px] flex flex-col gap-6 border-[3px] border-[var(--color-cyber-cyan)] shadow-[0_0_40px_rgba(0,255,204,0.15)] rounded-xl relative overflow-hidden">
          
          <h3 class="text-3xl font-mono font-extrabold text-[var(--color-cyber-cyan)] tracking-wider">
            {{ $t('data.apiConfigTitle') }}
          </h3>
          <p class="text-base font-bold text-[var(--text-muted)] leading-relaxed">
            Please select your preferred computation node for the backend inference.
          </p>
          
          <!-- Tabs Selection -->
          <div class="flex gap-4 p-1 bg-black/20 dark:bg-white/5 rounded-lg border border-[var(--border-panel)]">
            <button 
              @click="activeTab = 'cloud'"
              class="flex-1 py-3 px-4 rounded-md font-mono text-lg font-bold transition-all duration-300 flex items-center justify-center gap-2"
              :class="activeTab === 'cloud' ? 'bg-[var(--color-cyber-cyan)] text-black shadow-md' : 'text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-black/10'"
            >
              <i class="ri-cloud-windy-line"></i> HuggingFace Cloud
            </button>
            <button 
              @click="activeTab = 'local'"
              class="flex-1 py-3 px-4 rounded-md font-mono text-lg font-bold transition-all duration-300 flex items-center justify-center gap-2"
              :class="activeTab === 'local' ? 'bg-[var(--color-cyber-cyan)] text-black shadow-md' : 'text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-black/10'"
            >
              <i class="ri-macbook-line"></i> Custom / Local
            </button>
          </div>

          <!-- Tab Content: Cloud -->
          <div v-if="activeTab === 'cloud'" class="flex flex-col gap-4 py-4 animate-fade-in">
            <div class="p-4 border-l-4 border-[var(--color-cyber-cyan)] bg-black/5 dark:bg-white/5 flex flex-col gap-2 rounded-r-lg">
              <span class="text-sm font-bold text-[var(--text-muted)] uppercase tracking-wider">Preset HF Endpoint</span>
              <span class="font-mono text-xl text-[var(--text-primary)] font-bold select-all break-words">
                https://feiyang0102-ani-foregery-detector.hf.space
              </span>
            </div>
            <p class="text-sm text-[var(--text-muted)] mt-2 italic">
              * Note: Running on HuggingFace free spaces might experience queue times during heavy loads.
            </p>
          </div>

          <!-- Tab Content: Local/Custom -->
          <div v-if="activeTab === 'local'" class="flex flex-col gap-4 py-4 animate-fade-in">
             <div class="flex flex-col gap-2">
                <span class="text-sm font-bold text-[var(--text-muted)] uppercase tracking-wider">Custom Endpoint URL</span>
                <input 
                  type="text" 
                  v-model="tempApiUrl"
                  class="bg-black/5 dark:bg-black/50 border-2 border-[var(--border-panel)] text-[var(--text-primary)] p-5 font-mono text-xl font-bold w-full focus:outline-none focus:border-[var(--color-cyber-cyan)] rounded-lg"
                  placeholder="e.g. http://127.0.0.1:7860"
                  @keyup.enter="saveApi"
                >
             </div>
          </div>
          
          <!-- Actions -->
          <div class="flex justify-end gap-6 mt-4 items-center border-t border-[var(--border-panel)] pt-6">
            <button @click="showModal = false" class="text-base font-bold text-[var(--text-muted)] hover:text-[var(--text-primary)] px-4 uppercase tracking-widest transition-colors">
              CANCEL
            </button>
            <button @click="saveApi" class="bg-[var(--color-cyber-cyan)] text-black font-extrabold font-display px-8 py-4 text-xl tracking-widest hover:bg-white transition-colors rounded-lg uppercase shadow-[0_4px_15px_rgba(0,0,0,0.1)] dark:shadow-none">
              <i class="ri-link h-4 w-4 mr-2"></i> CONNECT
            </button>
          </div>
          
        </div>
      </div>
    </Teleport>

  </div>
</template>

<script setup>
import { ref } from 'vue';

defineProps({
  analysisState: String,
  file: File,
  reportData: Object,
  probeVal: String,
  dangerLevel: String
});

const emit = defineEmits(['execute', 'api-changed']);

const showModal = ref(false);
const activeTab = ref('cloud'); // 'cloud' | 'local'
const cloudUrl = 'https://feiyang0102-ani-foregery-detector.hf.space';

// Initialize with the cloud URL directly making it the default
const currentApiUrl = ref(cloudUrl);
const tempApiUrl = ref('http://127.0.0.1:7865');

const saveApi = () => {
  if (activeTab.value === 'cloud') {
    currentApiUrl.value = cloudUrl;
    emit('api-changed', cloudUrl);
  } else {
    const val = tempApiUrl.value.trim();
    if (val) {
      currentApiUrl.value = val;
      emit('api-changed', val);
    }
  }
  showModal.value = false;
};

// Helper for CSS Donut Chart
const getDonutGradient = (probs) => {
  if (!probs || Object.keys(probs).length === 0) return 'background: var(--border-panel)';
  
  let currentStart = 0;
  
  const colors = {
    'Human': 'var(--color-cyber-cyan)',
    'SDXL': 'var(--color-cyber-red)',
    'FLUX': 'var(--color-cyber-warning)',
    'SD': '#a855f7' // vivid purple hex to avoid undefined css var
  };

  const getSourceColor = (src) => {
    if (src.includes('Human') || src.includes('Real')) return colors['Human'];
    if (src === 'SDXL' || src.includes('SDXL')) return colors['SDXL'];
    if (src.includes('FLUX')) return colors['FLUX'];
    return colors['SD'];
  };

  const segments = [];
  Object.entries(probs).forEach(([src, val]) => {
    if (val > 0) {
      const color = getSourceColor(src);
      const endPoint = currentStart + val;
      segments.push(`${color} ${currentStart}% ${endPoint}%`);
      currentStart = endPoint;
    }
  });
  
  if (segments.length === 0) return 'background: var(--border-panel)';
  
  return `background: conic-gradient(${segments.join(', ')})`;
};
</script>

<style scoped>
/* Add a simple fade-in animation for tab transitions */
.animate-fade-in {
  animation: fadeIn 0.3s ease-out forwards;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(5px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
