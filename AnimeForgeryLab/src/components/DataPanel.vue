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
      <div class="p-5 text-base font-bold leading-relaxed text-[var(--text-primary)] whitespace-pre-wrap overflow-y-auto flex-1 font-mono">
        {{ report || $t('data.reportWait') }}
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
  report: String,
  probeVal: String,
  dangerLevel: String
});

const emit = defineEmits(['execute', 'api-changed']);

const showModal = ref(false);
const activeTab = ref('cloud'); // 'cloud' | 'local'
const cloudUrl = 'https://feiyang0102-ani-foregery-detector.hf.space';

// Initialize with the cloud URL directly making it the default
const currentApiUrl = ref(cloudUrl);
const tempApiUrl = ref('http://127.0.0.1:7860');

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
