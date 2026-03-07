<template>
  <div class="glass-panel rounded-lg flex flex-col relative overflow-hidden group">
    <!-- Top Border Glow -->
    <div class="absolute top-0 left-0 w-full h-[2px] bg-gradient-to-r from-[var(--color-cyber-cyan)] to-transparent opacity-50"></div>
    
    <!-- Header -->
    <div class="p-4 px-5 border-b-2 border-dashed border-[var(--border-panel)] flex justify-between items-center text-sm font-bold text-[var(--text-muted)]">
      <span class="tracking-widest">{{ $t('upload.title') }}</span>
      <button 
        @click="$emit('clear')"
        :disabled="analysisState === 'awaiting'"
        class="bg-transparent border border-transparent text-[var(--text-muted)] font-mono text-sm px-3 py-1 transition-all hover:text-[var(--color-cyber-red)] hover:border-[var(--color-cyber-red)] disabled:opacity-30 disabled:cursor-not-allowed uppercase flex items-center gap-1 font-bold"
      >
        <i class="ri-delete-bin-line text-lg"></i> {{ $t('upload.purge') }}
      </button>
    </div>

    <!-- Upload Zone -->
    <div 
      ref="dropZone"
      class="flex-1 flex justify-center items-center relative overflow-hidden p-4 transition-all duration-300 min-h-[400px]"
      :class="isDragging ? 'border-2 border-dashed border-[var(--color-cyber-cyan)] bg-[var(--color-cyber-cyan)]/5' : ''"
      @click="triggerFileInput"
      @dragover.prevent="isDragging = true"
      @dragleave.prevent="isDragging = false"
      @drop.prevent="handleDrop"
    >
      <input 
        type="file" 
        ref="fileInput" 
        accept="image/*" 
        class="hidden" 
        @change="handleFileChange"
      >
      
      <!-- Placeholder -->
      <div v-if="analysisState === 'awaiting'" class="text-center text-[var(--text-primary)] transition-all duration-300 group-hover:scale-105 group-hover:text-[var(--color-cyber-cyan)]">
        <i class="ri-upload-cloud-2-line text-7xl mb-4 block"></i>
        <p class="text-2xl mb-2 font-display font-extrabold">{{ $t('upload.promptMain') }}</p>
        <p class="text-sm font-bold tracking-widest opacity-80">{{ $t('upload.promptSub') }}</p>
      </div>

      <!-- Image Preview & Overlay -->
      <div 
        v-else 
        class="relative w-full h-full flex justify-center items-center"
        @click.stop="handleImageClick"
      >
        <img 
          ref="sourceImageEl"
          :src="previewUrl" 
          class="max-w-full max-h-full object-contain absolute transition-all duration-500"
          :class="{ 'brightness-[0.25] blur-sm': analysisState === 'analyzing' }"
          alt="Source"
        >
        <img 
          v-if="heatmapUrl"
          :src="heatmapUrl" 
          class="max-w-full max-h-full object-contain absolute transition-opacity duration-1000"
          :class="analysisState === 'done' ? 'opacity-100' : 'opacity-0'"
          alt="Heatmap"
        >
        
        <!-- Loading Spinner Animation -->
        <div 
          v-if="analysisState === 'analyzing'"
          class="absolute inset-0 flex flex-col justify-center items-center z-20"
        >
          <i class="ri-loader-4-line text-6xl text-[var(--color-cyber-cyan)] animate-spin"></i>
          <span class="mt-4 text-[var(--color-cyber-cyan)] font-mono font-bold tracking-widest animate-pulse">ANALYZING LOGIC...</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, onBeforeUnmount } from 'vue';

const props = defineProps({
  analysisState: String,
  heatmapUrl: String,
  rawData: Array // 2D array of mask probabilities
});

const emit = defineEmits(['image-selected', 'clear', 'probe-update']);

const fileInput = ref(null);
const dropZone = ref(null);
const sourceImageEl = ref(null);
const isDragging = ref(false);
const previewUrl = ref('');

// Clean up object URLs to prevent memory leaks
onBeforeUnmount(() => {
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value);
});

// Watch analysis state to clear preview when state goes back to awaiting
watch(() => props.analysisState, (newVal) => {
  if (newVal === 'awaiting') {
    if (previewUrl.value) URL.revokeObjectURL(previewUrl.value);
    previewUrl.value = '';
    if (fileInput.value) fileInput.value.value = '';
  }
});

const triggerFileInput = () => {
  if (props.analysisState === 'awaiting' && fileInput.value) {
    fileInput.value.click();
  }
};

const processFile = (file) => {
  if (!file || !file.type.startsWith('image/')) {
    alert('Please upload an image file.');
    return;
  }
  
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value);
  previewUrl.value = URL.createObjectURL(file);
  emit('image-selected', file);
};

const handleFileChange = (e) => {
  if (e.target.files && e.target.files.length > 0) {
    processFile(e.target.files[0]);
  }
};

const handleDrop = (e) => {
  isDragging.value = false;
  if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
    processFile(e.dataTransfer.files[0]);
  }
};

const handleImageClick = (e) => {
  if (props.analysisState !== 'done' || !props.rawData || !sourceImageEl.value) return;
  
  const rect = sourceImageEl.value.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const clickY = e.clientY - rect.top;

  const percentX = clickX / rect.width;
  const percentY = clickY / rect.height;

  const maskHeight = props.rawData.length;
  const maskWidth = props.rawData[0].length;

  const arrX = Math.floor(percentX * maskWidth);
  const arrY = Math.floor(percentY * maskHeight);

  if (arrY >= 0 && arrY < maskHeight && arrX >= 0 && arrX < maskWidth) {
    const prob = props.rawData[arrY][arrX] * 100;
    emit('probe-update', `${prob.toFixed(2)}%`);
    createProbePing(e.clientX, e.clientY);
  }
};

const createProbePing = (x, y) => {
  const ping = document.createElement('div');
  Object.assign(ping.style, {
    position: 'fixed',
    left: `${x - 10}px`,
    top: `${y - 10}px`,
    width: '20px',
    height: '20px',
    border: '2px solid var(--color-cyber-cyan)',
    borderRadius: '50%',
    pointerEvents: 'none',
    zIndex: '9999',
    transition: 'all 0.5s cubic-bezier(0.1, 0.9, 0.2, 1)',
    transform: 'scale(1)',
    opacity: '1'
  });
  
  document.body.appendChild(ping);
  
  requestAnimationFrame(() => {
    ping.style.transform = 'scale(3)';
    ping.style.opacity = '0';
    setTimeout(() => ping.remove(), 500);
  });
};
</script>

<style scoped>
</style>
