<template>
  <header class="flex justify-between items-end border-b-[3px] border-[var(--color-cyber-cyan)] pb-4 uppercase">
    <div class="flex items-center gap-4">
      <i class="ri-radar-scan-line text-5xl text-[var(--color-cyber-cyan)] animate-[spin_10s_linear_infinite]"></i>
      <div>
        <h1 class="font-display text-5xl font-extrabold tracking-widest -mb-1 text-[var(--text-primary)] drop-shadow-md dark:drop-shadow-[0_0_10px_rgba(255,255,255,0.3)]">
          {{ $t('navbar.title') }}
        </h1>
        <span class="text-sm font-bold text-[var(--color-cyber-cyan)] tracking-widest">{{ $t('navbar.version') }}</span>
      </div>
    </div>
    
    <div class="flex items-center gap-4">
      <!-- Status Indicator -->
      <div class="hidden md:flex items-center gap-2 text-base font-bold text-[var(--color-cyber-cyan)]">
        <span class="w-3 h-3 rounded-full bg-[var(--color-cyber-cyan)] shadow-[0_0_10px_var(--color-cyber-cyan)] animate-pulse"></span>
        <span class="status-text">{{ statusText }}</span>
      </div>
      
      <!-- Controls -->
      <div class="flex items-center gap-4 border-l-2 border-[var(--border-panel)] pl-5">
        <!-- Language Switcher -->
        <button @click="toggleLang" class="text-[var(--text-primary)] hover:text-[var(--color-cyber-cyan)] transition-colors cursor-pointer px-2 text-base font-extrabold">
          {{ locale === 'en' ? 'EN' : 'ZH' }}
        </button>
        
        <!-- Dark Mode Toggle -->
        <button @click="toggleDark()" class="text-[var(--text-primary)] hover:text-[var(--color-cyber-cyan)] cursor-pointer transition-colors px-2">
          <i :class="isDark ? 'ri-moon-line' : 'ri-sun-line'" class="text-2xl font-bold"></i>
        </button>

        <!-- GitHub Link -->
        <a href="https://github.com/ZiweikWang/AnimeDL2M" target="_blank" class="text-[var(--text-primary)] hover:text-[var(--color-cyber-cyan)] cursor-pointer transition-colors px-2 ml-2 flex items-center gap-1 text-sm font-bold border border-transparent hover:border-[var(--border-panel)] p-1 rounded">
          <i class="ri-github-fill text-2xl"></i> <span class="hidden sm:inline">{{ $t('navbar.github') }}</span>
        </a>
      </div>
    </div>
  </header>
</template>

<script setup>
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import { useDark, useToggle } from '@vueuse/core';

const props = defineProps({
  analysisState: {
    type: String,
    default: 'awaiting'
  }
});

const { t, locale } = useI18n();

const isDark = useDark({
  selector: 'html',
  attribute: 'class',
  valueDark: 'dark',
  valueLight: '',
});
const toggleDark = useToggle(isDark);

const toggleLang = () => {
  locale.value = locale.value === 'en' ? 'zh' : 'en';
};

const statusText = computed(() => {
  return t(`status.${props.analysisState}`);
});
</script>
