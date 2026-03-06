import { createI18n } from 'vue-i18n';

const messages = {
    en: {
        navbar: {
            title: 'ANIFORGERY_LAB',
            version: 'SYS_V2.0 // VUE_NODE',
            github: 'Source Code'
        },
        status: {
            awaiting: 'AWAITING_INPUT',
            ready: 'READY_FOR_ANALYSIS',
            scanning: 'SCANNING_MATRIX...',
            threat: 'THREAT_DETECTED',
            suspicious: 'SUSPICIOUS_ANOMALY',
            safe: 'VERIFIED_SAFE',
            error: 'CONNECTION_LOST'
        },
        upload: {
            title: '[VISUAL_INPUT_MATRIX]',
            purge: 'PURGE',
            promptMain: 'INITIATE UPLOAD SEQUENCE',
            promptSub: 'CLICK OR DRAG ANIME SOURCE FILE HERE'
        },
        data: {
            title: '[DIAGNOSTIC_DATA_LINK]',
            executeBtn: 'EXECUTE ANALYSIS',
            reportLabel: '/// ANALYSIS_THREAT_REPORT',
            reportWait: 'Waiting for diagnostic telemetry...',
            probeTitle: 'PIXEL_CONFIDENCE_PROBE',
            probeDesc: 'Click anywhere on the heatmap matrix to sample local AI-tampering probability.',
            apiConfig: 'REMOTE_API_NODE',
            apiConfigTitle: '[CONFIGURE_REMOTE_NODE]',
            apiConfigDesc: 'Please enter the HuggingFace Spaces API Endpoint for ANIFORGERY_LAB backend:',
            saveConfig: 'SAVE CONFIG'
        },
        report: {
            dangerWhole: '🚨 DANGER: Highly suspected WHOLE Image AI Generation!\n',
            dangerPart: '🚨 DANGER: Highly suspected PARTIAL AI Inpainting!\n',
            warning: '⚠️ WARNING: AI generation traces detected.\n',
            safe: '✅ SAFE: Highly likely authentic human hand-drawn art.\n',
            clsProb: '▸ AI Gen Probability: ',
            maskProb: '▸ Local Tamper Confidence: ',
            source: '▸ Detected Source Model: ',
            error: 'CRITICAL ERROR: Failed to execute analysis.\n\nDetails: '
        }
    },
    zh: {
        navbar: {
            title: 'ANIFORGERY_LAB',
            version: 'SYS_V2.0 // VUE_NODE',
            github: '查看源码'
        },
        status: {
            awaiting: '等待输入',
            ready: '准备就绪',
            scanning: '扫描矩阵...',
            threat: '发现威胁',
            suspicious: '可疑异常',
            safe: '验证安全',
            error: '连接中断'
        },
        upload: {
            title: '[视觉输入矩阵]',
            purge: '清除数据',
            promptMain: '启动上传序列',
            promptSub: '点击或拖拽动漫原图至此区域'
        },
        data: {
            title: '[诊断数据链路]',
            executeBtn: '执行鉴证分析',
            reportLabel: '/// 威胁分析报告',
            reportWait: '等待诊断遥测...',
            probeTitle: '像素级探针',
            probeDesc: '点击热力图矩阵的任意位置，采样局部 AI 篡改概率。',
            apiConfig: '远程 API 节点',
            apiConfigTitle: '[配置远程节点]',
            apiConfigDesc: '请输入用于 ANIFORGERY_LAB 后端的 HuggingFace Spaces API 端点：',
            saveConfig: '保存配置'
        },
        report: {
            dangerWhole: '🚨 危险：高度疑似 AI 整图生成！\n',
            dangerPart: '🚨 危险：高度疑似局部 AI 重绘！\n',
            warning: '⚠️ 可疑：检测到 AI 生成痕迹。\n',
            safe: '✅ 安全：大概率为真实人类手绘作品。\n',
            clsProb: '▸ AI生成概率: ',
            maskProb: '▸ 局部篡改置信度: ',
            source: '▸ 判定模型来源: ',
            error: '严重错误：分析执行失败。\n\n详情: '
        }
    }
};

const i18n = createI18n({
    locale: 'en', // default
    fallbackLocale: 'zh',
    messages,
});

export default i18n;
