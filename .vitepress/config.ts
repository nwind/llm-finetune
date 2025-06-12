import {defineConfig} from 'vitepress';

// https://vitepress.dev/reference/site-config
export default defineConfig({
  lang: 'zh',
  title: '大模型微调与部署指南',
  description: '',
  markdown: {
    math: true
  },
  base: '/llm-finetune/',
  appearance: {
    // @ts-expect-error not fully supported yet
    initialValue: 'light'
  },
  themeConfig: {
    logo: './images/logo.png',
    // https://vitepress.dev/reference/default-theme-config
    nav: [{text: '首页', link: '/'}],
    sidebar: [
      {
        text: '',
        items: [
          {text: '前言', link: '/intro.html'},
          {text: '大模型微调入门', link: '/start.html'},
          {text: '大模型基础', link: '/basic.html'},
          {text: '微调训练', link: '/sft.html'},
          {text: '对齐训练', link: '/rl.html'},
          {text: '训练数据构造及管理', link: '/data.html'},
          {text: '评估', link: '/eval.html'},
          {text: '微调实践', link: '/practice.html'},
          {text: '模型部署', link: '/deploy.html'},
          {text: '附录', link: '/appendix.html'},
          {text: '拓展阅读', link: '/extend.html'}
        ]
      }
    ],

    socialLinks: [
      {icon: 'github', link: 'https://github.com/nwind/llm-finetune'}
    ],

    search: {
      provider: 'local',
      options: {
        translations: {
          button: {
            buttonText: '搜索',
            buttonAriaLabel: '搜索'
          },
          modal: {
            displayDetails: '显示详细列表',
            resetButtonTitle: '重置搜索',
            backButtonTitle: '关闭搜索',
            noResultsText: '没有结果',
            footer: {
              selectText: '选择',
              selectKeyAriaLabel: '输入',
              navigateText: '导航',
              navigateUpKeyAriaLabel: '上箭头',
              navigateDownKeyAriaLabel: '下箭头',
              closeText: '关闭',
              closeKeyAriaLabel: 'esc'
            }
          }
        }
      }
    },

    editLink: {
      pattern: 'https://github.com/nwind/llm-finetune/edit/main/:path'
    }
  }
});
