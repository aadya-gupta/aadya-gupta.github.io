// @ts-check
import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// https://astro.build/config
export default defineConfig({
  // Your GitHub Pages custom or username URL:
  site: 'https://aadya-gupta.github.io',
  
  // IF the repo name is "my-research-site" (a project repo):
  // base: '/my-research-site',
  // IF the repo name is "aadya-gupta.github.io" (a personal user site), OMIT the `base` line or set base: '/'
  
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
  vite: {
    plugins: [tailwindcss()],
  },
});