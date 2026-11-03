// @ts-check
import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import getReadingTime from 'reading-time';
import { toString } from 'mdast-util-to-string';

export function remarkReadingTime() {
  return function (tree, { data }) {
    const text = toString(tree);
    const readingTime = getReadingTime(text);
    data.astro.frontmatter.readingTime = `${Math.ceil(readingTime.minutes)} MIN READ`;
  };
}

export default defineConfig({
  site: 'https://aadya-gupta.github.io',

  markdown: {
    remarkPlugins: [remarkMath, remarkReadingTime],
    rehypePlugins: [rehypeKatex],
  },
  vite: {
    plugins: [tailwindcss()],
  },
});