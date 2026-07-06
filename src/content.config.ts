import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

// 1. Blog Collection
const blog = defineCollection({
  loader: glob({ pattern: '**/[^_]*.{md,mdx}', base: "./src/content/blog" }),
  schema: z.object({
    title: z.string(),
    date: z.date(),
    description: z.string().optional(),
  }),
});

// 2. Lab Collection
const lab = defineCollection({
  loader: glob({ pattern: '**/[^_]*.{md,mdx}', base: "./src/content/lab" }),
  schema: z.object({
    title: z.string(),
    date: z.date(),
    status: z.enum(['Hypothesis', 'Experiment', 'Result', 'Refinement']),
    tags: z.array(z.string()).optional(),
  }),
});

// 3. Photos Collection
const photos = defineCollection({
  type: 'content',
  schema: ({ image }) => z.object({
    cover: image(),
    alt: z.string(),
  }),
});

// 4. Export them ALL together in one single object
export const collections = { blog, lab, photos };