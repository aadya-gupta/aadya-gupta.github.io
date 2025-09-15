import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const blog = defineCollection({
  loader: glob({ pattern: '**/[^_]*.{md,mdx}', base: "./src/content/blog" }),
  schema: z.object({
    title: z.string(),
    date: z.date(),
    description: z.string().optional(),
  }),
});

const lab = defineCollection({
  loader: glob({ pattern: '**/[^_]*.{md,mdx}', base: "./src/content/lab" }),
  schema: z.object({
    title: z.string(),
    date: z.date(),
    status: z.enum(['Hypothesis', 'Experiment', 'Result', 'Refinement']),
    tags: z.array(z.string()).optional(),
  }),
});

export const collections = { blog, lab };
