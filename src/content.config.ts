import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const blog = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/blog' }),
  schema: z.object({
    title: z.string(),
    description: z.string().optional(),
    date: z.coerce.date().optional(),
    pubDate: z.coerce.date().optional(),
    tags: z.array(z.string()).default([]),
    readingTime: z.string().optional(),
    featured: z.boolean().default(false),
  }),
});

const lab = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/lab' }),
  schema: z.object({
    id_tag: z.string().default("EXP_00"),
    title: z.string(),
    category: z.string().default("EXPERIMENT"),
    date: z.coerce.date().default(() => new Date()),
    stack: z.array(z.string()).default([]),
    metrics: z.record(z.string(), z.string()).optional(),
    repo_url: z.string().optional(),
  }),
});

const photos = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/photos' }),
  schema: z.object({
    title: z.string(),
    caption: z.string().optional(),
    image: z.string(),
    date: z.coerce.date(),
    location: z.string().optional(),
    camera: z.string().default("Sony ZV-E10"),
    lens: z.string().optional(),
    focal_length: z.string().optional(),
    aperture: z.string().optional(),
    shutter: z.string().optional(),
    iso: z.string().optional(),
  }),
});

export const collections = { blog, lab, photos };