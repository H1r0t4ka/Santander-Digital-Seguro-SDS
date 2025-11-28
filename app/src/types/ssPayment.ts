import { z } from "zod";

// We're keeping a simple non-relational schema here.
// IRL, you will have a schema for your data models.
export const itemSchema = z.object({
  id: z.string(),
  number: z.string(),
  value: z.number().multipleOf(0.01),
  employerId: z.string(),
  periodEps: z.date(),
  periodAfp: z.date(),

  employer: z.object({
    name: z.string(),
  }),
});

export type Item = z.infer<typeof itemSchema>;
