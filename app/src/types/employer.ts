import { z } from "zod";

// We're keeping a simple non-relational schema here.
// IRL, you will have a schema for your data models.
export const itemSchema = z.object({
  id: z.string(),
  nit: z.string(),
  name: z.string(),
  address: z.string(),
  phone: z.string(),
  last_day_ss: z.string(),
});

export type Item = z.infer<typeof itemSchema>;
