import { z } from "zod";

// We're keeping a simple non-relational schema here.
// IRL, you will have a schema for your data models.
export const taskSchema = z.object({
  id: z.string(),
  device_code: z.string(),
  ip: z.string(),
  branch_offioce_id: z.string(),
  branchOffice: z.object({
    name: z.string(),
  }),
});

export type Task = z.infer<typeof taskSchema>;
