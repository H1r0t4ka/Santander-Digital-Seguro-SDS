import { z } from "zod";

// We're keeping a simple non-relational schema here.
// IRL, you will have a schema for your data models.
export const itemSchema = z.object({
  id: z.string(),
  employee_id: z.string(),
  employer_id: z.string(),
  arl: z.string(),
  risk: z.string(),
  arl_date: z.date(),
  eps: z.string(),
  eps_date: z.date(),
  afp: z.string(),
  afp_date: z.date(),
  state: z.boolean(),
  end_date: z.date(),

  employer: z.object({
    name: z.string(),
  }),
  employeeInfo: z.string().optional(),
  employee: z.object({
    name: z.string(),
    last_name: z.string(),
    doc_number: z.string(),
  }),
});

export type Item = z.infer<typeof itemSchema>;
