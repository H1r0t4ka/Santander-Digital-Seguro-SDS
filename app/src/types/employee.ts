import { z } from "zod";

// We're keeping a simple non-relational schema here.
// IRL, you will have a schema for your data models.
export const itemSchema = z.object({
  id: z.string(),
  doc_type: z.string(),
  doc_number: z.string(),
  name: z.string(),
  last_name: z.string(),
  date_of_birth: z.date(),
  address: z.string(),
  employer_id: z.string(),
  phone: z.string(),
  status: z.boolean(),
  contract_date: z.date(),
  contract_end: z.date(),
  employer: z.object({
    name: z.string(),
  }),
  employeeByBranchOffice: z.array(
    z.object({
      branchOffice: z.object({
        name: z.string(),
      }),
    })
  ),
});

export type Item = z.infer<typeof itemSchema>;
