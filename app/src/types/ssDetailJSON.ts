import { z } from "zod";

export interface PaymentJSON {
  textbox125: string;
  Tipo_Id_3: string;
  Textbox4: string;
  Textbox19: string;
  novedad_ing_letra3: string;
  novedad_ret_letra3: string;
  novedad_taa_letra3: string;
  Textbox57: string;
  novedad_tda_letra6: string;
  novedad_tda_letra7: string;
  novedad_vsp_letra3: string;
  novedad_correccion_letra3: string;
  novedad_vst_letra3: string;
  novedad_sln_letra3: string;
  novedad_ige_letra3: string;
  novedad_lma_letra3: string;
  novedad_vac_letra3: string;
  novedad_avp_letra3: string;
  novedad_vct_letra3: string;
  novedad_irp_letra3: string;
  Textbox60: string;
  textbox12: string;
  Textbox61: string;
  ibc_pension3: string;
  total_pension3: string;
  dias_salud3: string;
  Textbox62: string;
  ibc_salud3: string;
  total_salud3: string;
  dias_parafiscales3: string;
  Textbox64: string;
  ibc_parafiscales3: string;
  cotizacion_ccf3: string;
  dias_riesgos3: string;
  Textbox65: string;
  ibc_riesgos3: string;
  tarifa_riesgos3: string;
  cotizacion_riesgos3: string;
  Textbox66: string;
  ibc_senaicbf3: string;
  cotizacion_sena3: string;
  Textbox78: string;
  Textbox255: string;
}

// We're keeping a simple non-relational schema here.
// IRL, you will have a schema for your data models.
export const itemSchema = z.object({
  id: z.string(),
  // value: z.number().multipleOf(0.01),

  ssPayment: z.string(),
  employeeId: z.string(),
  employeeInfo: z.string(),
  nvIng: z.boolean(),
  nvRet: z.boolean(),
  nvTde: z.boolean(),
  nvTae: z.boolean(),
  nvTdp: z.boolean(),
  nvTap: z.boolean(),
  nvVsp: z.boolean(),
  nvCor: z.boolean(),
  nvVst: z.boolean(),
  nvSln: z.boolean(),
  nvIge: z.boolean(),
  nvLma: z.boolean(),
  nvVac: z.boolean(),
  nvAvp: z.boolean(),
  nvVct: z.boolean(),
  nvIrl: z.boolean(),
  nvVip: z.boolean(),
  pnCode: z.string(),
  pnDay: z.number(),
  pnIbc: z.number(),
  pbContribution: z.string(),
  slCode: z.string(),
  slDay: z.number(),
  slIbc: z.number(),
  slContribution: z.number(),
  ccCode: z.string(),
  ccDay: z.number(),
  ccIbc: z.number(),
  ccContribution: z.number(),
  rsCode: z.string(),
  rsDay: z.number(),
  rsIbc: z.number(),
  rsContribution: z.number(),
  pfDay: z.number(),
  pfIbc: z.number(),
  pfContribution: z.number(),
  pfSena: z.boolean(),
  total: z.number(),
  isAffiliated: z.boolean(),

  // employer: z.object({
  //   name: z.string(),
  // }),
});

export type Item = z.infer<typeof itemSchema>;
