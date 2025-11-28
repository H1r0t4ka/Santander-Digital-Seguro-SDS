import { z } from "zod";
import { createTRPCRouter, publicProcedure } from "~/server/api/trpc";
import * as fs from "fs";
import OpenAI from "openai";
const myHeaders = new Headers();
myHeaders.append("Content-Type", "application/json");

// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
const openai = new OpenAI();

const ex =
  '<div><p><strong>Resumen:</strong> Según la Ley 769 de 2002 del Código Nacional de Tránsito Terrestre, las normas aplican en todo el territorio colombiano y regulan la circulación de todos los usuarios de la vía pública. Según la Constitución Política, todos los colombianos tienen derecho a moverse libremente por el país, pero están sujetos a la intervención y regulación de las autoridades para garantizar la seguridad de los habitantes. En función de esto, durante ciertos días y horas se puede restringir la circulación de vehículos mediante el sistema de pico y placa. </p> <br/> <p><strong>Explicación:</strong> La pregunta pregunta si uno puede salir durante el pico y placa, esto depende de las regulaciones de cada ciudad. Estas regulaciones son definidas de acuerdo con el Código Nacional de Tránsito Terrestre y la Constitución Política, los cuales establecen que las autoridades pueden intervenir y regular la circulación para garantizar la seguridad. Por lo tanto, si estás en un horario restringido por pico y placa, no estás autorizado para circular con tu vehículo.</p> <br/> <p>En una cita directa de la Ley 769 de 2002, se expone: "<strong>En desarrollo de lo dispuesto por el artículo 24 de la Constitución Política, todo colombiano tiene derecho a circular libremente por el territorio nacional, pero está sujeto a la intervención y reglamentación de las autoridades para garantía de la seguridad y comodidad de los habitantes</strong>".</p></div>';

export const sdsRouter = createTRPCRouter({
  generateQuery: publicProcedure
    .input(z.object({ prompt: z.string() }))
    .mutation(async ({ input }) => {
      try {
        // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access
        const promptString = input.prompt;
        // console.log(promptString);
        if (!promptString) {
          return new Response("you need a prompt", { status: 400 });
        }

        let text = "";
        const filePath = "leytransito.txt";
        text = await fs.promises.readFile(filePath, "utf-8");

        // eslint-disable-next-line @typescript-eslint/restrict-plus-operands
        // const messageP = `Necesito responder la siguiente pregunta: ${promptString} con un breve resumen en palabras claras y una citación del artículo, debes responder a partir de este contexto: ${text}`; //text + " " + promptString;
        const messageP = `Necesito responder la siguiente pregunta: ${promptString} con un breve resumen del artículo, luego en otro párrafo explicar en  palabras claras el uso del articulo respecto a la pregunta y en otro párrafo una citación del artículo, esta respuesta debe ser en formato HTML con sus respectivas negritas y salto de líneas al inicio y al final de cada parrafo, ejemplo: ${ex}, debes responder a partir de este contexto: ${text}`;

        // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-call
        const completion = await openai.chat.completions.create({
          model: "gpt-4",
          messages: [{ role: "system", content: messageP }],
        });

        // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
        // console.log(completion);

        // eslint-disable-next-line @typescript-eslint/no-unsafe-return, @typescript-eslint/no-unsafe-member-access
        return completion.choices[0]?.message.content || "";
      } catch (error) {
        console.log(error);
      }
    }),
});
