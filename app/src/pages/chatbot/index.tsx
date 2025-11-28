/* eslint-disable @typescript-eslint/no-unsafe-return */
/* eslint-disable @typescript-eslint/no-unsafe-argument */
/* eslint-disable @typescript-eslint/no-unsafe-member-access */
/* eslint-disable @typescript-eslint/no-unsafe-assignment */
import { useState, useEffect, useRef } from "react";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import { Package2, Bot, Sparkles, Send, X } from "lucide-react";

import { Button } from "~/components/ui/button";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from "~/components/ui/card";
import { Badge } from "~/components/ui/badge";
import { Progress } from "~/components/ui/progress";

import { useSession } from "next-auth/react";
import { Textarea } from "~/components/ui/textarea";
import { GoogleGenAI } from "@google/genai";
import SpeechRecognitionButton, {
  type SpeechRecognitionButtonRef,
} from "~/components/SpeechRecognitionButton";
import { Header } from "~/components/header";

// Custom vector store implementation
interface Document {
  pageContent: string;
  metadata: Record<string, any>;
}

class MemoryVectorStore {
  private documents: Document[] = [];

  addDocuments(docs: Document[]): void {
    this.documents.push(...docs);
  }

  similaritySearch(query: string, k = 4): Promise<Document[]> {
    // Simple keyword-based similarity search for demo purposes
    const queryLower = query.toLowerCase();
    const scoredDocs = this.documents.map((doc) => {
      const contentLower = doc.pageContent.toLowerCase();
      let score = 0;

      // Simple keyword matching
      if (contentLower.includes(queryLower)) score += 3;

      // Check for individual word matches
      const queryWords = queryLower.split(/\s+/);
      queryWords.forEach((word) => {
        if (word.length > 3 && contentLower.includes(word)) {
          score += 1;
        }
      });

      return { doc, score };
    });

    // Sort by score and return top k
    const results = scoredDocs
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
      .map((item) => item.doc);

    return Promise.resolve(results);
  }
}

// Interfaces for each API response type
interface HurtoData {
  departamento: string;
  municipio: string;
  codigo_dane: string;
  armas_medios: string;
  fecha_hecho: string;
  genero: string;
  grupo_etario: string;
  cantidad: string;
  tipo_de_hurto?: string;
}

interface ViolenciaIntrafamiliarData {
  departamento: string;
  municipio: string;
  codigo_dane: string;
  armas_medios: string;
  fecha_hecho: string;
  genero: string;
  grupo_etario: string;
  cantidad: string;
}

interface DelitosSexualesData {
  departamento: string;
  municipio: string;
  codigo_dane: string;
  armas_medios: string;
  fecha_hecho: string;
  genero: string;
  grupo_etario: string;
  cantidad: string;
  delito?: string;
}

// Interface for grouped data
interface GroupedMunicipioData {
  count: number;
  totalCantidad: number;
  municipio: string;
}

export default function Chatbot() {
  const { data: sessionData } = useSession();
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [answer, setAnswer] = useState(true);
  const [answerText, setAnswerText] = useState("");
  const speechRecognitionRef = useRef<SpeechRecognitionButtonRef>(null);
  const [vectorStores, setVectorStores] = useState({
    hurto: null as MemoryVectorStore | null,
    violencia: null as MemoryVectorStore | null,
    sexual: null as MemoryVectorStore | null,
  });
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  // Initialize vector stores on component mount
  useEffect(() => {
    void fetchDataAndCreateDocuments();
  }, []);

  // Function to fetch data and create documents for vector stores
  async function fetchDataAndCreateDocuments() {
    try {
      console.log("Initializing vector stores with Santander crime data...");

      // Create new vector store instances
      const hurtoStore = new MemoryVectorStore();
      const violenciaStore = new MemoryVectorStore();
      const sexualStore = new MemoryVectorStore();

      // Fetch data from each API and create documents
      const apiUrls = {
        hurto:
          "https://www.datos.gov.co/resource/d4fr-sbn2.json?$where=departamento='SANTANDER'",
        violencia:
          "https://www.datos.gov.co/resource/vuyt-mqpw.json?$where=departamento='SANTANDER'",
        sexual:
          "https://www.datos.gov.co/resource/fpe5-yrmw.json?$where=departamento='SANTANDER'",
      };

      // Fetch and process each data type
      for (const [dataType, url] of Object.entries(apiUrls)) {
        try {
          console.log(`Fetching ${dataType} data...`);
          const response = await fetch(url);
          if (!response.ok) {
            console.warn(
              `Failed to fetch ${dataType} data: ${response.status}`
            );
            continue;
          }

          let data: any[] = await response.json();

          // Limit to first 1000 records for performance
          if (data.length > 1000) {
            data = data.slice(0, 1000);
          }

          // Create documents for vector store
          const documents = data.map((item) => ({
            pageContent: JSON.stringify(item, null, 2),
            metadata: { ...item, dataType },
          }));

          // Add documents to appropriate vector store
          if (dataType === "hurto") {
            hurtoStore.addDocuments(documents);
          } else if (dataType === "violencia") {
            violenciaStore.addDocuments(documents);
          } else if (dataType === "sexual") {
            sexualStore.addDocuments(documents);
          }

          console.log(
            `Added ${documents.length} documents to ${dataType} vector store`
          );
        } catch (error) {
          console.error(`Error processing ${dataType} data:`, error);
        }
      }

      // Update vector stores state
      setVectorStores({
        hurto: hurtoStore,
        violencia: violenciaStore,
        sexual: sexualStore,
      });

      console.log("Vector stores initialized successfully");
    } catch (error) {
      console.error("Error initializing vector stores:", error);
    }
  }

  async function handleGenerateQuery() {
    // Stop speech recognition if it's active
    speechRecognitionRef.current?.stopListening();
    setLoading(true);

    try {
      const ai = new GoogleGenAI({
        apiKey: process.env.NEXT_PUBLIC_GEMINI_API_KEY,
      });

      // Step 1: Determine which vector store to use based on query content
      let dataType: "hurto" | "violencia" | "sexual";
      let vectorStore: MemoryVectorStore | null = null;

      // Determine vector store based on query keywords
      const queryLower = prompt.toLowerCase();

      if (queryLower.includes("hurto") || queryLower.includes("robo")) {
        dataType = "hurto";
        vectorStore = vectorStores.hurto;
      } else if (
        queryLower.includes("violencia") ||
        queryLower.includes("intrafamiliar")
      ) {
        dataType = "violencia";
        vectorStore = vectorStores.violencia;
      } else if (
        queryLower.includes("sexual") ||
        queryLower.includes("delito sexual") ||
        queryLower.includes("abuso")
      ) {
        dataType = "sexual";
        vectorStore = vectorStores.sexual;
      } else {
        // Default to hurto vector store if unsure (more common queries)
        dataType = "hurto";
        vectorStore = vectorStores.hurto;
      }

      // Check if vector store is initialized
      if (!vectorStore) {
        throw new Error(
          "Los datos aún se están cargando. Por favor, espera un momento e intenta nuevamente."
        );
      }

      // Step 2: Generate WHERE clause from user query using Gemini
      const whereClausePrompt = `
Based on the user query: "${prompt}"

Generate a SQL-like WHERE clause for the Colombian crime data API with the following fields:
- departamento (department)
- municipio (municipality) 
- codigo_dane (DANE code)
- armas_medios (weapons/means)
- fecha_hecho (incident date in format DD/MM/YYYY)
- genero (gender: FEMENINO/MASCULINO)
- grupo_etario (age group)
- cantidad (quantity)
${
  dataType === "hurto"
    ? "- tipo_de_hurto (theft type - ONLY for theft crimes)"
    : ""
}
${dataType === "sexual" ? "- delito (crime type - for sexual crimes)" : ""}

IMPORTANT: Only include fields that are available in the specific API:
${
  dataType === "hurto"
    ? "• Hurto API fields: departamento, municipio, codigo_dane, armas_medios, fecha_hecho, genero, grupo_etario, cantidad, tipo_de_hurto"
    : ""
}
${
  dataType === "violencia"
    ? "• Violencia Intrafamiliar API fields: departamento, municipio, codigo_dane, armas_medios, fecha_hecho, genero, grupo_etario, cantidad"
    : ""
}
${
  dataType === "sexual"
    ? "• Delitos Sexuales API fields: departamento, municipio, codigo_dane, armas_medios, fecha_hecho, genero, grupo_etario, cantidad, delito"
    : ""
}

CRITICAL: You MUST only work with data from Santander department. If the user requests data from other departments, 
return an error message explaining that only Santander data is available.

If the query is general or doesn't specify filters, return an empty string.
`;

      const whereResponse = await ai.models.generateContent({
        model: "gemini-2.5-flash",
        contents: whereClausePrompt,
      });

      let whereClause = whereResponse.text?.trim() || "";

      console.log("=== WHERE CLAUSE RESPONSE ===");
      console.log("Where Clause:", whereClause);

      // Check if user is requesting non-Santander data
      const santanderRegex = /departamento\s*[=!]\s*['\"]SANTANDER['\"]/i;
      const otherDepartmentRegex =
        /departamento\s*[=!]\s*['\"](?!SANTANDER)[A-Z\s]+['\"]/i;

      // List of valid Santander municipalities
      const santanderMunicipios = [
        "BUCARAMANGA",
        "FLORIDABLANCA",
        "PIEDECUESTA",
        "GIRÓN",
        "BARRANCABERMEJA",
        "SAN GIL",
        "SOCORRO",
        "CHARALÁ",
        "CIMITARRA",
        "PUERTO WILCHES",
        "SABANA DE TORRES",
        "RIONEGRO",
        "ZAPATOCA",
        "SAN VICENTE DE CHUCURÍ",
        "LANDÁZURI",
        "EL PLAYÓN",
        "CAPITANEJO",
        "ENCISO",
        "CARCASÍ",
        "MACARAVITA",
        "MÁLAGA",
        "CONFINES",
        "GUACA",
        "GUADALUPE",
        "CEPITÁ",
        "CHIPATÁ",
        "COROMORO",
        "CURITÍ",
        "EL GUACAMAYO",
        "EL PEÑÓN",
        "GÁMBITA",
        "JESÚS MARÍA",
        "LA BELLEZA",
        "LA PAZ",
        "LEBRIJA",
        "LOS SANTOS",
        "MATANZA",
        "MOGOTES",
        "MOLAGAVITA",
        "OCAMONTE",
        "OIBA",
        "ONZAGA",
        "PALMAR",
        "PÁRAMO",
        "PINCHOTE",
        "PUENTE NACIONAL",
        "PUERTO PARRA",
        "PUERTO RICO",
        "SAN ANDRÉS",
        "SAN JOAQUÍN",
        "SAN JOSÉ DE MIRANDA",
        "SAN MIGUEL",
        "SANTA BÁRBARA",
        "SANTA HELENA DEL OPÓN",
        "SIMACOTA",
        "SUAITA",
        "SUCRE",
        "SURATÁ",
        "TONA",
        "VALLE DE SAN JOSÉ",
        "VÉLEZ",
        "VETAS",
        "VILLANUEVA",
        "WILCHES",
        "BETULIA",
        "CABRERA",
        "CALIFORNIA",
        "CONTRATACIÓN",
        "GALÁN",
        "GUAPOTÁ",
        "HATO",
        "JORDÁN",
        "LA PLATA DE ORO",
        "PALMAS DEL SOCORRO",
        "PIE DE CUESTA",
        "SAN BENITO",
        "SANTA ROSA",
        "TOGÜÍ",
        "ALBANIA",
        "ARATOCA",
        "BARICHARA",
        "CERRITO",
        "CHARTAS",
        "CHIMA",
        "CINCO DE AGOSTO",
        "CUCUTILLA",
        "EL CARMEN",
        "EL TARRA",
        "ENTRERRIOS",
        "FLORIÁN",
        "GACHANTIVÁ",
        "GAMEZA",
        "GÜICÁN",
        "LABRANZAGRANDE",
        "LA UVITA",
        "MARIPÍ",
        "MIRAFLORES",
        "MONGUÍ",
        "MONGUA",
        "MOTAVITA",
        "NOBSA",
        "NUEVO COLÓN",
        "PAIPA",
        "PASTO",
        "PAZ DE RÍO",
        "PESCA",
        "QUÍPAMA",
        "RAMIRIQUÍ",
        "RÁQUIRA",
        "SABOYÁ",
        "SAMACÁ",
        "SAN EDUARDO",
        "SAN LUIS DE GACENO",
        "SAN MATEO",
        "SAN PABLO DE BORBUR",
        "SANTANA",
        "SATIVANORTE",
        "SATIVASUR",
        "SIACHOQUE",
        "SOATÁ",
        "SOGAMOSO",
        "SORA",
        "SORACÁ",
        "SOTAQUIRÁ",
        "SUSACÓN",
        "SUTAMARCHÁN",
        "TASCO",
        "TIBANÁ",
        "TIBASOSA",
        "TINJACÁ",
        "TIPACOQUE",
        "TOCA",
        "TOGÜÍ",
        "TOPAGÁ",
        "TOTA",
        "TUNJA",
        "TUNUNGUÁ",
        "TURMEQUÉ",
        "TUTA",
        "TUTAZÁ",
        "UMBITA",
        "VENTAQUEMADA",
        "VILLA DE LEYVA",
        "VIRACACHÁ",
        "ZETAQUIRA",
      ];

      // Validate that user is not requesting other departments
      if (otherDepartmentRegex.test(whereClause)) {
        throw new Error(
          "Solo estamos trabajando con datos del departamento de Santander. Por favor, formula tu consulta relacionada con el departamento de Santander y sus municipios (Bucaramanga, Floridablanca, Piedecuesta, Girón)."
        );
      }

      // Validate that user is not requesting municipalities outside Santander
      const municipioRegex = /municipio\s*[=!]\s*['\"]([A-Z\sÁÉÍÓÚÑ]+)['\"]/gi;
      let match;
      while ((match = municipioRegex.exec(whereClause)) !== null) {
        if (match[1]) {
          const requestedMunicipio = match[1].toUpperCase().trim();
          if (!santanderMunicipios.includes(requestedMunicipio)) {
            throw new Error(
              `Solo tenemos información de municipios del departamento de Santander. El municipio "${requestedMunicipio}" no pertenece a Santander. Por favor, consulta por municipios de Santander como Bucaramanga, Floridablanca, Piedecuesta o Girón.`
            );
          }
        }
      }

      // Always filter by Santander department
      if (whereClause) {
        if (!santanderRegex.test(whereClause)) {
          whereClause = `departamento='SANTANDER' AND ${whereClause}`;
        }
      } else {
        whereClause = "departamento='SANTANDER'";
      }

      // Remove tipo_de_hurto filter if not in hurto API
      if (dataType !== "hurto" && whereClause.includes("tipo_de_hurto")) {
        whereClause = whereClause.replace(
          /tipo_de_hurto\s*[=!]\s*['\"][^'\"]*['\"]/gi,
          ""
        );
        whereClause = whereClause.replace(/AND\s+AND/gi, "AND").trim();
        if (whereClause.endsWith("AND")) {
          whereClause = whereClause.slice(0, -3).trim();
        }
      }

      // Remove delito filter if not in sexual API
      if (dataType !== "sexual" && whereClause.includes("delito")) {
        whereClause = whereClause.replace(
          /delito\s*[=!]\s*['\"][^'\"]*['\"]/gi,
          ""
        );
        whereClause = whereClause.replace(/AND\s+AND/gi, "AND").trim();
        if (whereClause.endsWith("AND")) {
          whereClause = whereClause.slice(0, -3).trim();
        }
      }

      // Step 3: Search for relevant documents in the vector store
      console.log("Searching vector store for query:", prompt);

      // Use the user's original query to search the vector store
      const relevantDocuments = await vectorStore.similaritySearch(prompt, 10);

      // Extract the actual crime data from the documents
      const crimeData = relevantDocuments.map((doc) => doc.metadata) as any[];

      console.log("=== VECTOR STORE SEARCH RESULTS ===");
      console.log("Query:", prompt);
      console.log("Found documents:", relevantDocuments.length);
      console.log("Crime data:", crimeData);

      // Process and group data for better analysis
      let processedData: unknown = crimeData;

      // If we have a lot of data, group it for better analysis
      if (crimeData.length > 10) {
        // Group by municipality for summary statistics
        const groupedByMunicipio = crimeData.reduce((acc, item) => {
          const municipio = item.municipio;
          if (!acc[municipio]) {
            acc[municipio] = {
              count: 0,
              totalCantidad: 0,
              municipio: municipio,
            };
          }
          acc[municipio].count += 1;
          acc[municipio].totalCantidad += parseInt(item.cantidad) || 1;
          return acc;
        }, {} as Record<string, GroupedMunicipioData>);

        // Convert to array and take top 5 municipalities
        const municipiosArray: GroupedMunicipioData[] =
          Object.values(groupedByMunicipio);
        const topMunicipios = municipiosArray
          .sort((a, b) => b.totalCantidad - a.totalCantidad)
          .slice(0, 5);

        processedData = topMunicipios;
      }

      // Step 3: Generate final response using the retrieved data
      const finalResponsePrompt = `
User query: "${prompt}"

Data retrieved from Colombian crime database${
        crimeData.length > 10 ? " (summarized top 5 municipalities)" : ""
      }:
${JSON.stringify(processedData, null, 2)}

Generate a comprehensive response in Spanish that answers the user's question based on the data. 
Include statistics, trends, or insights from the data. If no data was found, explain this to the user.

Response should be in Spanish and professional tone.

IMPORTANT: Do NOT include any signatures, closing remarks like "Atentamente", "Saludos", or any personal/entity names at the end of the response. Just provide the direct answer to the user's query.

If there is a large amount of data, provide a summary with the most relevant statistics.
`;

      const finalResponse = await ai.models.generateContent({
        model: "gemini-2.5-flash",
        contents: finalResponsePrompt,
      });

      console.log("=== FINAL RESPONSE ===");
      console.log("Final Response:", finalResponse.text);
      console.log("Full Final Response Object:", finalResponse);

      setAnswerText(
        finalResponse.text || "No se encontraron datos para su consulta."
      );
      setLoading(false);
      setAnswer(false);
    } catch (error) {
      console.error("Error generating content:", error);

      let errorMessage =
        "No pudimos generar una respuesta con la información disponible. Por favor intenta con una consulta más específica relacionada con el departamento de Santander y sus municipios (Bucaramanga, Floridablanca, Piedecuesta, Girón).\n\nPuedes intentar con preguntas como:\n• ¿Cuántos casos de hurto hubo en Bucaramanga en 2023?\n• ¿Qué tipos de delitos sexuales se reportaron en Floridablanca?\n• Estadísticas de violencia intrafamiliar en Piedecuesta";

      if (error instanceof Error) {
        if (
          error.message.includes(
            "Solo estamos trabajando con datos del departamento de Santander"
          )
        ) {
          errorMessage =
            "Solo estamos trabajando con datos del departamento de Santander y sus principales municipios (Bucaramanga, Floridablanca, Piedecuesta, Girón). Por favor, formula tu consulta relacionada con el departamento de Santander.";
        }
      }

      setAnswerText(errorMessage);
      setLoading(false);
      setAnswer(false);
    }
  }

  function handleCancel() {
    // Stop speech recognition if it's active
    speechRecognitionRef.current?.stopListening();
    setAnswer(true);
    setPrompt("");
  }

  if (!isClient) {
    return (
      <div className="flex min-h-screen w-full flex-col bg-muted/40">
        <div className="flex flex-col sm:gap-4">
          {/* Skeleton loader for server-side rendering */}
          <div className="sticky top-0 z-30 flex h-24 items-center gap-4 border-b-2 border-primary bg-white px-6 pb-1">
            <div className="flex items-center space-x-3">
              <div className="h-[50px] w-[170px] rounded-lg bg-gray-200"></div>
              <div className="flex flex-col">
                <div className="h-6 w-32 rounded bg-gray-200"></div>
                <div className="mt-1 h-4 w-16 rounded bg-gray-200"></div>
              </div>
            </div>
          </div>
          <main className="grid flex-1 items-start gap-4 p-4 sm:px-6 sm:py-0 md:gap-8">
            <div className="grid gap-6">
              <div className="flex flex-col items-center gap-4 text-center">
                <div className="flex flex-col items-center">
                  <div className="flex items-center justify-center">
                    <div className="h-[3.25rem] w-[3.25rem] rounded bg-gray-200"></div>
                  </div>
                  <div className="mt-2">
                    <div className="h-6 w-64 rounded bg-gray-200"></div>
                  </div>
                  <div className="mt-2 w-fit">
                    <div className="h-6 w-48 rounded bg-gray-200"></div>
                  </div>
                </div>
              </div>
              <div className="h-96 rounded-lg bg-gray-100"></div>
            </div>
          </main>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen w-full flex-col bg-muted/40">
      <div className="flex flex-col sm:gap-4">
        <Header />
        <main className="flex flex-1 justify-center p-4 sm:px-6 sm:py-0">
          <div className="w-full max-w-5xl">
            <div className="grid gap-6">
              {/* Header Section */}
              <div className="flex flex-col items-center gap-4 text-center">
                <div className="flex flex-col items-center">
                  <div className="flex items-center justify-center">
                    <img
                      src="/logoGabi.png"
                      alt="GABI Logo"
                      className="h-[3.25rem] w-auto object-contain"
                    />
                  </div>
                  <div>
                    <p className="text-lg text-muted-foreground">
                      Tu asistente de orientación y seguridad en Santander
                    </p>
                  </div>
                  <div className="mt-1 w-fit">
                    <Badge variant="secondary">
                      <Sparkles className="mr-1 h-3 w-3" />
                      Conectado a Datos Abiertos Colombia
                    </Badge>
                  </div>
                </div>
              </div>

              {/* Main Chat Card */}
              <Card className="shadow-lg">
                <CardHeader className="pb-4">
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Bot className="h-4 w-4" />
                    Realiza preguntas sobre datos de criminalidad en Santander y
                    sus municipios solamente
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-4">
                    <div className="relative">
                      <Textarea
                        id="prompt"
                        value={prompt}
                        onChange={(event) => setPrompt(event.target.value)}
                        onKeyDown={(event) => {
                          if (event.key === "Enter" && !event.shiftKey) {
                            event.preventDefault();
                            if (!loading && prompt.trim()) {
                              void handleGenerateQuery();
                            }
                          }
                        }}
                        placeholder="Ejemplo: ¿Cuántos casos de hurto hubo en Bucaramanga en 2023?"
                        className="min-h-[120px] resize-none"
                        disabled={loading}
                      />
                      {prompt && (
                        <Button
                          variant="ghost"
                          size="icon"
                          className="absolute right-10 top-2 h-6 w-6"
                          onClick={() => setPrompt("")}
                        >
                          <X className="h-3 w-3" />
                        </Button>
                      )}
                    </div>

                    {loading && (
                      <div className="space-y-2">
                        <Progress value={50} className="h-2" />
                        <p className="text-center text-sm text-muted-foreground">
                          Procesando consulta...
                        </p>
                      </div>
                    )}
                  </div>
                </CardContent>
                <CardFooter className="flex justify-between">
                  <Button
                    variant="outline"
                    onClick={handleCancel}
                    disabled={loading}
                  >
                    Cancelar
                  </Button>
                  <div className="flex items-center gap-2">
                    <SpeechRecognitionButton
                      ref={speechRecognitionRef}
                      onTranscriptChange={setPrompt}
                      disabled={loading}
                    />
                    <Button
                      onClick={() => void handleGenerateQuery()}
                      disabled={loading || !prompt.trim()}
                      className="gap-2"
                    >
                      {loading ? (
                        <>
                          <div className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                          Procesando...
                        </>
                      ) : (
                        <>
                          <Send className="h-4 w-4" />
                          Consultar
                        </>
                      )}
                    </Button>
                  </div>
                </CardFooter>
              </Card>

              {/* Response Card */}
              {!answer && (
                <Card className="border-l-4 border-l-primary">
                  <CardHeader className="pb-3">
                    <CardTitle className="flex items-center gap-2">
                      <Sparkles className="h-5 w-5 text-primary" />
                      Respuesta
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="rounded-lg bg-muted/50 p-4">
                      <div className="text-balance whitespace-pre-wrap leading-relaxed [&>p]:mb-4 [&>ul]:mb-4 [&>ul]:list-disc [&>ul]:pl-6">
                        <ReactMarkdown>{answerText}</ReactMarkdown>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
