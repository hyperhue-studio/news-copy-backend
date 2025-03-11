require("dotenv").config();
const express = require("express");
const cors = require("cors");
const fetch = require("node-fetch"); // Usado para scraping
const { Pinecone } = require("@pinecone-database/pinecone");
const { GoogleGenerativeAI } = require("@google/generative-ai");
const cheerio = require("cheerio");
const { HfInference } = require("@huggingface/inference");

// ────────────────────────────────────────────────────────────────────────────────
// CONFIGURACIÓN DEL SERVIDOR
// ────────────────────────────────────────────────────────────────────────────────
const app = express();
app.use(express.json());
app.use(cors());

// ────────────────────────────────────────────────────────────────────────────────
// API KEYS Y CONFIGURACIONES
// ────────────────────────────────────────────────────────────────────────────────
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const HUGGINGFACE_API_KEY = process.env.HUGGINGFACE_API_KEY;

// ────────────────────────────────────────────────────────────────────────────────
// INICIALIZAR PINECONE
// ────────────────────────────────────────────────────────────────────────────────
const pinecone = new Pinecone({ apiKey: PINECONE_API_KEY });
const index = pinecone.index(PINECONE_INDEX_NAME);

// ────────────────────────────────────────────────────────────────────────────────
// INICIALIZAR GOOGLE GENERATIVE AI (GEMINI)
// ────────────────────────────────────────────────────────────────────────────────
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

// ────────────────────────────────────────────────────────────────────────────────
// INSTANCIA DE HfInference
// ────────────────────────────────────────────────────────────────────────────────
const hf = new HfInference(HUGGINGFACE_API_KEY);

// ────────────────────────────────────────────────────────────────────────────────
// FUNCIÓN PARA OBTENER EMBEDDING DESDE Hugging Face Inference API
// ────────────────────────────────────────────────────────────────────────────────
async function getEmbedding(text) {
  try {
    const result = await hf.featureExtraction({
      model: "intfloat/multilingual-e5-large",
      inputs: text,
    });

    if (!Array.isArray(result)) {
      throw new Error("La respuesta de Hugging Face no es el embedding esperado.");
    }
    return result;
  } catch (error) {
    console.error("Error obteniendo embeddings (Hugging Face):", error);
    throw new Error("Error generando embeddings.");
  }
}

// ────────────────────────────────────────────────────────────────────────────────
// FUNCIÓN PARA OBTENER TÍTULO DE LA NOTICIA (SCRAPING)
// ────────────────────────────────────────────────────────────────────────────────
async function getTitleFromURL(url) {
  try {
    const response = await fetch(url);
    const html = await response.text();
    const $ = cheerio.load(html);

    // Buscamos un meta og:title o, en su defecto, la etiqueta <title>
    const title = $('meta[property="og:title"]').attr("content") || $("title").text();
    if (!title) throw new Error("No se encontró título");

    return title.trim();
  } catch (error) {
    console.error("Error obteniendo título:", error);
    throw new Error("No se pudo extraer el título de la noticia.");
  }
}

// ────────────────────────────────────────────────────────────────────────────────
// ENDPOINT: INDEXAR (GUARDAR) NOTICIA + COPY EN PINECONE (COPY DE FACEBOOK)
// ────────────────────────────────────────────────────────────────────────────────
app.post("/indexar", async (req, res) => {
  try {
    const { noticia, copy } = req.body;
    if (!noticia || !copy) {
      return res.status(400).json({ error: "La 'noticia' y el 'copy' son obligatorios." });
    }

    // 1) Generar embeddings del texto de la noticia
    const vector = await getEmbedding(noticia);

    // 2) Subir a Pinecone
    await index.upsert([
      {
        id: Date.now().toString(), // ID único
        values: vector,
        // Guardamos la noticia + copy. Este copy se asume que es de Facebook
        // (Así lo reutilizamos en el RAG)
        metadata: { noticia, copy },
      },
    ]);

    return res.json({ message: "Noticia indexada exitosamente." });
  } catch (error) {
    console.error("Error al indexar la noticia:", error);
    return res.status(500).json({ error: "Hubo un error al indexar la noticia." });
  }
});

// ────────────────────────────────────────────────────────────────────────────────
// NUEVO ENDPOINT: GENERAR COPY USANDO RAG
// ────────────────────────────────────────────────────────────────────────────────
app.post("/generar_copy_rag", async (req, res) => {
  try {
    const { url } = req.body;
    if (!url) {
      return res.status(400).json({ error: "La URL es obligatoria." });
    }

    // 1) Scraping: obtener título de la noticia
    const titulo = await getTitleFromURL(url);

    // 2) Generar embedding del título
    const vector = await getEmbedding(titulo);

    // 3) Buscar topK=3 en Pinecone para obtener los 3 copies de FB más similares
    const results = await index.namespace("").query({
      topK: 3,
      vector,
      includeMetadata: true,
    });

    const similares = results.matches?.map((match) => ({
      noticia: match.metadata?.noticia || "",
      copy: match.metadata?.copy || "", // copy de Facebook
    })) || [];

    // 4) Construir references para Facebook
    let referencesFb = "";
    if (similares.length > 0) {
      referencesFb = `Aquí hay algunos copies de Facebook anteriores (3 más similares):\n`;
      referencesFb += similares
        .map((s, i) => `Ejemplo ${i + 1}: \n  Título: "${s.noticia}"\n  FB Copy: "${s.copy}"`)
        .join("\n\n");
    }

    // 5) Prompt para Facebook con RAG
    const fbPrompt = `
      Basándote en estos copies de Facebook (ejemplos), crea un copy de Facebook
      para la siguiente noticia. El copy debe ser breve, informativo y puedes usar
      1-2 emojis si es adecuado. Máximo 2 líneas.

      ${referencesFb}

      Título de la noticia actual: "${titulo}"
    `;

    // 6) Prompt para Twitter y Wpp (sin RAG, puro prompt)
    const twitterPrompt = `
      Genera un tweet breve (máx. 10 palabras) para esta noticia, 
      con tono directo y un emoji al final.
      Noticia: "${titulo}"
    `;

    const wppPrompt = `
      Genera un mensaje corto para WhatsApp con TÍTULO (máx. 10 palabras) y 
      un párrafo breve. Usa 1-2 emojis si es apropiado.
      Noticia: "${titulo}"
    `;

    // 7) Llamamos a Gemini
    const [fbResp, twResp, wppResp] = await Promise.all([
      model.generateContent([fbPrompt]),
      model.generateContent([twitterPrompt]),
      model.generateContent([wppPrompt]),
    ]);

    const extractText = (response) =>
      response.candidates?.[0]?.content?.parts?.[0]?.text || "Texto no disponible";

    const facebookCopy = extractText(fbResp).trim();
    const twitterCopy = extractText(twResp).trim();
    const wppCopy = extractText(wppResp).trim();

    return res.json({
      facebook: facebookCopy,
      twitter: twitterCopy,
      wpp: wppCopy
    });
  } catch (error) {
    console.error("Error al generar copy (RAG):", error);
    res.status(500).json({ error: "Hubo un error al generar el copy con RAG." });
  }
});

// ────────────────────────────────────────────────────────────────────────────────
// ENDPOINT VIEJO (Opcional): /generar_copy - si ya no lo usas, puedes borrarlo
// ────────────────────────────────────────────────────────────────────────────────
app.post("/generar_copy", async (req, res) => {
  try {
    const { url } = req.body;
    if (!url) {
      return res.status(400).json({ error: "La URL es obligatoria." });
    }
    // ... Lógica anterior ...
    return res.json({ /* ... facebook, twitter, wpp ... */ });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Error" });
  }
});

// ────────────────────────────────────────────────────────────────────────────────
// ENDPOINT: BÚSQUEDA SIMILAR (opcional, para pruebas o debug)
// ────────────────────────────────────────────────────────────────────────────────
app.post("/buscar_similar", async (req, res) => {
  try {
    const { texto } = req.body;
    if (!texto) {
      return res.status(400).json({ error: "Se requiere un campo 'texto' para buscar similitud." });
    }

    const vector = await getEmbedding(texto);
    const results = await index.namespace("").query({
      topK: 1,
      vector,
      includeMetadata: true
    });

    if (!results.matches || results.matches.length === 0) {
      return res.status(404).json({ error: "No se encontraron coincidencias similares." });
    }

    const bestMatch = results.matches[0];
    return res.json({
      id: bestMatch.id,
      score: bestMatch.score,
      metadata: bestMatch.metadata
    });
  } catch (error) {
    console.error("Error al buscar similar:", error);
    res.status(500).json({ error: "Hubo un error al buscar la similitud." });
  }
});

// ────────────────────────────────────────────────────────────────────────────────
// INICIAR SERVIDOR
// ────────────────────────────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`Backend corriendo en puerto ${PORT}`);
});
