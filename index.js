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

    // console.log("Respuesta completa de Hugging Face:", result);

    // Verificamos que sea un array (el embedding unidimensional)
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
// ENDPOINT: INDEXAR (GUARDAR) NOTICIA + COPY EN PINECONE
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
    const upsertResponse = await index.upsert(
      [
        {
          id: Date.now().toString(), // ID único
          values: vector,
          metadata: { noticia, copy },
        },
      ]
    );

    // console.log("Indexación exitosa:", upsertResponse);
    return res.json({ message: "Noticia indexada exitosamente." });
  } catch (error) {
    console.error("Error al indexar la noticia:", error);
    return res.status(500).json({ error: "Hubo un error al indexar la noticia." });
  }
});

// ────────────────────────────────────────────────────────────────────────────────
// ENDPOINT: GENERAR COPIES
// ────────────────────────────────────────────────────────────────────────────────
app.post("/generar_copy", async (req, res) => {
  try {
    const { url } = req.body;
    if (!url) {
      return res.status(400).json({ error: "La URL es obligatoria." });
    }

    // 1️⃣ Obtener el título de la noticia
    const noticia = await getTitleFromURL(url);

    // 2️⃣ Generar embeddings del título para buscar noticias similares
    const vector = await getEmbedding(noticia);

    // 3️⃣ Buscar noticias similares en Pinecone
    const resultados = await index.namespace("").query({
      topK: 2,
      vector,
      includeMetadata: true,
    });

    // 4️⃣ Extraer los metadatos de noticias similares
    const similares = resultados.matches?.map((match) => ({
      noticia: match.metadata?.noticia || "",
      copy: match.metadata?.copy || "",
    })) || [];

    let referenciaTexto = "";
    if (similares.length > 0) {
      referenciaTexto = `Aquí hay algunos ejemplos de noticias anteriores con sus copies:\n` +
        similares
          .map(
            (s, i) => `Ejemplo ${i + 1}:\n   Título: "${s.noticia}"\n   Copy: "${s.copy}"`
          )
          .join("\n");
    }

    // 5️⃣ Crear los prompts para cada copy
    const fbPrompt = `
      Genera un título llamativo para esta noticia. 
      Debe ser informativo, breve (máx. 10 palabras) y con 1-2 emojis si es adecuado.
      ${referenciaTexto}
      Noticia: "${noticia}"
    `;

    const twitterPrompt = `
      Genera un título breve (máx. 10 palabras) para esta noticia, 
      con tono directo y un emoji al final.
      ${referenciaTexto}
      Noticia: "${noticia}"
    `;

    const wppPrompt = `
      Genera un copy corto con TÍTULO (máx. 10 palabras) y un párrafo breve. 
      Usa 1-2 emojis si es apropiado.
      ${referenciaTexto}
      Noticia: "${noticia}"
    `;

    // 6️⃣ Llamadas a la API de Gemini en paralelo
    const [fbCopy, twitterCopy, wppCopy] = await Promise.all([
      model.generateContent([fbPrompt]),
      model.generateContent([twitterPrompt]),
      model.generateContent([wppPrompt]),
    ]);

    // 7️⃣ Obtener el texto generado (Gemini retorna un objeto con candidates)
    const extractText = (response) =>
      response.candidates?.[0]?.content?.parts?.[0]?.text || "Texto no disponible";

    const fbText = extractText(fbCopy);
    const twitterText = extractText(twitterCopy);
    const wppText = extractText(wppCopy);

    // 8️⃣ Enviamos como respuesta los tres copies
    res.json({
      facebook: fbText.trim(),
      twitter: twitterText.trim(),
      wpp: wppText.trim(),
    });
  } catch (error) {
    console.error("Error al generar los copys:", error);
    res.status(500).json({ error: "Hubo un error al generar los copys." });
  }
});

// NUEVO ENDPOINT: BÚSQUEDA DE LA NOTICIA/COPY MÁS SIMILAR
app.post("/buscar_similar", async (req, res) => {
  try {
    const { texto } = req.body;
    if (!texto) {
      return res.status(400).json({ error: "Se requiere un campo 'texto' para buscar similitud." });
    }

    // 1) Generar embedding de la consulta
    const vector = await getEmbedding(texto);

    // 2) Consultar Pinecone
    const results = await index.namespace("").query({
      topK: 1,              // solo queremos el más parecido
      vector,               // embedding de la consulta
      includeMetadata: true // para obtener la noticia y el copy del match
    });

    // 3) Verificar si hay resultados
    if (!results.matches || results.matches.length === 0) {
      return res.status(404).json({ error: "No se encontraron coincidencias similares." });
    }

    // 4) Tomar el primer resultado (más parecido)
    const bestMatch = results.matches[0];
    // Podrías devolver score, ID y metadata
    return res.json({
      id: bestMatch.id,
      score: bestMatch.score,         // similitud o distancia, depende de la configuración de tu índice
      metadata: bestMatch.metadata    // { noticia, copy }
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
app.listen(PORT);
