require("dotenv").config();
const express = require("express");
const cors = require("cors");
const fetch = require("node-fetch");  // Puedes seguir usando fetch para scraping o cambiarlo a axios
const { Pinecone } = require("@pinecone-database/pinecone");
const { GoogleGenerativeAI } = require("@google/generative-ai");
const cheerio = require("cheerio");

// 1) Importar la librería oficial de Hugging Face
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
// Escoge tu modelo, p. ej. "gemini-1.5-flash"
const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

// 2) Crear instancia de HfInference con tu token
const hf = new HfInference(HUGGINGFACE_API_KEY);

// ────────────────────────────────────────────────────────────────────────────────
// FUNCIÓN PARA OBTENER EMBEDDING DESDE Hugging Face Inference API
// ────────────────────────────────────────────────────────────────────────────────
async function getEmbedding(text) {
  try {
    // 3) Llamar a featureExtraction() con tu modelo preferido
    // Por ejemplo: "intfloat/multilingual-e5-large"
    const result = await hf.featureExtraction({
      model: "intfloat/multilingual-e5-large",
      inputs: text,
    });
    // Normalmente 'result' será un array de arrays (batch).
    // Para un único texto, asume que es [ [embedding] ].
    // A veces ya te devuelve [embedding]. Revisa console.log si dudas.
    if (!Array.isArray(result) || !Array.isArray(result[0])) {
      throw new Error("La respuesta de Hugging Face no es el embedding esperado.");
    }
    return result[0]; // Devolvemos el vector
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
    const upsertResponse = await index.upsert({
      vectors: [
        {
          id: Date.now().toString(), // ID único
          values: vector,
          metadata: { noticia, copy },
        },
      ],
    });

    console.log("Indexación exitosa:", upsertResponse);
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
    //    Ajusta namespace("") o quita la propiedad si usas el namespace por defecto
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

    // Si Pinecone no encuentra nada, 'similares' estará vacío
    let referenciaTexto = "";
    if (similares.length > 0) {
      referenciaTexto = `Aquí hay algunos ejemplos de noticias anteriores con sus copies:\n` +
        similares
          .map(
            (s, i) =>
              `Ejemplo ${i + 1}:\n   Título: "${s.noticia}"\n   Copy: "${s.copy}"`
          )
          .join("\n");
    }

    // 5️⃣ Crear los prompts para cada copy
    //    Ajusta las instrucciones según la longitud y el tono que quieras.
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
    const extractText = (response) => {
      return response.candidates?.[0]?.content?.parts?.[0]?.text || "Texto no disponible";
    };

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

// ────────────────────────────────────────────────────────────────────────────────
// INICIAR SERVIDOR
// ────────────────────────────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`🚀 Backend corriendo en http://localhost:${PORT}`));
