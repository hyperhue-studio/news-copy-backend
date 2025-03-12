require("dotenv").config();
const express = require("express");
const cors = require("cors");
const fetch = require("node-fetch");
const { Pinecone } = require("@pinecone-database/pinecone");
const { GoogleGenerativeAI } = require("@google/generative-ai");
const cheerio = require("cheerio");
const { HfInference } = require("@huggingface/inference");

const app = express();
app.use(express.json());
app.use(cors());

const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const HUGGINGFACE_API_KEY = process.env.HUGGINGFACE_API_KEY;
const BITLY_TOKEN = process.env.BITLY_TOKEN; // Para acortar URLs

// Inicializar Pinecone
const pinecone = new Pinecone({ apiKey: PINECONE_API_KEY });
const index = pinecone.index(PINECONE_INDEX_NAME);

// Inicializar Google Generative AI
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

// HfInference para embeddings
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
async function getTitleAndDescription(url) {
  // Scrape para obtener og:title y og:description
  try {
    const response = await fetch(url);
    const html = await response.text();
    const $ = cheerio.load(html);

    const title = $('meta[property="og:title"]').attr("content");
    const description = $('meta[property="og:description"]').attr("content");

    if (!title || !description) {
      throw new Error("No se encontraron los meta tags adecuados (og:title, og:description).");
    }

    return { title: title.trim(), description: description.trim() };
  } catch (error) {
    console.error("Error obteniendo título/descripción:", error);
    throw new Error("No se pudo extraer los meta tags de la noticia.");
  }
}

// Función para acortar la URL usando Bitly (similar a tu versión anterior)
async function shortenUrl(url) {
  try {
    const longUrlWithUtm = `${url}?utm_source=whatsapp&utm_medium=social&utm_campaign=canal`;
    console.log("URL antes de acortar:", longUrlWithUtm);

    const resp = await fetch("https://api-ssl.bitly.com/v4/shorten", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${BITLY_TOKEN}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ long_url: longUrlWithUtm }),
    });

    if (!resp.ok) {
      const errorData = await resp.json();
      console.error("Error en la solicitud de acortamiento:", errorData);
      throw new Error("Error al acortar la URL.");
    }

    const data = await resp.json();
    console.log("URL acortada:", data.link);
    return data.link;
  } catch (error) {
    console.error("Error al acortar la URL:", error);
    throw new Error("Hubo un error al acortar la URL.");
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
// ENDPOINT: GENERAR COPIES (VERSIÓN RAG PARA FB, LÓGICA ORIGINAL PARA TW/WPP)
// ────────────────────────────────────────────────────────────────────────────────
app.post("/generate-copies", async (req, res) => {
  try {
    const { url } = req.body;
    console.log("[generate-copies] URL recibida:", url);

    if (!url) {
      return res.status(400).json({ error: "La URL es obligatoria." });
    }

    // 1. Obtener title + description via scraping
    const { title, description } = await getTitleAndDescription(url);
    console.log("[generate-copies] Título final:", title);
    console.log("[generate-copies] Descripción final:", description);

    // 2. Preparar combinedText para Twitter/Wpp
    const combinedText = `${title}. ${description}`;
    console.log("[generate-copies] combinedText:", combinedText);

    // 3. RAG SOLO PARA FACEBOOK
    console.log("[generate-copies] Generando embedding SOLO del título para RAG FB...");
    const fbVector = await getEmbedding(title);

    console.log("[generate-copies] Buscando topK=3 en Pinecone...");
    const results = await index.namespace("").query({
      topK: 3,
      vector: fbVector,
      includeMetadata: true,
    });

    console.log("[generate-copies] Pinecone results:", JSON.stringify(results, null, 2));
    const similares = results.matches?.map((match) => ({
      copy: match.metadata?.copy || "",
    })) || [];
    console.log("[generate-copies] Copias similares FB:", similares);

    // 3.3 Construir references (SOLO los copies)
    let referencesFb = "";
    if (similares.length > 0) {
      referencesFb = `Aquí hay algunos ejemplos de Facebook copies anteriores:\n\n` +
        similares
          .map(
            (s, i) => `Ejemplo ${i + 1}: "${s.copy}"`
          )
          .join("\n\n");
    }
    console.log("[generate-copies] referencesFb:\n", referencesFb);

    // 4. Generar FB copy con RAG
    const fbPrompt = `
    Basándote en estos copies de Facebook (ejemplos), genera un título sobre la noticia que mencionaré al final.
    Debe ser informativo y con un tono directo, pero también carismático y llamativo.
    Breve y al grano, idealmente no más de 10 palabras. Incluye de 1 a 2 emojis
    (pueden ir al principio, en medio, al final, o mixto, pero que sean respetuosos si el tema es sensible).
    No omitas datos importantes. Toma en cuenta para contexto que la noticia es de El Heraldo de Chihuahua, por lo que puede que sea o no noticia local.
    No respondas nada más que el copy que generarás.
    
    ${referencesFb}
    
    Este es el título de la noticia actual para generarle el copy: "${title}"
    `;
    console.log("[generate-copies] FB Prompt:\n", fbPrompt);

    const fbResp = await model.generateContent([fbPrompt]);
    console.log("[generate-copies] fbResp completo:", JSON.stringify(fbResp, null, 2));
    const facebookCopyRaw = fbResp.response?.candidates?.[0]?.content?.parts?.[0]?.text || "Texto no disponible";

    console.log("[generate-copies] facebookCopyRaw:", facebookCopyRaw);
    const facebookCopy = facebookCopyRaw.trim();

    // 5. Generar TWITTER copy (LÓGICA ORIGINAL)
    console.log("[generate-copies] Generando TWITTER copy...");
    const twitterPrompt = `
    Genera un copy sobre la siguiente noticia, debe ser informativo y con un tono directo,
    un solo emoji al final. Conciso y al grano, idealmente no más de 10 palabras ya que es para un tweet.
    No omitas datos importantes. Toma en cuenta para contexto que la noticia es de El Heraldo de Chihuahua, por lo que puede que sea o no noticia local.
    No respondas nada más que el copy: "${combinedText}"
    `;
    console.log("[generate-copies] twitterPrompt:\n", twitterPrompt);

    const twResp = await model.generateContent([twitterPrompt]);
    console.log("[generate-copies] twResp completo:", JSON.stringify(twResp, null, 2));
    let twitterText = twResp.response?.candidates?.[0]?.content?.parts?.[0]?.text || "Texto no disponible";

    console.log("[generate-copies] twitterText raw:", twitterText);
    twitterText = twitterText.trim();
    const twitterCopyFinal = `${twitterText}\n${url}`;

    // 6. Generar WPP copy (LÓGICA ORIGINAL)
    console.log("[generate-copies] Generando WPP copy...");
    const wppPrompt = `
    Genera un copy corto para la siguiente noticia. 
    Debe tener un título muy corto (no más de 10 palabras) seguido de un párrafo de máximo 2 renglones 
    describiendo un poco la noticia. Debe ser informativo y con un tono directo, 
    pero también carismático y llamativo (en caso de que la noticia no sea sensible).
    No omitas datos importantes. Toma en cuenta para contexto que la noticia es de El Heraldo de Chihuahua, por lo que puede que sea o no noticia local.
    Incluye de 1 a 2 emojis (para título y párrafo, pero que sean respetuosos si es tema sensible). 
    No respondas nada más que el copy: "${combinedText}"
    `;
    console.log("[generate-copies] wppPrompt:\n", wppPrompt);

    const wppResp = await model.generateContent([wppPrompt]);
    console.log("[generate-copies] wppResp completo:", JSON.stringify(wppResp, null, 2));
    let wppText =
      wppResp.response?.candidates?.[0]?.content?.parts?.[0]?.text ||
      "Texto no disponible";
    console.log("[generate-copies] wppText raw:", wppText);
    wppText = wppText.trim();

    // 6.1 Acortar la URL para WPP
    console.log("[generate-copies] Acortando URL para WPP...");
    const shortUrl = await shortenUrl(url);
    console.log("[generate-copies] shortUrl:", shortUrl);

    // 6.2 Concatenar la URL acortada al final
    const wppCopyFinal = `${wppText} Lee más aquí👉 ${shortUrl}`;

    // 7. Responder con los tres copies
    console.log("[generate-copies] facebookCopy final:", facebookCopy);
    console.log("[generate-copies] twitterCopy final:", twitterCopyFinal);
    console.log("[generate-copies] wppCopy final:", wppCopyFinal);

    return res.json({
  facebook: facebookCopy,
  twitter: twitterCopyFinal,
  wpp: wppCopyFinal,
  foundCopies: similares,  // <= agregas esta propiedad
});
  } catch (error) {
    console.error("[generate-copies] Error al generar los copys:", error);
    return res.status(500).json({ error: "Hubo un error al generar los copys." });
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
