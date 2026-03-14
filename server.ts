import express from "express";
import path from "path";
import { fileURLToPath } from "url";
import fs from "fs-extra";
import csv from "csv-parser";
import * as tf from "@tensorflow/tfjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let model: tf.Sequential;

async function trainModel() {
  console.log("Starting model training...");
  const dataPath = path.resolve(__dirname, 'data', 'insurance_claims.csv');
  
  if (!fs.existsSync(dataPath)) {
    console.log(`No dataset found at ${dataPath}. Run 'npm run generate-data' first.`);
    return;
  }

  const claims: any[] = [];
  return new Promise<void>((resolve, reject) => {
    fs.createReadStream(dataPath)
      .pipe(csv())
      .on("data", (data) => claims.push(data))
      .on("end", async () => {
        const inputs = claims.map(c => [
          parseFloat(c.claim_amount) / 10000,
          parseFloat(c.patient_age) / 80,
          parseFloat(c.severity_index) / 10,
          parseFloat(c.category) / 5
        ]);
        const labels = claims.map(c => parseInt(c.is_fraud));

        const xs = tf.tensor2d(inputs);
        const ys = tf.tensor2d(labels, [labels.length, 1]);

        model = tf.sequential();
        model.add(tf.layers.dense({ units: 8, inputShape: [4], activation: 'relu' }));
        model.add(tf.layers.dense({ units: 4, activation: 'relu' }));
        model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

        model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });

        await model.fit(xs, ys, {
          epochs: 20,
          verbose: 0
        });

        console.log("Model training complete.");
        resolve();
      });
  });
}

const app = express();
const PORT = 3000;

app.use(express.json());

// Trigger training
let trainingPromise = trainModel().catch(console.error);

// API Routes
app.get("/api/health", (req, res) => {
  res.json({ status: "ok" });
});

app.get("/api/medicines", (req, res) => {
  res.json([
    { name: "Amoxicillin", price: 15.50 },
    { name: "Lisinopril", price: 12.00 },
    { name: "Atorvastatin", price: 25.00 },
  ]);
});

app.post("/api/predict-fraud", async (req, res) => {
  const { claim_amount, patient_age, severity_index, category } = req.body;
  
  // Ensure training is finished before predicting (important for serverless cold starts)
  if (!model) {
    console.log("Model not ready, waiting for training...");
    await trainingPromise;
  }

  if (!model) {
    return res.status(503).json({ error: "Model training failed or is still in progress" });
  }

  const input = tf.tensor2d([[
    claim_amount / 10000,
    patient_age / 80,
    severity_index / 10,
    category / 5
  ]]);

  const prediction = model.predict(input) as tf.Tensor;
  const score = (await prediction.data())[0];

  res.json({
    is_fraud: score > 0.5,
    confidence: score
  });
});

// Vite middleware for development
if (process.env.NODE_ENV !== "production" && !process.env.VERCEL) {
  const { createServer } = await import("vite");
  const vite = await createServer({
    server: { middlewareMode: true },
    appType: "spa",
  });
  app.use(vite.middlewares);
} else {
  const distPath = path.join(process.cwd(), 'dist');
  app.use(express.static(distPath));
  // In serverless, we usually rely on Vercel's routes for the SPA, 
  // but keeping this for local prod testing.
  app.get(/^(?!\/api).*/, (req, res) => {
    res.sendFile(path.join(distPath, 'index.html'));
  });
}

// Only listen if this file is run directly (local dev)
if (process.env.NODE_ENV !== "production" || !process.env.VERCEL) {
  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

export default app;
