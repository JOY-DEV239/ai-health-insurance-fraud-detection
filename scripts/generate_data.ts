import fs from 'fs-extra';
import path from 'path';

const DATA_DIR = path.join(process.cwd(), 'data');
const DATA_FILE = path.join(DATA_DIR, 'insurance_claims.csv');

async function generateData() {
  await fs.ensureDir(DATA_DIR);

  const headers = 'claim_amount,patient_age,severity_index,category,is_fraud\n';
  let content = headers;

  for (let i = 0; i < 1000; i++) {
    const amount = Math.floor(Math.random() * 10000) + 500;
    const age = Math.floor(Math.random() * 60) + 18;
    const severity = Math.floor(Math.random() * 10) + 1;
    const category = Math.floor(Math.random() * 5) + 1;
    
    // Simple heuristic for fraud simulation: 
    // High amount + low severity + specific category = higher chance of fraud
    let isFraud = 0;
    if (amount > 8000 && severity < 3 && Math.random() > 0.5) {
      isFraud = 1;
    } else if (Math.random() > 0.95) {
      isFraud = 1; // Random noise
    }

    content += `${amount},${age},${severity},${category},${isFraud}\n`;
  }

  await fs.writeFile(DATA_FILE, content);
  console.log(`Generated synthetic dataset at ${DATA_FILE}`);
}

generateData().catch(console.error);
