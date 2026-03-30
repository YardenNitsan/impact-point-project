const BASE_URL = process.env.BASE_URL || "http://localhost:3000";
const TOTAL = 80;
const CONCURRENCY = 10;

async function oneRequest() {
  const res = await fetch(`${BASE_URL}/api/simulation`);
  const text = await res.text();
  return { status: res.status, bytes: text.length };
}

async function runBatch(size) {
  return Promise.all(Array.from({ length: size }, () => oneRequest()));
}

async function run() {
  const results = [];

  for (let i = 0; i < TOTAL; i += CONCURRENCY) {
    const batch = await runBatch(Math.min(CONCURRENCY, TOTAL - i));
    results.push(...batch);
  }

  const counts = {};
  let totalBytes = 0;

  for (const r of results) {
    counts[r.status] = (counts[r.status] || 0) + 1;
    totalBytes += r.bytes;
  }

  console.log("status counts:", counts);
  console.log("total response bytes:", totalBytes);
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
