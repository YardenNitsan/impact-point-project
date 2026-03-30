const BASE_URL = process.env.BASE_URL || "http://localhost:3000";
const TOTAL = 150;
const CONCURRENCY = 25;

async function oneRequest(i) {
  const res = await fetch(`${BASE_URL}/health`);
  return res.status;
}

async function runBatch(start, size) {
  const promises = [];
  for (let i = start; i < start + size; i++) {
    promises.push(oneRequest(i));
  }
  return Promise.all(promises);
}

async function run() {
  const allStatuses = [];

  for (let i = 0; i < TOTAL; i += CONCURRENCY) {
    const batch = await runBatch(i, Math.min(CONCURRENCY, TOTAL - i));
    allStatuses.push(...batch);
  }

  const counts = {};
  for (const status of allStatuses) {
    counts[status] = (counts[status] || 0) + 1;
  }

  console.log("status counts:", counts);
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
