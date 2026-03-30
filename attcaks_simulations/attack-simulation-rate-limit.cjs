const BASE_URL = process.env.BASE_URL || "http://localhost:3000";
const TOTAL = 40;
const CONCURRENCY = 10;

const invalidBody = {
  lat: "bad",
  lon: 34.8,
  alt: 1000,
  azimuth: 120,
  elevation: 30,
  mass: 50,
  initialSpeed: 300,
  weather_source: "machine",
};

async function oneRequest() {
  const res = await fetch(`${BASE_URL}/api/simulation`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(invalidBody),
  });

  return {
    status: res.status,
    body: await res.text(),
  };
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
  for (const result of results) {
    counts[result.status] = (counts[result.status] || 0) + 1;
  }

  console.log("status counts:", counts);

  const sample429 = results.find((x) => x.status === 429);
  if (sample429) {
    console.log("\nsample 429 response:");
    console.log(sample429.body);
  }
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
