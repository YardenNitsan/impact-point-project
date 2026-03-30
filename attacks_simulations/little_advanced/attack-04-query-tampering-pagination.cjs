const BASE_URL = process.env.BASE_URL || "http://localhost:3000";

async function test(url) {
  const res = await fetch(url);
  const text = await res.text();

  console.log(`\nURL: ${url}`);
  console.log("status:", res.status);
  console.log("x-page:", res.headers.get("x-page"));
  console.log("x-limit:", res.headers.get("x-limit"));
  console.log("x-total-count:", res.headers.get("x-total-count"));
  console.log("x-total-pages:", res.headers.get("x-total-pages"));

  try {
    const body = JSON.parse(text);
    console.log(
      "items returned:",
      Array.isArray(body) ? body.length : "not-array",
    );
    console.log("body preview:", JSON.stringify(body).slice(0, 300));
  } catch {
    console.log("raw body:", text.slice(0, 300));
  }
}

async function run() {
  await test(`${BASE_URL}/api/simulation`);
  await test(`${BASE_URL}/api/simulation?page=1&limit=5`);
  await test(`${BASE_URL}/api/simulation?page=1&limit=20`);
  await test(`${BASE_URL}/api/simulation?page=1&limit=100000`);
  await test(`${BASE_URL}/api/simulation?page=1&limit=-1`);
  await test(`${BASE_URL}/api/simulation?page=abc&limit=5`);
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
