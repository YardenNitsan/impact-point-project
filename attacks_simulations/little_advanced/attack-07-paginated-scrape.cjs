const BASE_URL = process.env.BASE_URL || "http://localhost:3000";
const LIMIT = 20;
const MAX_PAGES = 10;

async function run() {
  let totalItems = 0;

  for (let page = 1; page <= MAX_PAGES; page++) {
    const res = await fetch(
      `${BASE_URL}/api/simulation?page=${page}&limit=${LIMIT}`,
    );
    const body = await res.json().catch(() => null);

    console.log(`\nPAGE ${page}`);
    console.log("status:", res.status);
    console.log("x-page:", res.headers.get("x-page"));
    console.log("x-limit:", res.headers.get("x-limit"));
    console.log("x-total-count:", res.headers.get("x-total-count"));
    console.log("x-total-pages:", res.headers.get("x-total-pages"));

    if (!Array.isArray(body)) {
      console.log("Non-array body:", body);
      break;
    }

    console.log("items returned:", body.length);
    totalItems += body.length;

    if (body.length === 0) {
      console.log("Reached empty page, stopping.");
      break;
    }
  }

  console.log(`\nTotal summary items scraped: ${totalItems}`);
  console.log(
    "RESULT: This shows what an attacker can still scrape if history summaries stay public.",
  );
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
