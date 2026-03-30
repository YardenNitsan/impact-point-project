const BASE_URL = process.env.BASE_URL || "http://localhost:3000";

async function run() {
  const listRes = await fetch(`${BASE_URL}/api/simulation`);
  const listText = await listRes.text();

  console.log("LIST status:", listRes.status);
  console.log("LIST body:", listText);

  let sims;
  try {
    sims = JSON.parse(listText);
  } catch {
    console.log("Could not parse list response");
    return;
  }

  if (!Array.isArray(sims) || sims.length === 0) {
    console.log("No simulations found");
    return;
  }

  const id = sims[0].id;
  console.log("\nUsing id:", id);

  const detailsRes = await fetch(`${BASE_URL}/api/simulation/${id}/details`);
  console.log("DETAILS status:", detailsRes.status);
  console.log("DETAILS body:", await detailsRes.text());

  // Be careful: destructive
  // const deleteRes = await fetch(`${BASE_URL}/api/simulation/${id}`, { method: "DELETE" });
  // console.log("DELETE status:", deleteRes.status);
  // console.log("DELETE body:", await deleteRes.text());
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
