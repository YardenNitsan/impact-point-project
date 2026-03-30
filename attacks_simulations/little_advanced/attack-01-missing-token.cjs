const BASE_URL = process.env.BASE_URL || "http://localhost:3000";

async function run() {
  const listRes = await fetch(`${BASE_URL}/api/simulation`);
  const listBody = await listRes.json().catch(() => null);

  console.log("LIST status:", listRes.status);
  if (!Array.isArray(listBody) || listBody.length === 0) {
    console.log("No simulations found. Create one first and rerun.");
    return;
  }

  const id = listBody[0].id;
  console.log("Target simulation id:", id);

  const attempts = [
    {
      label: "DETAILS without token",
      url: `${BASE_URL}/api/simulation/${id}/details`,
      init: {},
    },
    {
      label: "WATCH without token",
      url: `${BASE_URL}/api/simulation/${id}`,
      init: {},
    },
    {
      label: "DELETE without token",
      url: `${BASE_URL}/api/simulation/${id}`,
      init: { method: "DELETE" },
    },
  ];

  for (const attempt of attempts) {
    const res = await fetch(attempt.url, attempt.init);
    const body = await res.text();
    console.log(`\n${attempt.label}`);
    console.log("status:", res.status);
    console.log("body:", body);
    console.log(
      res.status === 401 || res.status === 403
        ? "RESULT: PROTECTED"
        : "RESULT: VULNERABLE",
    );
  }
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
