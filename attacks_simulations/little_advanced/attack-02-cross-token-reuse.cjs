const BASE_URL = process.env.BASE_URL || "http://localhost:3000";

const A_ID = process.env.A_ID;
const A_TOKEN = process.env.A_TOKEN;
const B_ID = process.env.B_ID;

if (!A_ID || !A_TOKEN || !B_ID) {
  console.error("Missing A_ID, A_TOKEN, or B_ID");
  process.exit(1);
}

async function run() {
  const attempts = [
    {
      label: "DETAILS B with token A",
      url: `${BASE_URL}/api/simulation/${B_ID}/details`,
      init: { headers: { "x-simulation-token": A_TOKEN } },
    },
    {
      label: "WATCH B with token A",
      url: `${BASE_URL}/api/simulation/${B_ID}`,
      init: { headers: { "x-simulation-token": A_TOKEN } },
    },
    {
      label: "DELETE B with token A",
      url: `${BASE_URL}/api/simulation/${B_ID}`,
      init: {
        method: "DELETE",
        headers: { "x-simulation-token": A_TOKEN },
      },
    },
  ];

  for (const attempt of attempts) {
    const res = await fetch(attempt.url, attempt.init);
    const body = await res.text();

    console.log(`\n${attempt.label}`);
    console.log("status:", res.status);
    console.log("body:", body);
    console.log(
      res.status === 403 ? "RESULT: PROTECTED" : "RESULT: VULNERABLE",
    );
  }
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
